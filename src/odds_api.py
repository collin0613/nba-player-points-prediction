"""SportsGameOdds v2 client and strict individual full-game points parser.

Official contract: https://sportsgameodds.com/openapi.json
No secrets in URLs, caches, exceptions or logs. No automatic quota-burning retries.
"""
import hashlib
import json
import os
import re
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import requests
from src.market_mapping import aware_datetime

BOOKMAKERS = ('draftkings', 'fanduel')


class OddsAPIError(RuntimeError):
    pass


class SportsGameOddsClient:
    url = 'https://api.sportsgameodds.com/v2/events'

    def __init__(self, api_key=None, cache_dir=None, cache_minutes=10, session=None):
        self.api_key = api_key or os.getenv('SPORTSGAMEODDS_API_KEY')
        if not self.api_key:
            raise OddsAPIError('Set SPORTSGAMEODDS_API_KEY in .env')
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_minutes = cache_minutes
        self.session = session or requests.Session()
        self.last_request = 0

    def page(self, params, refresh=False, resource="events"):
        if resource not in ("events", "players"):
            raise ValueError("Unsupported SportsGameOdds resource")
        cache_params = params if resource == 'events' else {'resource': resource, 'params': params}
        key = hashlib.sha256(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f'{key}.json' if self.cache_dir else None
        if path and path.exists() and not refresh:
            try:
                saved = json.loads(path.read_text())
                age = (datetime.now(timezone.utc) - aware_datetime(saved['fetched_at'])).total_seconds()
                if 0 <= age < self.cache_minutes * 60:
                    return saved
            except (ValueError, KeyError):
                pass
        # Free-plan request rate is 10/minute. Spacing also applies between pages.
        delay = 6.1 - (time.monotonic() - self.last_request)
        if delay > 0:
            time.sleep(delay)
        try:
            self.last_request = time.monotonic()
            response = self.session.get(self.url if resource == 'events' else self.url.replace('/events', '/players'), headers={'x-api-key': self.api_key}, params=params, timeout=30)
        except requests.RequestException:
            raise OddsAPIError('SportsGameOdds network request failed; retry later') from None
        if response.status_code != 200:
            reason = {401: 'invalid or missing API key', 403: 'key or plan access denied',
                      429: 'rate/quota limit reached; no automatic retry'}.get(response.status_code, 'provider request failed')
            raise OddsAPIError(f'SportsGameOdds HTTP {response.status_code}: {reason}')
        try:
            data = response.json()
        except ValueError:
            raise OddsAPIError('SportsGameOdds returned invalid JSON') from None
        if (not isinstance(data, dict) or data.get('success') is False or not isinstance(data.get('data'), list)
                or any(not isinstance(event, dict) for event in data['data'])):
            raise OddsAPIError('SportsGameOdds returned an invalid event envelope')
        saved = {'fetched_at': datetime.now(timezone.utc).isoformat(), 'params': params, 'response': data}
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
                json.dump(saved, f)
                tmp = Path(f.name)
            tmp.replace(path)
        return saved

    def events(self, params, max_pages=3, refresh=False):
        if max_pages < 1:
            raise OddsAPIError('max_pages must be positive')
        cursor, seen = None, set()
        for _ in range(max_pages):
            page = self.page({**params, **({'cursor': cursor} if cursor else {})}, refresh=refresh)
            yield page
            cursor = page['response'].get('nextCursor')
            if not cursor:
                return
            if not isinstance(cursor, str) or cursor in seen:
                raise OddsAPIError('Invalid/repeated pagination cursor; refusing to loop')
            seen.add(cursor)
        raise OddsAPIError('Pagination cap reached; imported pages are preserved. Narrow the date window or raise --max-pages.')


    def players(self, event_id, max_pages=3):
        """Fetch event player metadata only when an unknown identity needs a name."""
        result, cursor, seen = {}, None, set()
        for _ in range(max_pages):
            params = {'eventID': event_id, 'limit': 100}
            if cursor:
                params['cursor'] = cursor
            page = self.page(params, resource='players')
            for player in page['response']['data']:
                player_id = player.get('playerID')
                if not isinstance(player_id, str) or player_id in result:
                    raise OddsAPIError('Invalid or duplicate SportsGameOdds player identity')
                result[player_id] = player
            cursor = page['response'].get('nextCursor')
            if not cursor:
                return result
            if not isinstance(cursor, str) or cursor in seen:
                raise OddsAPIError('Invalid/repeated player pagination cursor')
            seen.add(cursor)
        raise OddsAPIError('Player pagination cap reached; refusing incomplete identities')


def american_price(value):
    # Prices are American integers, never decimal odds.
    if not isinstance(value, str) or not re.fullmatch(r'[+-]?\d+', value):
        return None
    number = int(value)
    return number if 100 <= abs(number) <= 1000000 else None


def extract_points(event, bookmakers=BOOKMAKERS):
    """Group sides by exact player/book/threshold; retain unavailable observations."""
    grouped, diagnostics = {}, Counter()
    players = event.get('players') or {}
    if not isinstance(players, dict) or not isinstance(event.get('odds', {}), dict):
        raise ValueError('Invalid SportsGameOdds players/odds objects')
    for odd in (event.get('odds') or {}).values():
        if not isinstance(odd, dict):
            diagnostics['invalid_market'] += 1
            continue
        player_id = odd.get('statEntityID')
        if (odd.get('statID') != 'points' or odd.get('periodID') != 'game'
                or odd.get('betTypeID') != 'ou' or odd.get('sideID') not in ('over', 'under')
                or not isinstance(player_id, str) or player_id not in players or player_id in ('home', 'away', 'all')
                or (odd.get('playerID') and odd['playerID'] != player_id)):
            diagnostics['excluded_market'] += 1
            continue
        side = odd['sideID']
        for book, quote in (odd.get('byBookmaker') or {}).items():
            if book not in bookmakers or not isinstance(quote, dict) or quote.get('isMainLine') is False:
                continue
            try:
                threshold = Decimal(str(quote.get('overUnder')))
                if not threshold.is_finite() or not 0 <= threshold < 1000 or threshold != threshold.quantize(Decimal('.01')):
                    raise ValueError()
            except (InvalidOperation, ValueError):
                diagnostics['invalid_line'] += 1
                continue
            updated = None
            if quote.get('lastUpdatedAt'):
                try:
                    updated = aware_datetime(quote['lastUpdatedAt'])
                except ValueError:
                    diagnostics['invalid_timestamp'] += 1
                    continue
            row = grouped.setdefault((player_id, book, threshold), {'provider_player_id': player_id,
                'bookmaker': book, 'line': threshold, 'over_price': None, 'under_price': None,
                'over_available': False, 'under_available': False,
                'over_updated_at': None, 'under_updated_at': None})
            if side in row.setdefault('_seen', set()):
                row['_conflict'] = True
            row['_seen'].add(side)
            row[f'{side}_price'] = american_price(quote.get('odds'))
            row[f'{side}_available'] = (quote.get('available') is True and
                not any(odd.get(flag) is True for flag in ('started', 'ended', 'cancelled')))
            row[f'{side}_updated_at'] = updated
    result = []
    for row in grouped.values():
        row.pop('_seen')
        if row.pop('_conflict', False):
            diagnostics['ambiguous_side'] += 1
        else:
            result.append(row)
    return result, diagnostics


def extract_closing_points(event, bookmakers=BOOKMAKERS):
    """Parse explicit bookmaker event-start values, never last-live fallbacks.

    Contract: https://sportsgameodds.com/docs/data-types/odds
    closeOverUnder / closeOdds are the book's values at event start time.
    lastUpdatedAt describes the current/live quote, not the closing snapshot.
    """
    import copy
    closing = copy.deepcopy(event)
    for odd in (closing.get('odds') or {}).values():
        if not isinstance(odd, dict):
            continue
        for quote in (odd.get('byBookmaker') or {}).values():
            if not isinstance(quote, dict):
                continue
            quote['overUnder'] = quote.get('closeOverUnder')
            quote['odds'] = quote.get('closeOdds')
            quote.pop('lastUpdatedAt', None)
    return extract_points(closing, bookmakers)
