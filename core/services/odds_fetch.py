"""Bounded, cached fetches for dates/games that already have NBA forecasts."""
from collections import Counter
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from django.conf import settings
from django.core.management.base import CommandError
from django.utils import timezone
from core.models import Game
from core.services.sportsbook import MappingError, import_event
from src.odds_api import SportsGameOddsClient, OddsAPIError, BOOKMAKERS
from src.market_mapping import aware_datetime, normalize_team


def parse_books(value):
    books = sorted(set(x.strip().lower() for x in value.split(',') if x.strip()))
    if not books or set(books) - set(BOOKMAKERS):
        raise CommandError('Supported bookmakers: draftkings,fanduel')
    return books


def add_fetch_arguments(parser):
    parser.add_argument('--date', help='NBA game date in America/New_York (YYYY-MM-DD); default today')
    parser.add_argument('--days', type=int, default=1)
    parser.add_argument('--game-id', help='Official 10-digit NBA game ID')
    parser.add_argument('--event-id', help='SportsGameOdds string ID, validated against an existing NBA game')
    parser.add_argument('--bookmakers', default='draftkings,fanduel')
    parser.add_argument('--max-pages', type=int, default=3)
    parser.add_argument('--refresh', action='store_true', help='Bypass the 10-minute response cache (uses quota)')


def selected_games(options):
    games = Game.objects.filter(nba_game_id__isnull=False)
    if options.get('game_id'):
        return games.filter(nba_game_id=options['game_id'])
    try:
        day = datetime.strptime(options['date'], '%Y-%m-%d').date() if options.get('date') else timezone.localdate()
    except ValueError:
        raise CommandError('--date must be YYYY-MM-DD') from None
    if not 1 <= options['days'] <= 14:
        raise CommandError('--days must be between 1 and 14')
    start = datetime.combine(day, time.min, ZoneInfo('America/New_York'))
    end = datetime.combine(day + timedelta(days=options['days']), time.min, ZoneInfo('America/New_York'))
    return games.filter(scheduled_at__gte=start, scheduled_at__lt=end)


def fetch_for_games(options, output, store_quotes=True, upcoming_only=True):
    books = parse_books(options['bookmakers'])
    queryset = selected_games(options)
    if upcoming_only:
        queryset = queryset.filter(scheduled_at__gt=timezone.now()).exclude(status__in=['final','cancelled','postponed','live'])
    if options.get('historical'):
        queryset = queryset.filter(model_predictions__isnull=False).distinct()
    games = list(queryset.order_by('scheduled_at'))
    if not games:
        output.write('No existing NBA games in this window; no API requests made.')
        return
    ids = {g.pk for g in games}
    params = {'leagueID': 'NBA', 'includeAltLines': 'false', 'bookmakerID': ','.join(books), 'limit': 10}
    if options.get('historical'):
        params['includeOpenCloseOdds'] = 'true'
    if options.get('event_id'):
        params['eventID'] = options['event_id']
    elif options.get('game_id') and all(g.sportsgameodds_event_id for g in games):
        params['eventIDs'] = ','.join(g.sportsgameodds_event_id for g in games)
    else:
        # Do not filter oddsAvailable=true: suspended/missing quotes must supersede
        # earlier active snapshots instead of silently leaving those actionable.
        params.update(startsAfter=(min(g.scheduled_at for g in games) - timedelta(minutes=30)).isoformat(),
                      startsBefore=(max(g.scheduled_at for g in games) + timedelta(minutes=30)).isoformat())
    totals = Counter()
    try:
        client = SportsGameOddsClient(cache_dir=settings.BASE_DIR / '.cache' / 'sportsgameodds')
        for page in client.events(params, max_pages=options['max_pages'], refresh=options['refresh']):
            observed = aware_datetime(page['fetched_at'])
            for event in page['response']['data']:
                try:
                    # match_event never creates games. Restrict persistence to requested games.
                    from core.services.sportsbook import event_identity, EVENT_TOLERANCE
                    home, away, start = event_identity(event)
                    relevant = [g for g in games if normalize_team(g.home_team) == home and normalize_team(g.away_team) == away
                                and abs(g.scheduled_at - start) <= EVENT_TOLERANCE]
                    if not relevant:
                        totals['unmatched_or_outside_requested_games'] += 1
                        output.write(f"SKIP event={event.get('eventID')}: unmatched_or_outside_requested_games")
                        continue
                    metadata = None
                    def player_lookup(provider_id):
                        nonlocal metadata
                        if metadata is None:
                            metadata = client.players(event['eventID'], max_pages=options['max_pages'])
                        return metadata.get(provider_id)
                    game, snapshot, counts = import_event(event, observed, books, store_quotes,
                        historical=options.get('historical', False), player_lookup=player_lookup)
                    assert game.pk in ids
                    totals.update(counts)
                    output.write(f"{game}: {counts}")
                    if snapshot:
                        for issue in snapshot.diagnostics.get('player_mapping_issues', []):
                            output.write(f"  SKIP player={issue['provider_player_id']}: {issue['reason']}")
                except MappingError as exc:
                    totals[str(exc)] += 1
                    output.write(f"SKIP event={event.get('eventID')}: {exc}")
    except (OddsAPIError, ValueError) as exc:
        raise CommandError(str(exc)) from None
    output.write(f'SportsGameOdds import: {dict(totals)}')
