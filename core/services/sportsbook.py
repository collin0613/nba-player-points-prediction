"""Canonical identities and newest-observation sportsbook upserts."""
from collections import Counter
from datetime import timedelta
from zoneinfo import ZoneInfo
from django.db import transaction
from core.models import Game, Player, SportsbookLine, ModelPrediction
from core.services.imports import record_import
from django.utils import timezone
from src.market_mapping import aware_datetime, normalize_player_name, normalize_team, provider_team
from src.odds_api import extract_points

LOCAL_TZ = ZoneInfo('America/New_York')
EVENT_TOLERANCE = timedelta(minutes=30)  # Small schedule shifts, not timezone-sized discrepancies.


class MappingError(ValueError):
    pass


def event_identity(event):
    try:
        home = provider_team(event['teams']['home'])
        away = provider_team(event['teams']['away'])
        start = aware_datetime(event['status']['startsAt'])
    except (KeyError, TypeError, ValueError):
        raise MappingError('invalid_event_identity') from None
    if event.get('leagueID') != 'NBA' or not home or not away or home == away or not event.get('eventID'):
        raise MappingError('invalid_event_identity')
    return home, away, start


@transaction.atomic
def match_event(event):
    home, away, start = event_identity(event)
    candidates = [g for g in Game.objects.select_for_update().filter(
        nba_game_id__isnull=False, scheduled_at__gte=start - EVENT_TOLERANCE,
        scheduled_at__lte=start + EVENT_TOLERANCE)
        if normalize_team(g.home_team) == home and normalize_team(g.away_team) == away
        and g.scheduled_at.astimezone(LOCAL_TZ).date() == start.astimezone(LOCAL_TZ).date()]
    if len(candidates) != 1:
        raise MappingError('ambiguous_event' if candidates else 'unmatched_event')
    game = candidates[0]
    event_id = event['eventID']
    if (game.sportsgameodds_event_id not in (None, event_id)
            or Game.objects.filter(sportsgameodds_event_id=event_id).exclude(pk=game.pk).exists()):
        raise MappingError('event_id_conflict')
    if game.sportsgameodds_event_id is None:
        game.sportsgameodds_event_id = event_id
        game.save(update_fields=['sportsgameodds_event_id'])
    return game


def player_name(info):
    names = info.get('names') or {}
    if not isinstance(names, dict):
        names = {}
    values = [info.get('name'), names.get('display')]
    values.append(' '.join(v for v in (names.get('firstName'), names.get('lastName')) if isinstance(v, str)))
    values.append(' '.join(v for v in (info.get('firstName'), info.get('lastName')) if isinstance(v, str)))
    return next((v.strip() for v in values if isinstance(v, str) and v.strip()), '')


class PlayerResolver:
    def __init__(self, game, event, observed, player_lookup=None):
        self.player_lookup = player_lookup
        self.game, self.event, self.observed = game, event, observed
        self.team_by_id = {v.get('teamID'): provider_team(v) for v in event['teams'].values()}

    def resolve(self, provider_id):
        info = (self.event.get('players') or {}).get(provider_id, {})
        if not isinstance(info, dict) or info.get('playerID', provider_id) != provider_id:
            raise MappingError('invalid_player_identity')
        name = player_name(info)
        team = self.team_by_id.get(info.get('teamID'))
        if not team:
            raise MappingError('missing_player_name_or_team')
        player = Player.objects.filter(sportsgameodds_player_id=provider_id).first()
        if not name and not player and self.player_lookup:
            metadata = self.player_lookup(provider_id)
            if metadata:
                if metadata.get('playerID') != provider_id:
                    raise MappingError('invalid_player_identity')
                name = player_name(metadata)
        if not name and not player:
            raise MappingError('missing_player_name_or_team')
        normalized = normalize_player_name(name)
        if player:
            if player.current_team != team or player.roster_season != self.game.season:
                player.mapping_status = 'conflict'; player.save(update_fields=['mapping_status'])
                raise MappingError('mapped_player_roster_conflict')
            if name and normalized not in {normalize_player_name(player.name), normalize_player_name(player.sportsgameodds_name)}:
                raise MappingError('mapped_player_name_conflict')
        else:
            candidates = [p for p in Player.objects.filter(current_team=team, roster_season=self.game.season)
                          if normalize_player_name(p.name) == normalized]
            if len(candidates) != 1:
                raise MappingError('ambiguous_player' if candidates else 'unmatched_player')
            player = candidates[0]
            if player.sportsgameodds_player_id:
                raise MappingError('provider_identity_conflict')
            player.sportsgameodds_player_id = provider_id
            player.mapping_method = 'normalized_name_team'
        player.sportsgameodds_name = name or player.sportsgameodds_name; player.mapping_status = 'verified'
        player.mapping_validated_team = team; player.mapping_validated_at = timezone.now()
        player.save()
        return player


def paired_values(entries, event_status):
    if len(entries) != 1:
        return dict(line=None, over_price=None, under_price=None, over_available=False,
                    under_available=False, over_updated_at=None, under_updated_at=None, market_status='conflict')
    row = {k: v for k, v in entries[0].items() if k not in ('provider_player_id', 'bookmaker')}
    closed = any(event_status.get(k) is True for k in ('started','ended','cancelled','completed','finalized','delayed')) or event_status.get('oddsAvailable') is False
    if closed:
        row['over_available'] = row['under_available'] = False
    row['market_status'] = ('incomplete' if row['over_price'] is None or row['under_price'] is None else
                           'available' if row['over_available'] and row['under_available'] else 'suspended')
    return row


@transaction.atomic
def import_event(event, fetched_at, bookmakers, store_quotes=True, historical=False, player_lookup=None):
    game = match_event(event)
    now = timezone.now()
    if historical:
        if game.scheduled_at >= now:
            raise MappingError('historical_request_for_future_game')
        return import_closing_event(game, event, fetched_at, bookmakers)
    provider_started = any(event['status'].get(key) is True for key in ('started', 'ended', 'completed', 'finalized'))
    if provider_started:
        if game.status != 'final':
            game.status = 'live'
            game.save(update_fields=['status'])
        lock_time = min(now, game.scheduled_at)
        ModelPrediction.objects.filter(game=game, locked_at__isnull=True).update(locked_at=lock_time)
        SportsbookLine.objects.filter(game=game, locked_at__isnull=True).update(locked_at=lock_time)
    if not store_quotes:
        return game, None, {'matched_event': 1}
    if game.is_locked(now) or event['status'].get('started') is True:
        SportsbookLine.objects.filter(game=game, locked_at__isnull=True).update(locked_at=game.scheduled_at)
        return game, None, {'locked_game': 1}
    if fetched_at > now or fetched_at >= game.scheduled_at:
        raise MappingError('invalid_observation_time')
    scope = {'eventID': event['eventID'], 'game_id': game.pk, 'bookmakers': sorted(bookmakers),
             'complete': True, 'scheduled_start': event['status']['startsAt']}
    item, created = record_import('sportsgameodds', 'odds', fetched_at, scope, event)
    if not created and not item.diagnostics.get('missing_player_name_or_team'):
        return game, item, {'reused_import': 1}
    rows, diagnostics = extract_points(event, bookmakers)
    resolver = PlayerResolver(game, event, fetched_at, player_lookup)
    mapped, issues = {}, []
    for provider_id in {r['provider_player_id'] for r in rows}:
        try:
            mapped[provider_id] = resolver.resolve(provider_id)
        except MappingError as exc:
            issues.append({'provider_player_id': provider_id, 'reason': str(exc)})
            diagnostics[str(exc)] += 1
    counts = Counter(p.pk for p in mapped.values())
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        player = mapped.get(row['provider_player_id'])
        if player and counts[player.pk] == 1:
            groups[(player.pk, row['bookmaker'])].append(row)
    existing = {(q.player_id, q.bookmaker): q for q in SportsbookLine.objects.select_for_update().filter(game=game, bookmaker__in=bookmakers)}
    for key in set(existing) | set(groups):
        line = existing.get(key)
        if line and (line.locked_at or line.observed_at > fetched_at or (line.observed_at == fetched_at and line.source_import_id != item.pk)):
            diagnostics['older_or_locked_ignored'] += 1
            continue
        values = paired_values(groups[key], event['status']) if key in groups else dict(
            line=None, over_price=None, under_price=None, over_available=False, under_available=False,
            over_updated_at=None, under_updated_at=None, market_status='missing')
        if line is None:
            line = SportsbookLine(game=game, player_id=key[0], bookmaker=key[1], observed_at=fetched_at)
        changed = any(getattr(line, k) != v for k, v in values.items() if k not in ('over_updated_at','under_updated_at'))
        for k, v in values.items():
            setattr(line, k, v)
        line.observed_at = fetched_at; line.source_import = item; line.updated_at = now
        if changed:
            line.value_changed_at = fetched_at
        line.save()
        diagnostics['upserted_lines'] += 1
    item.diagnostics = {**dict(diagnostics), 'player_mapping_issues': issues}
    item.row_count = diagnostics['upserted_lines']; item.save(update_fields=['diagnostics','row_count'])
    return game, item, dict(diagnostics)


def import_closing_event(game, event, fetched_at, bookmakers):
    """Fill missing lines from explicit provider closes, retaining real fetch time."""
    from src.odds_api import extract_closing_points
    from collections import defaultdict
    raw_rows, _ = extract_points(event, bookmakers)
    rows, diagnostics = extract_closing_points(event, bookmakers)
    item, _ = record_import('sportsgameodds', 'odds', fetched_at,
        {'eventID':event['eventID'], 'game_id':game.pk, 'historical':True,
         'line_type':'provider_closing', 'bookmakers':sorted(bookmakers)}, event, row_count=len(rows))
    counts = Counter(historical_events=1, historical_quote_rows=len(raw_rows), closing_quote_rows=len(rows))
    if event['status'].get('cancelled') or not event['status'].get('finalized'):
        counts['not_finalized_or_cancelled'] += 1
        rows = []
    # Resolve against the historical prediction's team, never today's roster.
    predictions = list(ModelPrediction.objects.filter(game=game).select_related('player'))
    teams = {v.get('teamID'): provider_team(v) for v in event['teams'].values()}
    groups = defaultdict(list)
    for row in rows:
        info = event.get('players', {}).get(row['provider_player_id'], {})
        name = info.get('name') or ' '.join(filter(None,[info.get('firstName'), info.get('lastName')]))
        team = teams.get(info.get('teamID'))
        matches = [p for p in predictions if team and p.team_at_prediction == team
                   and name and normalize_player_name(p.player.name) == normalize_player_name(name)]
        if len(matches) != 1 or info.get('playerID',row['provider_player_id']) != row['provider_player_id']:
            counts['unmatched_historical_player'] += 1
            continue
        groups[(matches[0].player_id,row['bookmaker'])].append(row)
    for (player_id,book), entries in groups.items():
        if len(entries) != 1:
            counts['conflicting_closing_lines'] += 1
            continue
        row=entries[0]
        if row['over_price'] is None or row['under_price'] is None:
            counts['incomplete_closing_pair'] += 1
            continue
        line, created = SportsbookLine.objects.get_or_create(game=game,player_id=player_id,bookmaker=book,
            defaults={'line':row['line'], 'over_price':row['over_price'], 'under_price':row['under_price'],
                'line_type':'provider_closing','market_status':'suspended',
                'over_available':False,'under_available':False,
                'observed_at':fetched_at,'value_changed_at':fetched_at,
                'source_import':item,'locked_at':game.scheduled_at})
        counts['closing_lines_created' if created else 'existing_lines_preserved'] += 1
    item.diagnostics = {**dict(diagnostics), **dict(counts),
        'pregame_provenance_verified':bool(counts['closing_lines_created'] or counts['existing_lines_preserved']),
        'provenance':'provider-defined event-start closeOverUnder and closeOdds; not an archived live observation'}
    item.save(update_fields=['diagnostics'])
    return game, item, dict(counts)
