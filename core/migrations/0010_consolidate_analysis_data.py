"""Archive old tables and migrate latest provably pregame state before retiring them."""
import hashlib
import json
import math
import uuid
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo
from django.conf import settings
from django.core.serializers.json import DjangoJSONEncoder
from django.db import migrations
from django.utils import timezone


def consolidate(apps, schema_editor):
    db = schema_editor.connection.alias
    get = lambda name: apps.get_model('core', name).objects.using(db)
    now = timezone.now()
    local = ZoneInfo('America/New_York')
    old_names = ['LegacyModelPrediction', 'LegacySportsbookLine', 'RosterSnapshot',
                 'PlayerForecast', 'ProviderPlayer', 'OddsSnapshot', 'SportsbookQuote',
                 'ForecastComparison', 'PlayerGameResult']
    archive = {name: list(get(name).values()) for name in old_names}
    archive['Game'] = list(get('Game').values())
    archive['PredictionRun'] = list(get('PredictionRun').values())
    if not any(archive.values()):
        return
    report = {'forecast_redirects': {}, 'quote_redirects': {}, 'comparison_redirects': {},
              'snapshot_redirects': {}, 'provider_redirects': {}, 'result_redirects': {},
              'roster_redirects': {}, 'legacy_prediction_redirects': {}, 'quarantined': []}
    archive_id = uuid.uuid4()
    directory = Path(settings.BASE_DIR) / 'data' / 'archive'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'pre_consolidation_{archive_id}.json'
    raw = json.dumps(archive, cls=DjangoJSONEncoder, sort_keys=True)
    path.write_text(raw)
    audit = get('DataImport').create(id=archive_id, source='nba', kind='roster', observed_at=now,
        request_fingerprint=hashlib.sha256(b'six_table_consolidation').hexdigest(),
        request_scope={'operation': 'six_table_consolidation'}, payload_path=str(path.relative_to(settings.BASE_DIR)),
        payload_sha256=hashlib.sha256(raw.encode()).hexdigest(), row_count=sum(len(v) for v in archive.values()))
    latest_rosters = {}
    for r in get('RosterSnapshot').order_by('-observed_at', '-pk'):
        latest_rosters.setdefault(r.player_id, r)
        report['roster_redirects'][str(r.pk)] = r.player_id
    for player_id, r in latest_rosters.items():
        get('Player').filter(pk=player_id).update(current_team=r.team or None, roster_season=r.season,
            prior_mpg=r.prior_mpg, prior_games=r.prior_games, eligible=r.eligible,
            roster_observed_at=r.observed_at, roster_import_id=audit.pk)
    grouped = defaultdict(list)
    for m in get('ProviderPlayer').all():
        grouped[m.player_id].append(m)
        report['provider_redirects'][str(m.pk)] = m.player_id
    for player_id, mappings in grouped.items():
        if len(mappings) != 1:
            report['quarantined'].append({'player_id': player_id, 'reason': 'multiple_provider_aliases'})
            get('Player').filter(pk=player_id).update(mapping_status='conflict')
            continue
        m = mappings[0]
        get('Player').filter(pk=player_id).update(sportsgameodds_player_id=m.provider_player_id,
            sportsgameodds_name=m.provider_name, mapping_method=m.match_method,
            mapping_validated_team=m.team, mapping_validated_at=m.validated_at,
            mapping_status='verified' if latest_rosters.get(player_id) and latest_rosters[player_id].team == m.team else 'conflict')
    games = {g.pk: g for g in get('Game').all()}
    for g in games.values():
        if g.status in ('', 'scheduled') and g.scheduled_at <= now:
            # Time alone is enough to lock forecasts, not to assert an official final.
            g.status = 'live'
            g.save(using=db, update_fields=['status'])
    for run in get('PredictionRun').all():
        output = Path(settings.BASE_DIR) / 'data' / 'predictions' / run.season / f'{run.pk}.csv'
        run.row_count = get('PlayerForecast').filter(run_id=run.pk).count()
        if output.exists():
            run.output_path = str(output.relative_to(settings.BASE_DIR))
            run.output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
        model = Path(settings.BASE_DIR) / 'models' / f'{run.model_version}.joblib'
        if model.exists():
            run.model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
        run.feature_schema_version = 'population-v1'
        run.save(using=db)
    forecasts = defaultdict(list)
    for f in get('PlayerForecast').select_related('run').order_by('-run__generated_at', '-pk'):
        forecasts[(f.game_id, f.player_id)].append(f)
    selected = {}
    for (game_id, player_id), versions in forecasts.items():
        g = games[game_id]
        valid = [f for f in versions if f.run.generated_at < g.scheduled_at
                 and f.run.feature_cutoff <= f.run.generated_at.astimezone(local).date()
                 and f.run.feature_cutoff <= g.scheduled_at.astimezone(local).date()
                 and math.isfinite(f.predicted_points) and f.predicted_points >= 0]
        if not valid:
            report['quarantined'].append({'game_id': game_id, 'player_id': player_id, 'reason': 'no_verified_pregame_forecast'})
            continue
        f = valid[0]
        team = f.team
        if not team:
            roster = get('RosterSnapshot').filter(player_id=player_id, observed_at__lte=f.run.generated_at).order_by('-observed_at').first()
            team = roster.team if roster else ''
        review = team not in (g.home_team, g.away_team)
        tied = [v for v in valid if v.run.generated_at == f.run.generated_at]
        if any(v.predicted_points != f.predicted_points or v.features != f.features for v in tied):
            review = True
        prediction = get('ModelPrediction').create(game_id=game_id, player_id=player_id,
            predicted_points=f.predicted_points, team_at_prediction=team,
            prediction_run_id=f.run_id, features=f.features, minutes_history_estimate=f.minutes_history_estimate,
            participation_assumption=f.participation_assumption, last_checked_at=f.run.generated_at,
            value_changed_at=f.run.generated_at, created_at=f.run.generated_at,
            prediction_status='needs_review' if review else 'current',
            status_reason='Migration context/chronology requires review' if review else '',
            locked_at=g.scheduled_at if g.scheduled_at <= now or g.status in ('live','final','cancelled') else None)
        selected[(game_id, player_id)] = prediction
        for old in versions:
            report['forecast_redirects'][str(old.pk)] = prediction.pk
    # V1 rows without feature/cutoff provenance stay in the archive rather than being
    # represented as auditable V2 forecasts. Current installation has no V1 rows.
    for old in get('LegacyModelPrediction').all():
        target = selected.get((old.game_id, old.player_id))
        if target:
            report['legacy_prediction_redirects'][str(old.pk)] = target.pk
        else:
            report['quarantined'].append({'legacy_prediction': old.pk, 'reason': 'legacy_prediction_provenance_unverified'})
    for result in get('PlayerGameResult').filter(final_verified=True):
        p = selected.get((result.game_id, result.player_id))
        if p:
            get('ModelPrediction').filter(pk=p.pk).update(actual_points=result.points if result.played else None,
                participation='played' if result.played else 'dnp', result_verified_at=result.fetched_at, result_import_id=audit.pk)
            report['result_redirects'][str(result.pk)] = p.pk
    # Replay complete observations in source-time order. This preserves disappearance
    # and suspension instead of selecting an older available quote.
    lines = {}
    for snapshot in get('OddsSnapshot').order_by('fetched_at', 'pk'):
        g = games[snapshot.game_id]
        if snapshot.fetched_at >= min(g.scheduled_at, snapshot.scheduled_start):
            continue
        observation = get('DataImport').create(source='sportsgameodds', kind='odds', observed_at=snapshot.fetched_at,
            request_fingerprint=hashlib.sha256(f'legacy-snapshot-{snapshot.pk}'.encode()).hexdigest(),
            request_scope={'eventID': snapshot.event_id, 'game_id': g.pk, 'bookmakers': snapshot.bookmakers,
                           'complete': True, 'scheduled_start': snapshot.scheduled_start.isoformat()},
            diagnostics=snapshot.diagnostics, payload_path=audit.payload_path, payload_sha256=audit.payload_sha256)
        report['snapshot_redirects'][str(snapshot.pk)] = str(observation.pk)
        quotes = defaultdict(list)
        for q in get('SportsbookQuote').filter(snapshot_id=snapshot.pk):
            quotes[(q.player_id, q.bookmaker)].append(q)
        for key, existing in list(lines.items()):
            if key[0] == g.pk and key[2] in snapshot.bookmakers and key[1:] not in quotes:
                existing.market_status = 'missing'
                existing.over_available = existing.under_available = False
                existing.line = existing.over_price = existing.under_price = None
                existing.observed_at = snapshot.fetched_at
                existing.source_import_id = observation.pk
                existing.save(using=db)
        for (player_id, book), entries in quotes.items():
            if book not in ('draftkings', 'fanduel'):
                continue
            q = entries[0]
            conflict = len(entries) != 1
            event_closed = any(snapshot.event_status.get(k) is True for k in ('started','ended','cancelled','completed','finalized','delayed')) or snapshot.event_status.get('oddsAvailable') is False
            status = 'conflict' if conflict else ('incomplete' if q.over_price is None or q.under_price is None else
                ('available' if q.over_available and q.under_available and not event_closed else 'suspended'))
            values = dict(line=None if conflict else q.line, over_price=None if conflict else q.over_price,
                under_price=None if conflict else q.under_price, over_available=q.over_available and not conflict and not event_closed,
                under_available=q.under_available and not conflict and not event_closed, market_status=status,
                observed_at=snapshot.fetched_at, value_changed_at=snapshot.fetched_at,
                over_updated_at=q.over_updated_at, under_updated_at=q.under_updated_at,
                source_import_id=observation.pk, locked_at=g.scheduled_at if g.scheduled_at <= now else None)
            line, _ = get('SportsbookLine').update_or_create(game_id=g.pk, player_id=player_id, bookmaker=book, defaults=values)
            lines[(g.pk, player_id, book)] = line
            for old in entries:
                report['quote_redirects'][str(old.pk)] = line.pk
    for comparison in get('ForecastComparison').all():
        target = report['quote_redirects'].get(str(comparison.quote_id))
        if target:
            report['comparison_redirects'][str(comparison.pk)] = target
    report['predictions_preserved'] = get('ModelPrediction').count()
    report['lines_preserved'] = get('SportsbookLine').count()
    report['boston_detroit_preserved'] = get('ModelPrediction').filter(game__nba_game_id='0022600001').count()
    audit.diagnostics = report
    audit.save(using=db, update_fields=['diagnostics'])


class Migration(migrations.Migration):
    dependencies = [('core', '0009_six_table_additions')]
    operations = [migrations.RunPython(consolidate)]
