"""Atomic newest-wins prediction and roster imports; no post-tipoff mutation."""
import csv
import hashlib
import json
import math
import uuid
from collections import Counter
from pathlib import Path
from django.conf import settings
from django.db import transaction
from django.utils import timezone
from core.models import Game, Player, ModelPrediction, PredictionRun
from core.services.imports import record_import
from src.market_mapping import aware_datetime, normalize_team


def optional_float(value):
    return float(value) if value not in ('', None) else None


@transaction.atomic
def sync_roster(season):
    root = Path(settings.BASE_DIR)
    observed = aware_datetime(json.loads((root/'data/raw'/season/'all_roster.meta.json').read_text())['fetched_at'])
    with (root/'data/processed'/season/'roster_eligibility.csv').open() as f:
        rows = list(csv.DictReader(f))
    item, _ = record_import('nba', 'roster', observed, {'season': season}, rows, row_count=len(rows))
    seen = set()
    for row in rows:
        pid = int(row['player_id'])
        if pid in seen:
            raise ValueError('Duplicate player in roster')
        seen.add(pid)
        player, _ = Player.objects.get_or_create(pk=pid, defaults={'name': row['player_name']})
        if player.roster_observed_at and player.roster_observed_at >= observed:
            continue
        team = normalize_team(row['team']) if row['team'] else None
        player.name = row['player_name']; player.current_team = team
        player.roster_season = season; player.prior_mpg = optional_float(row['prior_mpg'])
        player.prior_games = int(float(row['prior_games'])) if row['prior_games'] else 0
        player.eligible = row['eligible'].lower() == 'true'
        player.roster_observed_at = observed; player.roster_import = item
        if player.sportsgameodds_player_id and player.mapping_validated_team != team:
            player.mapping_status = 'conflict'
        player.save()
    # A complete new roster can remove membership without deleting canonical players.
    Player.objects.filter(roster_season=season).exclude(pk__in=seen).filter(roster_observed_at__lt=observed).update(
        current_team=None, eligible=False, roster_observed_at=observed, roster_import=item, mapping_status='conflict')
    return len(rows)


@transaction.atomic
def upsert_prediction(run, row, now=None):
    now = now or timezone.now()
    start = aware_datetime(row['next_game_start_utc'])
    historical = run.run_type == 'historical_backfill'
    if run.generated_at > now:
        return 'non_pregame'
    if historical:
        from core.services.history import LOCAL_TZ
        if (not run.target_date or run.target_date >= now.astimezone(LOCAL_TZ).date()
                or start.astimezone(LOCAL_TZ).date() != run.target_date
                or run.feature_cutoff != run.target_date):
            raise ValueError('Invalid historical target date/cutoff')
        from datetime import date
        if date.fromisoformat(run.metadata['data_through']) >= run.feature_cutoff:
            raise ValueError('Historical model training overlaps prediction cutoff')
        observed = aware_datetime(row['roster_observed_at'])
        if observed.astimezone(LOCAL_TZ).date() >= run.target_date:
            raise ValueError('Historical roster must have been observed before target date')
    elif run.generated_at >= start:
        return 'non_pregame'
    if run.feature_cutoff > timezone.localtime(run.generated_at).date() or run.feature_cutoff > timezone.localtime(start).date():
        raise ValueError('Invalid feature cutoff for a pregame prediction')
    value = float(row['predicted_pts'])
    if not math.isfinite(value) or value < 0:
        raise ValueError('Prediction must be finite and nonnegative')
    home, away, team = [normalize_team(row[k]) for k in ('home_team', 'away_team', 'team')]
    if not home or not away or home == away or team not in (home, away):
        raise ValueError('Invalid prediction team/game identity')
    game_id = str(row['nba_game_id']).zfill(10)
    if len(game_id) != 10 or not game_id.isdigit():
        raise ValueError('Invalid NBA game ID')
    game, _ = Game.objects.get_or_create(nba_game_id=game_id, defaults={
        'season': run.season, 'home_team': home, 'away_team': away, 'scheduled_at': start})
    game = Game.objects.select_for_update().get(pk=game.pk)
    if game.is_locked(now) and not historical:
        ModelPrediction.objects.filter(game=game, locked_at__isnull=True).update(locked_at=game.scheduled_at)
        return 'locked'
    if game.status != 'scheduled' and not historical:
        return 'not_scheduled'
    if (game.home_team, game.away_team) != (home, away):
        raise ValueError('Prediction matchup disagrees with canonical NBA game')
    if game.scheduled_at != start:
        raise ValueError('Prediction start disagrees with the refreshed schedule; regenerate predictions')
    prediction = ModelPrediction.objects.select_related('prediction_run').filter(game=game, player_id=int(row['player_id'])).first()
    features = json.loads(row['features_json'])
    json.dumps(features, allow_nan=False)
    if not isinstance(features, dict):
        raise ValueError('Features must be a JSON object')
    minutes = optional_float(row.get('minutes_history_estimate'))
    if minutes is not None and (not math.isfinite(minutes) or minutes < 0):
        raise ValueError('Invalid minutes estimate')
    if prediction:
        if historical:
            return 'existing_preserved'
        if prediction.locked_at:
            return 'locked'
        previous_time = prediction.prediction_run.generated_at
        if run.generated_at < previous_time:
            return 'older_ignored'
        if run.pk == prediction.prediction_run_id:
            return 'unchanged'
        if run.generated_at == previous_time:
            raise ValueError('Conflicting runs have identical generation timestamps')
        changed = prediction.predicted_points != value
    else:
        prediction = ModelPrediction(game=game, player_id=int(row['player_id']))
        changed = True
    created = prediction.pk is None
    prediction.predicted_points = value; prediction.prediction_run = run
    prediction.team_at_prediction = team; prediction.features = features
    prediction.minutes_history_estimate = minutes
    prediction.prediction_status = 'current'; prediction.status_reason = ''
    prediction.last_checked_at = run.generated_at
    if changed:
        prediction.value_changed_at = run.generated_at
    prediction.updated_at = now
    if historical:
        prediction.locked_at = game.scheduled_at
    prediction.save()
    return 'created' if created else ('updated' if changed else 'provenance_updated')


@transaction.atomic
def import_predictions(path, season):
    path = Path(path)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Prediction CSV is empty')
    first = rows[0]
    for key in ('run_id', 'model_version', 'prediction_season', 'predicted_at_utc', 'feature_cutoff_date_exclusive'):
        if len({r[key] for r in rows}) != 1:
            raise ValueError(f'CSV contains inconsistent {key}')
    if first['prediction_season'] != season:
        raise ValueError('Prediction season mismatch')
    if len({(r['nba_game_id'], r['player_id']) for r in rows}) != len(rows):
        raise ValueError('Duplicate player/game in prediction CSV')
    for key in ('run_type', 'target_date'):
        if len({r.get(key, '') for r in rows}) != 1:
            raise ValueError(f'CSV contains inconsistent {key}')
    run_type = first.get('run_type') or 'live'
    if run_type not in ('live', 'historical_backfill'):
        raise ValueError('Invalid prediction run type')
    version = first['model_version']
    if Path(version).name != version:
        raise ValueError('Invalid model artifact name')
    root = Path(settings.BASE_DIR)
    metadata = json.loads((root/'models'/f'{version}.json').read_text())
    artifact = root/'models'/f'{version}.joblib'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    run, created = PredictionRun.objects.get_or_create(pk=uuid.UUID(first['run_id']), defaults={
        'run_type':run_type, 'target_date':first.get('target_date') or None,
        'model_version': version, 'season': season, 'generated_at': aware_datetime(first['predicted_at_utc']),
        'feature_cutoff': first['feature_cutoff_date_exclusive'], 'metadata': metadata,
        'output_path': str(path.resolve()), 'output_sha256': digest, 'row_count': len(rows),
        'model_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest() if artifact.exists() else '',
        'feature_schema_version': metadata.get('feature_schema_version',
            'population-v1' if version.startswith('points_population_') else 'legacy-v1')})
    if not created and run.output_sha256 and run.output_sha256 != digest:
        raise ValueError('An existing run UUID cannot be reused with changed CSV content')
    run.refresh_from_db()
    if (run.run_type != run_type or (run.target_date.isoformat() if run.target_date else '') != first.get('target_date', '')
            or run.season != season or run.model_version != version or run.generated_at != aware_datetime(first['predicted_at_utc'])
            or run.feature_cutoff.isoformat() != first['feature_cutoff_date_exclusive']):
        raise ValueError('Existing run metadata disagrees with CSV')
    if run.run_type == 'historical_backfill':
        for row in rows:
            Player.objects.get_or_create(pk=int(row['player_id']), defaults={'name':row['player_name']})
    counts = Counter(upsert_prediction(run, row) for row in rows)
    # Only a saved complete generator manifest authorizes withdrawals. Manual/partial
    # CSV imports must never withdraw other players' predictions.
    manifest = path.with_suffix('.json')
    if manifest.exists():
        info = json.loads(manifest.read_text())
        complete = info.get('run_id') == str(run.pk) and info.get('rows') == len(rows) and info.get('skipped') == []
        run.status = 'success' if complete else 'partial'
        run.save(update_fields=['status'])
        if complete and run.run_type == 'live':
            represented = {(str(r['nba_game_id']).zfill(10), int(r['player_id'])) for r in rows}
            for p in ModelPrediction.objects.select_related('game', 'prediction_run', 'player').filter(
                    game__season=season, game__scheduled_at__gt=timezone.now(), locked_at__isnull=True):
                if (p.game.nba_game_id, p.player_id) not in represented and p.prediction_run.generated_at < run.generated_at and (not p.player.eligible or p.player.current_team != p.team_at_prediction):
                    p.prediction_status = 'withdrawn'; p.status_reason = 'Eligibility or team changed in complete refreshed roster'
                    p.save(update_fields=['prediction_status', 'status_reason'])
    return run, dict(counts)
