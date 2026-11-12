import csv
import json
from pathlib import Path
from django.conf import settings
from django.db import transaction
from django.utils import timezone
from core.models import Game, ModelPrediction, SportsbookLine
from core.services.imports import record_import
from src.market_mapping import aware_datetime, normalize_team


@transaction.atomic
def sync_schedule(season):
    root = Path(settings.BASE_DIR)/'data/raw'/season
    path = root/'schedule.csv'
    if not path.exists():
        return 0
    observed = aware_datetime(json.loads(path.with_suffix('.meta.json').read_text())['fetched_at'])
    with path.open() as f:
        rows = list(csv.DictReader(f))
    item, _ = record_import('nba', 'schedule', observed, {'season': season}, rows, row_count=len(rows))
    pending, synced = [], 0
    now = timezone.now()
    for row in rows:
        home = normalize_team(row['homeTeam_teamTricode']); away = normalize_team(row['awayTeam_teamTricode'])
        if row['seasonYear'] == season and all(not row[f'{side}_teamTricode'] and row.get(f'{side}_teamId') == '0' for side in ('homeTeam', 'awayTeam')):
            pending.append(row['gameId'])
            continue
        if not home or not away or home == away or row['seasonYear'] != season:
            raise ValueError('Invalid NBA schedule row')
        synced += 1
        start = aware_datetime(row['gameDateTimeUTC'])
        status = {'1': 'scheduled', '2': 'live', '3': 'final'}.get(str(row['gameStatus']), 'postponed')
        game, created = Game.objects.get_or_create(nba_game_id=row['gameId'].zfill(10), defaults={
            'season': season, 'home_team': home, 'away_team': away, 'scheduled_at': start, 'status': status})
        game = Game.objects.select_for_update().get(pk=game.pk)
        if game.schedule_checked_at and game.schedule_checked_at >= observed:
            continue
        if (game.home_team, game.away_team) != (home, away):
            raise ValueError('Official matchup changed; investigate before replacing historical teams')
        locked = game.is_locked(now)
        if locked:
            ModelPrediction.objects.filter(game=game, locked_at__isnull=True).update(locked_at=game.scheduled_at)
            SportsbookLine.objects.filter(game=game, locked_at__isnull=True).update(locked_at=game.scheduled_at)
            if status in ('live', 'final') and game.status != 'final':
                game.status = status
        else:
            if game.scheduled_at != start or status != 'scheduled':
                ModelPrediction.objects.filter(game=game).update(prediction_status='needs_review', status_reason='Schedule changed or game is not scheduled; review before predicting')
            game.scheduled_at = start; game.status = status
        game.schedule_checked_at = observed; game.updated_at = now
        game.save()
    item.diagnostics = {'unassigned_matchups': pending, 'synced_games': synced}
    item.row_count = synced
    item.save(update_fields=['diagnostics', 'row_count'])
    return synced
