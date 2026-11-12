"""Finish a stored reconstruction without coupling results to odds availability."""
import sys
from django.core.management import call_command
from django.core.management.base import CommandError, OutputWrapper
from core.models import Game, ModelPrediction
from core.services.results import backfill_games


def complete_historical_run(game_ids, day, *, offline=False):
    games = Game.objects.filter(nba_game_id__in=game_ids)
    if not offline:
        try:
            call_command('fetch_odds', date=day, historical=True)
        except CommandError as exc:
            print(f'Historical odds unavailable: {exc}. Continuing with NBA results.')
        pending = games.filter(model_predictions__result_verified_at__isnull=True,
                               model_predictions__isnull=False).distinct()
        totals = backfill_games(pending, OutputWrapper(sys.stdout), OutputWrapper(sys.stderr), verbose=True)
        print(f'Historical results: {dict(totals)}')
    else:
        print('Offline: odds/results requests skipped; rerun online to settle pending predictions.')
    predictions = ModelPrediction.objects.filter(game__in=games)
    unresolved = predictions.filter(result_verified_at__isnull=True)
    print(f'Result completeness: {predictions.count()-unresolved.count()}/{predictions.count()} verified; '
          f'{unresolved.count()} unresolved; {predictions.filter(participation="dnp").count()} DNP.')
    if unresolved.exists():
        for p in unresolved.select_related('game','player'):
            print(f'UNRESOLVED game={p.game.nba_game_id} player={p.player_id} {p.player.name}')
        print(f'Retry: python manage.py update_actual_points --date {day} --verbose')
        return False
    return True
