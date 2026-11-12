"""Score original pregame forecasts; never feed results back into their inputs."""
import math
import uuid
from collections import defaultdict, Counter
from django.core.management.base import BaseCommand, CommandError
from core.models import ModelPrediction, SportsbookLine
from core.services.history import add_date_arguments, date_range, filter_dates, pregame, prediction_exclusion, line_exclusion


def metrics(rows):
    errors = [pred - actual for pred, actual in rows]
    n = len(errors)
    mean = sum(actual for _, actual in rows) / n
    sst = sum((actual - mean)**2 for _, actual in rows)
    sse = sum(e*e for e in errors)
    r2 = f'{1-sse/sst:.3f}' if n > 1 and sst else 'n/a'
    return f'n={n} MAE={sum(abs(e) for e in errors)/n:.3f} RMSE={math.sqrt(sse/n):.3f} R2={r2} bias={sum(errors)/n:+.3f}'


class Command(BaseCommand):
    help = 'Evaluate retained pregame player-game predictions and matched pregame lines'

    def add_arguments(self, parser):
        add_date_arguments(parser)
        parser.add_argument('--max-age-minutes', type=int, default=30, help='Maximum pre-tipoff line age (default 30 minutes)')
        parser.add_argument('--run', dest='run_id')
        parser.add_argument('--season', default='2026-27')
        parser.add_argument('--model-version', help='Evaluate only this saved model version')

    def handle(self, *args, **options):
        if options['max_age_minutes'] <= 0:
            raise CommandError('--max-age-minutes must be positive')
        bounds = date_range(options)
        qs = ModelPrediction.objects.filter(game__season=options['season']).select_related('prediction_run','game','player')
        qs = filter_dates(qs, 'game__scheduled_at', bounds)
        if options['model_version']:
            qs = qs.filter(prediction_run__model_version=options['model_version'])
        if options['run_id']:
            try:
                qs = qs.filter(prediction_run_id=uuid.UUID(options['run_id']))
            except ValueError:
                raise CommandError('Invalid --run UUID') from None
        groups, excluded, eligible = defaultdict(list), Counter(), {}
        for p in qs:
            reason = prediction_exclusion(p)
            if reason:
                excluded[reason] += 1
            else:
                eligible[(p.game_id, p.player_id)] = p
                groups[p.prediction_run_id].append((p.predicted_points, p.actual_points))
        for run_id, rows in groups.items():
            self.stdout.write(f'Run {run_id} retained predictions: {metrics(rows)}')
        by_book = defaultdict(list)
        for line in SportsbookLine.objects.filter(game_id__in={k[0] for k in eligible}).select_related('game'):
            p = eligible.get((line.game_id, line.player_id))
            if p is None:
                continue
            if line_exclusion(line, options['max_age_minutes']):
                excluded['line_not_valid_at_pregame_cutoff'] += 1
                continue
            by_book[line.bookmaker].append((p.predicted_points, float(line.line), p.actual_points))
        for book, rows in by_book.items():
            self.stdout.write(f'{book} paired model: {metrics([(p,a) for p,l,a in rows])}; line: {metrics([(l,a) for p,l,a in rows])}')
        if not groups:
            self.stdout.write('No verified final, played, pregame predictions to evaluate yet.')
        self.stdout.write(f'Excluded: {dict(excluded)}. Evaluates latest retained pregame state, not every earlier run or recommendation.')
