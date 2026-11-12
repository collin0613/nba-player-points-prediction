"""Settle all prediction runs only after official NBA final-status verification."""
from datetime import timedelta
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from core.models import Game, ModelPrediction
from core.services.history import add_date_arguments, date_range, filter_dates
from core.services.results import backfill_games, cleanup_dnps


class Command(BaseCommand):
    help = 'Backfill ModelPrediction actual points using verified final NBA V3 boxscores'

    def add_arguments(self, parser):
        add_date_arguments(parser)
        parser.add_argument('--days', type=int, default=10)
        parser.add_argument('--game-id', help='Official NBA game ID; overrides lookback window')
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--refresh', action='store_true', help='Recheck settled games for official stat corrections')
        parser.add_argument('--verbose', action='store_true')
        parser.add_argument('--buffer-hours', type=int, default=3, help='Request delay only; final status is always required')

    def handle(self, *args, **options):
        if options['days'] <= 0 or options['buffer_hours'] < 0:
            raise CommandError('days must be positive and buffer-hours nonnegative')
        bounds = date_range(options)
        if options['game_id'] and any(bounds):
            raise CommandError('Use --game-id or date filters, not both')
        now = timezone.now()
        predictions = ModelPrediction.objects.all()
        cleanup_scope = filter_dates(predictions, 'game__scheduled_at', bounds)
        if options['game_id']:
            cleanup_scope = cleanup_scope.filter(game__nba_game_id=options['game_id'])
        elif not any(bounds):
            cleanup_scope = cleanup_scope.filter(game__scheduled_at__gte=now-timedelta(days=options['days']))
        deleted = cleanup_dnps(cleanup_scope, now, dry_run=options['dry_run'])
        if not options['refresh']:
            predictions = predictions.filter(result_verified_at__isnull=True)
        games = Game.objects.filter(pk__in=predictions.values('game_id'))
        if options['game_id']:
            games = games.filter(nba_game_id=options['game_id'])
        elif any(bounds):
            games = filter_dates(games, 'scheduled_at', bounds).filter(scheduled_at__lte=now)
        else:
            games = games.filter(scheduled_at__gte=now - timedelta(days=options['days']),
                                 scheduled_at__lte=now - timedelta(hours=options['buffer_hours']))
        if not games.exists():
            self.stdout.write('No unsettled predictions in the selected games (use --refresh to recheck settled results).')
        totals = backfill_games(games, self.stdout, self.stderr,
                                dry_run=options['dry_run'], verbose=options['verbose'])
        totals['previously_verified_dnps_deleted'] = deleted
        self.stdout.write(f'Actual results: {dict(totals)}')
