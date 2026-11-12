"""Synchronize identities/schedule and upsert the latest player-game predictions."""
from django.core.management.base import BaseCommand, CommandError
from core.services.predictions import sync_roster, import_predictions
from core.services.schedule import sync_schedule


class Command(BaseCommand):
    help = 'Sync current players/games and import predictions without duplicate player-game rows'

    def add_arguments(self, parser):
        parser.add_argument('--season', default='2026-27')
        parser.add_argument('--csv')
        parser.add_argument('--historical', action='store_true', help='Import historical predictions without changing current roster state')

    def handle(self, *args, **options):
        try:
            players = 0 if options['historical'] else sync_roster(options['season'])
            self.stdout.write(f"Synced {players} current players and {sync_schedule(options['season'])} schedule rows")
            if options['csv']:
                run, counts = import_predictions(options['csv'], options['season'])
                self.stdout.write(f'Run {run.pk}: {counts}')
        except (ValueError, KeyError, OSError) as exc:
            raise CommandError(str(exc)) from None
