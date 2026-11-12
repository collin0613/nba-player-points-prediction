from django.core.management.base import BaseCommand
from core.services.odds_fetch import add_fetch_arguments, fetch_for_games


class Command(BaseCommand):
    help = 'Fetch/cache SportsGameOdds player points and upsert latest pregame lines'

    def add_arguments(self, parser):
        add_fetch_arguments(parser)
        parser.add_argument('--historical', action='store_true', help='Retrieve past events and explicit bookmaker closing points lines')
        parser.add_argument('--store', action='store_true', help='Compatibility option; saving is now the default')

    def handle(self, *args, **options):
        fetch_for_games(options, self.stdout, upcoming_only=not options['historical'])
