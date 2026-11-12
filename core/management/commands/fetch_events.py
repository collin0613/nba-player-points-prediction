from django.core.management.base import BaseCommand
from core.services.odds_fetch import add_fetch_arguments, fetch_for_games


class Command(BaseCommand):
    help = 'Match SportsGameOdds event IDs onto existing NBA games; never create games'

    def add_arguments(self, parser):
        add_fetch_arguments(parser)

    def handle(self, *args, **options):
        fetch_for_games(options, self.stdout, store_quotes=False)
