"""Refresh upcoming-game odds and compare canonical predictions."""
import csv
import uuid
from io import StringIO
from zoneinfo import ZoneInfo
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError, OutputWrapper
from django.utils import timezone
from core.models import PredictionRun, SportsbookLine
from core.services.comparisons import collect_comparisons, quote_exclusion, select_preferred_rows
from collections import Counter
from core.services.odds_fetch import add_fetch_arguments, fetch_for_games, parse_books, selected_games


class Command(BaseCommand):
    help = 'Compare stored ModelPredictions with fresh, available SportsGameOdds points quotes'

    def add_arguments(self, parser):
        add_fetch_arguments(parser)
        parser.add_argument('--bookmaker', dest='bookmakers', help='Filter to a bookmaker; alias of --bookmakers')
        parser.set_defaults(bookmakers=None)
        parser.add_argument('--fresh', action='store_true', help='Compatibility flag; fetching current odds is now the default')
        parser.add_argument('--stored-only', action='store_true', help='Use stored quotes without API requests')
        parser.add_argument('--run', dest='run_id', help='PredictionRun UUID; default latest pregame forecast per player/game')
        parser.add_argument('--csv', help='Select an already-imported V2 run by its CSV; does not generate/import predictions')
        parser.add_argument('--max-age-minutes', type=int, default=30)
        parser.add_argument('--sort', choices=['abs-diff', 'player'], default='abs-diff')
        parser.add_argument('--details', action='store_true', help='Show import diagnostics and exclusion counts after the table')
        parser.add_argument('--no-record', action='store_true', help='Compatibility flag; comparisons no longer create database rows')

    def handle(self, *args, **options):
        choose_one = options['bookmakers'] is None
        options['bookmakers'] = options['bookmakers'] or 'draftkings,fanduel'
        books = parse_books(options['bookmakers'])
        if options['max_age_minutes'] <= 0:
            raise CommandError('--max-age-minutes must be positive')
        run_id = options['run_id']
        if options['csv']:
            if run_id:
                raise CommandError('Use --run or --csv, not both')
            try:
                with Path(options['csv']).open() as f:
                    ids = {r['run_id'] for r in csv.DictReader(f)}
                if len(ids) != 1:
                    raise ValueError()
                run_id = ids.pop()
            except (OSError, ValueError, KeyError):
                raise CommandError('--csv must reference one V2 prediction run') from None
        if run_id:
            try:
                run_id = uuid.UUID(str(run_id))
            except ValueError:
                raise CommandError('Invalid --run UUID') from None
            if not PredictionRun.objects.filter(pk=run_id).exists():
                raise CommandError('Run not found; import it with sync_population first')
        if options['stored_only'] and (options['fresh'] or options['refresh']):
            raise CommandError('--stored-only cannot be combined with --fresh or --refresh')
        diagnostics = StringIO()
        if not options['stored_only']:
            fetch_for_games(options, OutputWrapper(diagnostics), upcoming_only=True)
        rows, excluded, forecast_count = collect_comparisons(selected_games(options), books,
            timezone.now(), options['max_age_minutes'], run_id, not options['no_record'])
        if choose_one:
            # Count all usable player props for each whole local day, including
            # players without forecasts and games outside a --game-id selection.
            now = timezone.now()
            coverage = Counter()
            days = {timezone.localtime(p.game.scheduled_at).date() for p, _, _ in rows}
            for day in days:
                scope = {**options, 'date': day.isoformat(), 'days': 1, 'game_id': None}
                quotes = SportsbookLine.objects.filter(game__in=selected_games(scope),
                    bookmaker__in=books).select_related('game')
                for quote in quotes:
                    if not quote.game.is_locked(now) and not quote_exclusion(quote, now, options['max_age_minutes']):
                        coverage[(day, quote.bookmaker)] += 1
            rows = select_preferred_rows(rows, coverage)
        rows.sort(key=(lambda r: -abs(r[2])) if options['sort'] == 'abs-diff' else (lambda r: r[0].player.name))
        local_zone = ZoneInfo('America/New_York')
        day = options['date'] or timezone.localtime(timezone.now(), local_zone).date().isoformat()
        self.stdout.write(f'Predictions vs. Props: {day}')
        self.stdout.write('')
        headers = ('Player name', 'Pred', 'Line', 'Diff', 'Bookmaker', 'Game', 'Time', 'Over', 'Under')
        cells = []
        for forecast, quote, diff in rows:
            start = timezone.localtime(forecast.game.scheduled_at, local_zone)
            time_label = start.strftime('%I:%M%p').lstrip('0').lower() + ' ' + start.tzname()
            cells.append((forecast.player.name, f'{forecast.predicted_points:.1f}',
                f'{quote.line:.1f}', f'{diff:+.1f}', {'draftkings': 'DraftKings', 'fanduel': 'FanDuel'}.get(quote.bookmaker, quote.bookmaker),
                f'{forecast.game.away_team} @ {forecast.game.home_team}', time_label,
                f'{quote.over_price:+d}', f'{quote.under_price:+d}'))
        widths = [max(len(row[i]) for row in [headers, *cells]) for i in range(len(headers))]

        def format_row(row):
            return '  '.join(value.rjust(widths[i]) if i in (1, 2, 3, 7, 8) else value.ljust(widths[i])
                             for i, value in enumerate(row)).rstrip()

        self.stdout.write(format_row(headers))
        self.stdout.write('  '.join('-' * width for width in widths))
        for row in cells:
            self.stdout.write(format_row(row))
        if not cells:
            self.stdout.write('No available pregame comparisons for this date.')
        if options['details']:
            self.stdout.write('')
            if diagnostics.getvalue():
                self.stdout.write(diagnostics.getvalue().rstrip())
            matched = len({f.pk for f, _, _ in rows})
            self.stdout.write(f'Matched predictions={matched}/{forecast_count}; bookmaker comparisons={len(rows)}; '
                              f'excluded prediction/book pairs={dict(excluded)}')
