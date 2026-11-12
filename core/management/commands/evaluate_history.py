"""Evaluate retained V2 pregame predictions against verified results and stored props."""
import math
from collections import Counter, defaultdict
from django.core.management.base import BaseCommand, CommandError
from core.models import Game, ModelPrediction, SportsbookLine
from core.services.results import backfill_games, cleanup_dnps
from django.utils import timezone
from core.services.history import (LOCAL_TZ, add_date_arguments, date_range, filter_dates,
                                   prediction_exclusion, line_exclusion)
from core.services.odds_fetch import parse_books
from core.management.commands.evaluate_forecasts import metrics
from src.market_mapping import normalize_player_name


def threshold_diff(diff):
    """Use the same one-decimal magnitude as the displayed Diff column."""
    return float(f'{abs(diff):.1f}')


class Command(BaseCommand):
    help = 'Backfill selected games and evaluate V2 predictions: points error, rounded exact hits, and O/U wins/losses/pushes'

    def add_arguments(self, parser):
        add_date_arguments(parser)
        parser.add_argument('--prediction-kind', choices=['live','backfill','all'], default='all',
                            help='Separate genuine pregame forecasts from historical reconstructions')
        parser.add_argument('--stored-only', action='store_true', help='Skip backfill; evaluate stored results without API requests')
        parser.add_argument('--stat', choices=['points'], default='points', help='V1 compatibility; only points are supported')
        parser.add_argument('--bookmaker', default='DraftKings', help='DraftKings or FanDuel (case insensitive)')
        parser.add_argument('--diff-threshold', '--diff_threshold', dest='diff_threshold', type=float, default=0,
                            help='O/U only: abs(prediction - line) >= threshold; displayed one-decimal diff, inclusive')
        parser.add_argument('--player', help='Exact normalized player name')
        parser.add_argument('--season', help='Optional season, e.g. 2026-27')
        parser.add_argument('--model-version', help='Evaluate only this saved model version')
        parser.add_argument('--per-player', action='store_true')
        parser.add_argument('--min-samples', type=int, default=1, help='Minimum decided O/U outcomes per player')
        parser.add_argument('--top-n', type=int, default=25)
        parser.add_argument('--show-points', action='store_true', help='Show all verified point predictions, including missing odds')
        parser.add_argument('--show-rows', action='store_true', help='Print only O/U rows that meet all selection filters')
        parser.add_argument('--max-age-minutes', type=int,
                            help='Optional pre-tipoff quote age limit; default uses latest retained pregame lines of any age')

    def handle(self, *args, **options):
        bounds = date_range(options)
        threshold = options['diff_threshold']
        if not math.isfinite(threshold) or threshold < 0:
            raise CommandError('--diff-threshold must be finite and nonnegative')
        if options['min_samples'] < 1 or options['top_n'] < 1:
            raise CommandError('--min-samples and --top-n must be positive')
        age = options['max_age_minutes']
        if age is not None and age <= 0:
            raise CommandError('--max-age-minutes must be positive')
        books = parse_books(options['bookmaker'])
        if len(books) != 1:
            raise CommandError('Select one --bookmaker per evaluation to avoid counting a player twice')
        book = books[0]
        qs = filter_dates(ModelPrediction.objects.select_related('game','player','prediction_run'), 'game__scheduled_at', bounds)
        kind = options['prediction_kind']
        if kind != 'all':
            qs = qs.filter(prediction_run__run_type='historical_backfill' if kind == 'backfill' else 'live')
        if options['season']:
            qs = qs.filter(game__season=options['season'])
        if options['model_version']:
            qs = qs.filter(prediction_run__model_version=options['model_version'])
        player_key = normalize_player_name(options['player']) if options['player'] else None
        totals = None
        if not options['stored_only']:
            cleanup_scope = qs
            if player_key:
                cleanup_scope = qs.filter(player_id__in=[p.player_id for p in qs if normalize_player_name(p.player.name) == player_key])
            cleanup_dnps(cleanup_scope, timezone.now())
            pending = qs.filter(result_verified_at__isnull=True, game__scheduled_at__lte=timezone.now())
            if player_key:
                player_ids = [p.player_id for p in pending if normalize_player_name(p.player.name) == player_key]
                pending = pending.filter(player_id__in=player_ids)
            games = Game.objects.filter(pk__in=pending.values('game_id'))
            if games.exists():
                self.stdout.write(f'Backfilling results for {games.count()} selected game(s)...')
                totals = backfill_games(games, self.stdout, self.stderr)

        excluded, eligible = Counter(), []
        for prediction in qs.order_by('game__scheduled_at','player__name'):
            if player_key and normalize_player_name(prediction.player.name) != player_key:
                continue
            reason = prediction_exclusion(prediction, include_backfills=kind in ('all', 'backfill'))
            if reason:
                excluded[reason] += 1
            else:
                eligible.append(prediction)
        lines = {(q.game_id,q.player_id):q for q in SportsbookLine.objects.filter(
            game_id__in={p.game_id for p in eligible},bookmaker=book).select_related('game','source_import')}
        candidates = []
        line_types = Counter()
        outcomes, by_player, rows, pairs = Counter(), defaultdict(Counter), [], []
        for p in eligible:
            line = lines.get((p.game_id,p.player_id))
            reason = 'missing_line' if line is None else line_exclusion(line,age)
            if reason:
                excluded[reason] += 1
                continue
            line_types[line.line_type] += 1
            diff = p.predicted_points - float(line.line)
            if diff == 0:
                excluded['zero_diff_no_pick'] += 1
                continue
            pick = 'OVER' if diff > 0 else 'UNDER'
            result = ('PUSH' if p.actual_points == line.line else
                      'WIN' if (p.actual_points > line.line) == (diff > 0) else 'LOSS')
            candidates.append((p,line,diff,pick,result))
            if threshold_diff(diff) < threshold:
                excluded['below_diff_threshold'] += 1
                continue
            outcomes[result] += 1
            by_player[p.player_id][result] += 1
            rows.append((p,line,diff,pick,result))
            pairs.append((p.predicted_points,float(line.line),p.actual_points))
        from datetime import timedelta
        def box(lines):
            self.stdout.write('')
            width = max(map(len, lines))
            self.stdout.write('+' + '-' * (width + 2) + '+')
            for text in lines:
                self.stdout.write('| ' + text.ljust(width) + ' |')
            self.stdout.write('+' + '-' * (width + 2) + '+')
            self.stdout.write('')
        first = str(bounds[0].date()) if bounds[0] else None
        last = str((bounds[1]-timedelta(days=1)).date()) if bounds[1] else None
        label = (first if first == last else f'{first} -- {last}') if first and last else (f'Since {first}' if first else f'Through {last}' if last else 'All time')
        box([f'Prediction History: {label}', f'Prediction kind: {kind}'])
        for run_type in ('live', 'historical_backfill'):
            subset = [p for p in eligible if p.prediction_run.run_type == run_type]
            if subset:
                self.stdout.write(f'{run_type}: '+metrics([(p.predicted_points,p.actual_points) for p in subset]))
        decided = outcomes['WIN'] + outcomes['LOSS']
        rate = f'{outcomes["WIN"]/decided:.1%}' if decided else 'N/A'
        hits = sum(round(p.predicted_points) == p.actual_points for p in eligible)
        exact_rate = f'{hits/len(eligible):.1%}' if eligible else 'N/A'
        box([f'O/U Win Rate: {rate} | (excluding pushes | diff_threshold >= {threshold:g})',
             f'N={len(rows)} | Wins: {outcomes["WIN"]} | Losses: {outcomes["LOSS"]} | Pushes: {outcomes["PUSH"]}',
             '------------------------------------------------',
             f'Exact points accuracy (rounded): {hits}/{len(eligible)} ({exact_rate})'])
        if options['per_player']:
            names = {p.player_id:p.player.name for p in eligible}
            self.stdout.write('\nPlayer                         Decided   W   L   P    Win%')
            selected = sorted(by_player.items(),key=lambda item:(-(item[1]['WIN']+item[1]['LOSS']),names[item[0]]))
            selected = [(pid,c) for pid,c in selected if c['WIN']+c['LOSS'] >= options['min_samples']][:options['top_n']]
            for pid,c in selected:
                n=c['WIN']+c['LOSS']
                self.stdout.write(f'{names[pid]:30} {n:7} {c["WIN"]:3} {c["LOSS"]:3} {c["PUSH"]:3} {c["WIN"]/n:7.1%}')
            if not selected: self.stdout.write('(No players meet the minimum decided sample count.)')
        if options['show_points']:
            self.stdout.write('\nVerified points results (independent of odds):')
            for p in eligible:
                self.stdout.write(f'{p.game.scheduled_at.astimezone(LOCAL_TZ):%Y-%m-%d} {p.game.nba_game_id} '
                    f'{p.player.name:30} pred={p.predicted_points:.1f} actual={p.actual_points} '
                    f'error={p.predicted_points-p.actual_points:+.1f} {p.prediction_run.run_type}')
            self.stdout.write('')
        if options['show_rows']:
            headers = ('Player name','Game','Date/time','Pred','Line','Diff','Actual','Result')
            cells = [headers]
            for p,line,diff,pick,result in sorted(rows,key=lambda r:(-abs(r[2]),r[0].player.name,r[0].pk)):
                mark = {'WIN':'✅','LOSS':'❌','PUSH':'PUSH'}[result]
                cells.append((p.player.name,f'{p.game.away_team} @ {p.game.home_team}',
                    f'{p.game.scheduled_at.astimezone(LOCAL_TZ):%Y-%m-%d %I:%M%p %Z}',
                    f'{p.predicted_points:.1f}',f'{line.line:.1f}',f'{diff:+.1f}',str(p.actual_points),f'{pick:<5} {mark}'))
            widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
            for i,row in enumerate(cells):
                self.stdout.write('  '.join(value.ljust(widths[j]) for j,value in enumerate(row)).rstrip())
                if i == 0:
                    self.stdout.write('  '.join('-'*w for w in widths))
            self.stdout.write('')
        box(['Prediction model: '+(metrics([(p,a) for p,l,a in pairs]) if pairs else 'N/A'),
             'Sportsbook performance: '+(metrics([(l,a) for p,l,a in pairs]) if pairs else 'N/A'),
             '------------------------------------------------',
             'Exact points performance: '+(metrics([(p.predicted_points,p.actual_points) for p in eligible]) if eligible else 'N/A')])
        required = max(len(candidates)//10,3)
        best = None
        counts = Counter()
        ordered = sorted(candidates,key=lambda r:-abs(r[2]))
        from itertools import groupby
        from fractions import Fraction
        for diff, group in groupby(ordered,key=lambda r:threshold_diff(r[2])):
            counts.update(r[4] for r in group)
            n_decided = counts['WIN'] + counts['LOSS']
            n_total = sum(counts.values())
            if n_decided < required:
                continue
            key = (Fraction(counts['WIN'],n_decided),n_decided,n_total,-diff)
            if best is None or key > best[0]:
                best = (key,diff,counts.copy())
        if best:
            key,diff,c = best
            box([f'Optimal Diff ({label}): {diff:.1f}',
                 f'Win rate = {float(key[0]):.1%}, O/U: n={sum(c.values())} | Wins: {c["WIN"]} | Losses: {c["LOSS"]} | Pushes: {c["PUSH"]}',
                 f'(Optimal diff must meet at least {required} decided predictions).'])
        else:
            box([f'Optimal Diff ({label}): N/A — insufficient samples',
                 f'(Optimal diff must meet at least {required} decided predictions).'])
        self.stdout.write('')
        self.stdout.write(f'Bookmaker: {book}')
        self.stdout.write(f'Eligible line sources: {dict(line_types)}')
        if line_types['provider_closing']:
            self.stdout.write('Provider closing lines: event-start prices retrieved later; historical comparison, not proof of a wager placed.')
        if totals is not None:
            self.stdout.write(f'Backfill: {dict(totals)}')
        self.stdout.write(f'Exclusions: {dict(excluded)}')
        self.stdout.write('Historical backfills are reconstructions, not live pregame recommendations.' if kind != 'live' else
                          'Evaluates retained pregame state, not every prior recommendation or actual bet.')
        self.stdout.write('')
