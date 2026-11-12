"""Shared local-date filters and point-in-time validation for settled predictions."""
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from django.core.management.base import CommandError
from django.utils import timezone
from core.services.comparisons import quote_exclusion

LOCAL_TZ = ZoneInfo('America/New_York')


def add_date_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--date', help='Game date in New York (YYYY-MM-DD)')
    group.add_argument('--yesterday', action='store_true', help='Previous calendar day in New York')
    parser.add_argument('--since', help='First game date, inclusive (YYYY-MM-DD, New York)')
    parser.add_argument('--until', help='Last game date, inclusive (YYYY-MM-DD, New York)')


def date_range(options):
    if (options.get('date') or options.get('yesterday')) and (options.get('since') or options.get('until')):
        raise CommandError('Use --date/--yesterday or --since/--until, not both')
    def parse(value):
        if not value:
            return None
        try:
            return datetime.strptime(value, '%Y-%m-%d').date()
        except ValueError:
            raise CommandError('Dates must be valid YYYY-MM-DD values') from None
    day = (timezone.now().astimezone(LOCAL_TZ).date() - timedelta(days=1)
           if options.get('yesterday') else parse(options.get('date')))
    first, last = (day, day) if day else (parse(options.get('since')), parse(options.get('until')))
    if first and last and first > last:
        raise CommandError('--since must not be after --until')
    start = datetime.combine(first, time.min, LOCAL_TZ) if first else None
    end = datetime.combine(last + timedelta(days=1), time.min, LOCAL_TZ) if last else None
    return start, end


def filter_dates(queryset, field, bounds):
    start, end = bounds
    if start:
        queryset = queryset.filter(**{field+'__gte': start})
    if end:
        queryset = queryset.filter(**{field+'__lt': end})
    return queryset


def game_cutoff(game):
    return min(game.scheduled_at, game.actual_started_at) if game.actual_started_at else game.scheduled_at


def pregame(prediction):
    run = prediction.prediction_run
    cutoff = game_cutoff(prediction.game)
    return (run.generated_at < cutoff
            and run.feature_cutoff <= run.generated_at.astimezone(LOCAL_TZ).date()
            and run.feature_cutoff <= cutoff.astimezone(LOCAL_TZ).date())


def prediction_exclusion(prediction, include_backfills=False):
    historical = prediction.prediction_run.run_type == 'historical_backfill'
    if historical and not include_backfills:
        return 'historical_backfill_excluded'
    valid = pregame(prediction)
    if historical:
        run = prediction.prediction_run
        valid = (run.target_date == prediction.game.scheduled_at.astimezone(LOCAL_TZ).date()
                 and run.feature_cutoff == run.target_date
                 and run.generated_at >= game_cutoff(prediction.game))
    if not valid or prediction.prediction_status not in ('current', 'settled'):
        return 'non_pregame_or_invalid_context'
    if not prediction.result_verified_at or prediction.game.status != 'final':
        return 'awaiting_verified_final_result'
    if prediction.participation != 'played' or prediction.actual_points is None:
        return 'DNP_or_unknown'
    return None


def line_exclusion(line, max_age_minutes=None):
    cutoff = game_cutoff(line.game)
    if line.line_type == 'provider_closing':
        source = line.source_import
        if (source.source != 'sportsgameodds' or source.request_scope.get('line_type') != 'provider_closing'
                or source.request_scope.get('game_id') != line.game_id or line.game.status != 'final'):
            return 'unverified_closing_source'
        if line.line is None or line.over_price is None or line.under_price is None:
            return 'incomplete_closing_pair'
        if max_age_minutes is not None:
            return 'closing_quote_age_unknown'
        return None
    if line.observed_at >= cutoff:
        return 'post_tipoff_line'
    return quote_exclusion(line, cutoff, max_age_minutes)
