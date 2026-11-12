"""Join current predictions to current lines; displaying a comparison writes no rows."""
from collections import Counter
from datetime import timedelta
from core.models import ModelPrediction, SportsbookLine


def quote_exclusion(quote, now, max_age_minutes):
    if quote.market_status != 'available':
        return quote.market_status
    if quote.line is None or quote.over_price is None or quote.under_price is None:
        return 'incomplete'
    if not quote.over_available or not quote.under_available:
        return 'unavailable'
    oldest = now - timedelta(minutes=max_age_minutes) if max_age_minutes is not None else None
    if quote.observed_at > now or (oldest is not None and quote.observed_at < oldest):
        return 'stale_observation'
    for stamp in (quote.over_updated_at, quote.under_updated_at):
        if stamp and (stamp > now or (oldest is not None and stamp < oldest)):
            return 'stale_bookmaker_quote'
    return None


def collect_comparisons(games, books, now, max_age_minutes=30, run_id=None, record=False):
    predictions = ModelPrediction.objects.filter(game__in=games).select_related('player','game','prediction_run')
    if run_id:
        predictions = predictions.filter(prediction_run_id=run_id)
    lines = {(q.game_id, q.player_id, q.bookmaker): q for q in SportsbookLine.objects.filter(
        game__in=games, bookmaker__in=books).select_related('game')}
    result, excluded = [], Counter()
    count = 0
    for p in predictions:
        count += 1
        for book in books:
            if p.game.is_locked(now) or p.locked_at:
                excluded['game_not_pregame'] += 1; continue
            if p.prediction_status != 'current':
                excluded[p.prediction_status] += 1; continue
            if p.prediction_run.generated_at > now or p.prediction_run.generated_at >= p.game.scheduled_at:
                excluded['non_pregame_forecast'] += 1; continue
            q = lines.get((p.game_id, p.player_id, book))
            reason = 'missing_or_unmatched_player_line' if q is None else quote_exclusion(q, now, max_age_minutes)
            if reason:
                excluded[reason] += 1; continue
            result.append((p, q, p.predicted_points - float(q.line)))
    return result, excluded, count


def select_preferred_rows(rows, coverage):
    """One quote per player/game: absolute edge, daily coverage, then DraftKings."""
    from django.utils import timezone
    chosen = {}
    def rank(row):
        prediction, quote, diff = row
        day = timezone.localtime(prediction.game.scheduled_at).date()
        return (abs(diff), coverage[(day, quote.bookmaker)], quote.bookmaker == 'draftkings')
    for row in rows:
        prediction = row[0]
        key = (prediction.game_id, prediction.player_id)
        if key not in chosen or rank(row) > rank(chosen[key]):
            chosen[key] = row
    return list(chosen.values())
