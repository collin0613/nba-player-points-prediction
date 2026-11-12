"""Verified final NBA Stats V3 boxscores attached to canonical predictions."""
from datetime import timedelta
import math
import re
from django.db import transaction
from nba_api.stats.endpoints import boxscoresummaryv3, boxscoretraditionalv3
from core.models import Game, ModelPrediction
from core.services.imports import record_import
from src.market_mapping import normalize_team

SOURCE = 'nba_api.BoxScoreSummaryV3 + BoxScoreTraditionalV3'


def fetch_final_boxscore(game_id):
    try:
        summary = boxscoresummaryv3.BoxScoreSummaryV3(game_id=game_id, timeout=25).get_dict()
        if summary.get('boxScoreSummary', {}).get('gameStatus') != 3:
            return summary, None
        stats = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, timeout=25).get_dict()
        return summary, stats
    except Exception as exc:
        raise ValueError(f'NBA boxscore request failed ({type(exc).__name__})') from None


def parse_final(game, summary_payload, stats_payload):
    summary = summary_payload.get('boxScoreSummary', {})
    if summary.get('gameStatus') != 3:
        return None
    stats = (stats_payload or {}).get('boxScoreTraditional', {})
    if summary.get('gameId') != game.nba_game_id or stats.get('gameId') != game.nba_game_id:
        raise ValueError('NBA boxscore game ID mismatch')
    results = {}
    for side, expected_team in [('homeTeam', game.home_team), ('awayTeam', game.away_team)]:
        team, final_team = stats.get(side, {}), summary.get(side, {})
        expected_team = normalize_team(expected_team)
        if (not expected_team or normalize_team(team.get('teamTricode')) != expected_team
                or normalize_team(final_team.get('teamTricode')) != expected_team
                or team.get('teamId') != final_team.get('teamId')):
            raise ValueError('NBA boxscore team mismatch')
        players = team.get('players') or []
        if not players:
            raise ValueError('Final boxscore has no player rows')
        total = 0
        for row in players:
            player_id = int(row['personId'])
            if player_id in results:
                raise ValueError('Duplicate player in final boxscore')
            statistics = row.get('statistics') or {}
            minutes = statistics.get('minutes')
            comment = (row.get('comment') or '').upper()
            if minutes and re.fullmatch(r'\d+:\d{2}(?:\.\d+)?', str(minutes)):
                # A nonempty 0:00 can be a recorded sub-second appearance.
                played = True
                points = statistics.get('points')
                if isinstance(points, bool) or not isinstance(points, (int, float)) or not math.isfinite(points) or points < 0 or int(points) != points:
                    raise ValueError('Invalid final points total')
                points = int(points)
                total += points
            elif not minutes and comment.startswith(('DNP', 'DND', 'NWT', 'INACTIVE', 'NOT WITH')):
                played, points = False, None
            else:
                # Missing/unknown participation must not be scored as zero.
                continue
            results[player_id] = {'played': played, 'points': points,
                'name': ' '.join(filter(None, [row.get('firstName'), row.get('familyName')]))}
        if total != final_team.get('score'):
            raise ValueError('Player totals disagree with verified final team score; boxscore may be incomplete')
    return results


@transaction.atomic
def save_results(game, results, fetched_at, payload=None):
    game = Game.objects.select_for_update().get(pk=game.pk)
    item, _ = record_import('nba', 'results', fetched_at, {'nba_game_id': game.nba_game_id},
                            payload or results, row_count=len(results))
    updated, missing = 0, 0
    for prediction in ModelPrediction.objects.filter(game=game):
        if prediction.result_verified_at and prediction.result_verified_at >= fetched_at:
            continue
        row = results.get(prediction.player_id)
        if row is None:
            missing += 1
            continue
        if prediction.prediction_status == 'current':
            prediction.prediction_status = 'settled'
        prediction.actual_points = row['points'] if row['played'] else None
        prediction.participation = 'played' if row['played'] else 'dnp'
        prediction.result_verified_at = fetched_at
        prediction.result_import = item
        prediction.locked_at = prediction.locked_at or game.scheduled_at
        prediction.save(update_fields=['prediction_status','actual_points','participation','result_verified_at','result_import','locked_at'])
        updated += 1
    game.status = 'final'; game.save(update_fields=['status'])
    deleted = cleanup_dnps(ModelPrediction.objects.filter(game=game), fetched_at)
    return {'predictions_settled': updated, 'dnps_deleted': deleted, 'missing_player_results': missing}


def backfill_games(games, stdout, stderr, *, dry_run=False, verbose=False):
    """Shared final-result workflow for standalone backfill and history evaluation."""
    from collections import Counter
    from django.utils import timezone
    totals = Counter()
    for game in games.order_by('scheduled_at'):
        try:
            summary, stats = fetch_final_boxscore(game.nba_game_id)
            result = parse_final(game, summary, stats)
            if result is None:
                totals['not_final'] += 1
                if verbose:
                    stdout.write(f'NOT FINAL {game}: no results written')
                continue
            if dry_run:
                stdout.write(f'DRY RUN {game}: {len(result)} verified player results; no writes')
            else:
                counts = save_results(game, result, timezone.now(), {'summary': summary, 'boxscore': stats})
                totals.update(counts)
                if verbose:
                    stdout.write(f'{game}: {counts}')
            totals['final_games'] += 1
        except (ValueError, KeyError, TypeError) as exc:
            totals['skipped'] += 1
            stderr.write(f'SKIP {game}: {exc}')
    return totals


def cleanup_dnps(predictions, now, *, dry_run=False):
    """Only remove verified nonparticipants after 24 elapsed hours."""
    ids = [p.pk for p in predictions.filter(participation='dnp', actual_points__isnull=True,
        result_verified_at__isnull=False, game__status='final').select_related('game', 'prediction_run')
        if p.prediction_run.run_type != 'historical_backfill' and now - (p.game.actual_started_at or p.game.scheduled_at) > timedelta(hours=24)]
    if not dry_run:
        ModelPrediction.objects.filter(pk__in=ids).delete()
    return len(ids)
