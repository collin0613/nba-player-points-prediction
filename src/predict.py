"""Prediction entry point: league-wide by default; --legacy selects this module.

The default pipeline accepts --predict-date YYYY-MM-DD to reconstruct an exact
New York game date with strictly earlier features, then settle stored past games.

Legacy: generate pregame predictions using current rosters and cross-season history.

CSV output is independent of PostgreSQL and sportsbook availability. Use --store
for the existing Django/odds workflow. Run update_actual_points separately.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import joblib
import pandas as pd
from dotenv import load_dotenv
from config import (ROOT, PLAYERS, NUM_FEATURES, CAT_FEATURES, TRAIN_SEASON,
                    PREDICT_SEASON, model_path, season_value, raw_dir, player_path, league_path)
from data_io import fetch_player, fetch_league, save_frame
from features import recent_player_features, recent_team_features
from fetch_season_context import fetch_context

LOCAL_TZ = ZoneInfo('America/New_York')

def select_next_game(schedule, team_id, now, lookahead_days=14):
    frame = schedule.copy()
    frame['start'] = pd.to_datetime(frame.gameDateTimeUTC, utc=True, errors='coerce')
    home = pd.to_numeric(frame.homeTeam_teamId, errors='coerce').eq(team_id)
    away = pd.to_numeric(frame.awayTeam_teamId, errors='coerce').eq(team_id)
    eligible = ((home | away) & pd.to_numeric(frame.gameStatus, errors='coerce').eq(1)
                & frame.start.gt(now) & frame.start.le(now + timedelta(days=lookahead_days)))
    frame = frame[eligible].sort_values('start')
    if frame.empty:
        raise ValueError(f'No future scheduled game for team {team_id} in the next {lookahead_days} days')
    return frame.iloc[0]

def combine_history(prior, current, team=False):
    frame = pd.concat([prior, current], ignore_index=True)
    keys = ['GAME_ID', 'TEAM_ID'] if team else ['Game_ID', 'Player_ID']
    return frame.drop_duplicates(keys, keep='last')

def create_predictions(season, history_season, offline=False):
    now = datetime.now(timezone.utc)
    today = now.astimezone(LOCAL_TZ).date()
    if int(history_season[:4]) >= int(season[:4]):
        raise ValueError('History season must precede prediction season')
    path = model_path(history_season)
    pipeline = joblib.load(path)
    metadata = json.loads(path.with_suffix('.json').read_text())
    if metadata.get('season') != history_season:
        raise ValueError('Model metadata season does not match selected history season')
    if metadata['features'] != NUM_FEATURES + CAT_FEATURES:
        raise ValueError('Model feature schema does not match prediction code; retrain first')
    if pd.Timestamp(metadata['data_through']).date() >= today:
        raise ValueError('Model training data must precede the prediction cutoff')
    if offline:
        required = [raw_dir(season) / 'roster.csv', raw_dir(season) / 'schedule.csv',
                    league_path(history_season)] + [player_path(history_season, p['player_name']) for p in PLAYERS]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError('Offline inputs missing: ' + ', '.join(missing))
    roster, schedule = fetch_context(season, refresh=not offline)
    # No target-season logs are needed before its first regular-season game.
    starts = pd.to_datetime(schedule.gameDateTimeUTC, utc=True, errors='coerce')
    season_started = starts.min() < now
    prior_teams = fetch_league(history_season)
    if offline and season_started:
        required = [league_path(season)] + [player_path(season, p['player_name']) for p in PLAYERS]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError('Refresh current-season logs online before using --offline: ' + ', '.join(missing))
    if season_started:
        current_teams = fetch_league(season, refresh=not offline, allow_empty=True)
        team_history = combine_history(prior_teams, current_teams, team=True)
    else:
        team_history = prior_teams
    results, failures = [], []
    for player in PLAYERS:
        name, player_id = player['player_name'], player['player_id']
        try:
            membership = roster[roster.PERSON_ID.astype(int).eq(player_id)]
            if len(membership) != 1 or int(membership.iloc[0].ROSTERSTATUS) != 1:
                raise ValueError('No unambiguous active current roster assignment')
            membership = membership.iloc[0]
            team_id = int(membership.TEAM_ID)
            game = select_next_game(schedule, team_id, now)
            is_home = int(game.homeTeam_teamId) == team_id
            opponent = game.awayTeam_teamTricode if is_home else game.homeTeam_teamTricode
            game_date = game.start.tz_convert(LOCAL_TZ).date()
            prior = fetch_player(player, history_season)
            current_count = 0
            if season_started:
                current = fetch_player(player, season, refresh=not offline, allow_empty=True)
                current_count = len(current)
                history = combine_history(prior, current)
            else:
                history = prior
            recent = recent_player_features(history, today)
            features = {key: recent[key] for key in ['pts_avg_5', 'pts_avg_10', 'min_avg_5']}
            features.update(recent_team_features(team_history, opponent, today))
            features.update(is_home=int(is_home),
                            rest_days=float(min(5, max(0, (pd.Timestamp(game_date) - recent['last_game_date']).days))),
                            player_name=name)
            X = pd.DataFrame([features])[NUM_FEATURES + CAT_FEATURES]
            prediction = float(pipeline.predict(X)[0])
            results.append({
                'player_name': name, 'player_id': player_id,
                'team': str(membership.TEAM_ABBREVIATION), 'opponent': str(opponent),
                'nba_game_id': str(game.gameId).zfill(10),
                'next_game_date_local': str(game_date),
                'next_game_start_utc': game.start.isoformat(),
                'home_team': str(game.homeTeam_teamTricode), 'away_team': str(game.awayTeam_teamTricode),
                'predicted_at_utc': now.isoformat(), 'feature_cutoff_date_exclusive': str(today),
                'model_version': path.stem, 'history_season': history_season,
                'prediction_season': season, 'current_season_appearances': current_count,
                'last_player_game_date': str(recent['last_game_date'].date()),
                'roster_checked_for_injuries': False,
                **features, 'predicted_pts': prediction,
                'odds_event_id': '',
            })
            print(f'{name}: {prediction:.1f} points, {membership.TEAM_ABBREVIATION} vs {opponent} on {game_date}')
        except ValueError as exc:
            failures.append({'player': name, 'reason': str(exc)})
    if failures:
        # Never silently label a partial seven-player run as complete.
        print(json.dumps({'skipped_players': failures}, indent=2))
    if not results:
        raise RuntimeError('No eligible predictions generated')
    return pd.DataFrame(results), failures

def store_predictions(frame, bookmakers=None):
    """Adapt the explicit legacy model to the same canonical persistence service."""
    import uuid
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tracker.settings')
    import django
    django.setup()
    from django.core.management import call_command
    frame = frame.copy()
    frame['run_id'] = str(uuid.uuid4())
    feature_columns = [c for c in NUM_FEATURES + CAT_FEATURES if c in frame]
    frame['features_json'] = frame.apply(lambda row: json.dumps({
        key: None if pd.isna(row[key]) else (row[key].item() if hasattr(row[key], 'item') else row[key]) for key in feature_columns}, allow_nan=False), axis=1)
    season = frame.prediction_season.iloc[0]
    path = ROOT / 'data/predictions' / season / (frame.run_id.iloc[0] + '.csv')
    save_frame(frame, path)
    call_command('sync_population', season=season, csv=str(path))


def main():
    load_dotenv(ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season', type=season_value, default=PREDICT_SEASON)
    parser.add_argument('--history-season', type=season_value, default=TRAIN_SEASON)
    parser.add_argument('--offline', action='store_true', help='Reuse cached roster, schedule and logs')
    parser.add_argument('--store', action='store_true', help='Also save to Django (odds fetched separately)')
    parser.add_argument('--bookmakers', default='DraftKings')
    args = parser.parse_args()
    frame, failures = create_predictions(args.season, args.history_season, args.offline)
    output = ROOT / f'predictions_next_games_{args.season}.csv'
    save_frame(frame, output)
    print(f'Saved {len(frame)} predictions -> {output}')
    print('Predictions are conditional on participation; injuries and starting lineups are not modeled.')
    if args.store:
        store_predictions(frame, args.bookmakers)
        save_frame(frame, output)
    if failures:
        raise SystemExit(2)

if __name__ == '__main__':
    if '--legacy' in sys.argv:
        sys.argv.remove('--legacy')
        main()
    else:
        from predict_population import main as population_main
        population_main()
