"""Validated, resumable NBA downloads. No credentials are logged."""
import json
import time
from datetime import datetime, timezone
import pandas as pd
from nba_api.stats.endpoints import playergamelog, leaguegamelog
from config import player_path, league_path

def nba_call(endpoint, **kwargs):
    for attempt in range(3):
        try:
            time.sleep(0.7)
            return endpoint(timeout=25, **kwargs)
        except Exception as exc:
            if attempt == 2:
                raise RuntimeError(f'{endpoint.__name__} failed after 3 attempts ({type(exc).__name__})') from None
            time.sleep(2 * (attempt + 1))

def read_logs(path):
    return pd.read_csv(path, dtype={'Game_ID': str, 'GAME_ID': str})

def save_frame(df, path, metadata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    if metadata is not None:
        path.with_suffix('.meta.json').write_text(json.dumps({
            **metadata, 'fetched_at': datetime.now(timezone.utc).isoformat(), 'rows': len(df)
        }, indent=2) + '\n')

def validate_logs(df, season, team=False, allow_empty=False):
    columns = {'GAME_DATE', 'MATCHUP', 'PTS', 'MIN', 'FGA', 'FTA', 'TOV'}
    columns |= {'GAME_ID', 'TEAM_ID', 'TEAM_ABBREVIATION'} if team else {'Game_ID', 'Player_ID'}
    if not columns.issubset(df.columns):
        raise ValueError(f'Missing NBA columns: {sorted(columns - set(df.columns))}')
    if df.empty:
        if allow_empty:
            return
        raise ValueError(f'No regular-season games returned for {season}; refusing an empty dataset')
    dates = pd.to_datetime(df.GAME_DATE, format='mixed')
    start = int(season[:4])
    if not dates.between(f'{start}-10-01', f'{start+1}-06-30').all():
        raise ValueError(f'Game dates do not belong to season {season}')
    if df[['GAME_DATE', 'PTS', 'MIN', 'MATCHUP']].isna().any().any():
        raise ValueError('Incomplete game result rows')
    keys = ['GAME_ID', 'TEAM_ID'] if team else ['Game_ID', 'Player_ID']
    if df.duplicated(keys).any():
        raise ValueError('Duplicate NBA game rows')
    if team and not df.groupby('GAME_ID').size().eq(2).all():
        raise ValueError('Expected exactly two team rows per completed game')

def fetch_player(player, season, refresh=False, allow_empty=False):
    path = player_path(season, player['player_name'])
    downloaded = not path.exists() or refresh
    if downloaded:
        df = nba_call(playergamelog.PlayerGameLog, player_id=player['player_id'],
                      season=season, season_type_all_star='Regular Season').get_data_frames()[0]
    else:
        df = read_logs(path)
    validate_logs(df, season, allow_empty=allow_empty)
    if not df.empty and not df.Player_ID.astype(int).eq(player['player_id']).all():
        raise ValueError('Player ID mismatch')
    if downloaded:
        save_frame(df, path, {'source': 'nba_api.PlayerGameLog', 'season': season,
                              'player_id': player['player_id'], 'season_type': 'Regular Season'})
    return df

def fetch_league(season, refresh=False, allow_empty=False):
    path = league_path(season)
    downloaded = not path.exists() or refresh
    if downloaded:
        df = nba_call(leaguegamelog.LeagueGameLog, season=season,
                      season_type_all_star='Regular Season', player_or_team_abbreviation='T').get_data_frames()[0]
    else:
        df = read_logs(path)
    validate_logs(df, season, team=True, allow_empty=allow_empty)
    if downloaded:
        save_frame(df, path, {'source': 'nba_api.LeagueGameLog', 'season': season,
                              'season_type': 'Regular Season'})
    return df
