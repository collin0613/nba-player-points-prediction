"""Shared historical and next-game feature calculations."""
import pandas as pd
from config import MIN_THRESHOLD

def clean_player(df):
    df = df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='mixed')
    df = df.dropna(subset=['PTS', 'MIN', 'MATCHUP'])
    return df[df.MIN >= MIN_THRESHOLD].sort_values('GAME_DATE').reset_index(drop=True)

def team_games(df):
    df = df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='mixed')
    if df.duplicated(['GAME_ID', 'TEAM_ID']).any() or not df.groupby('GAME_ID').size().eq(2).all():
        raise ValueError('Team context requires exactly two distinct teams per game')
    df['game_total_pts'] = df.groupby('GAME_ID').PTS.transform('sum')
    df['PTS_ALLOWED'] = df.game_total_pts - df.PTS
    return df.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE']).reset_index(drop=True)

def historical_team_context(df):
    df = team_games(df)
    for source, target in [('PTS_ALLOWED', 'pts_allowed_avg_10'), ('game_total_pts', 'pace_proxy_avg_10')]:
        df[target] = df.groupby('TEAM_ABBREVIATION')[source].transform(lambda s: s.shift(1).rolling(10).mean())
    return df[['TEAM_ABBREVIATION', 'GAME_DATE', 'pts_allowed_avg_10', 'pace_proxy_avg_10']]

def recent_player_features(df, cutoff):
    df = clean_player(df)
    # Date-only NBA logs cannot prove an intraday game finished. Exclude today.
    df = df[df.GAME_DATE < pd.Timestamp(cutoff).normalize()]
    if len(df) < 10:
        raise ValueError(f'Need 10 qualifying historical appearances; found {len(df)}')
    return {
        'pts_avg_5': float(df.PTS.tail(5).mean()),
        'pts_avg_10': float(df.PTS.tail(10).mean()),
        'min_avg_5': float(df.MIN.tail(5).mean()),
        'last_game_date': df.GAME_DATE.iloc[-1],
        'history_games': len(df),
    }

def recent_team_features(df, team, cutoff):
    df = team_games(df)
    df = df[(df.TEAM_ABBREVIATION == team) & (df.GAME_DATE < pd.Timestamp(cutoff).normalize())].tail(10)
    if len(df) < 10:
        raise ValueError(f'Need 10 historical team games for {team}; found {len(df)}')
    return {'pts_allowed_avg_10': float(df.PTS_ALLOWED.mean()),
            'pace_proxy_avg_10': float(df.game_total_pts.mean())}
