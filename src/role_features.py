"""Versioned current-role features; v1 definitions remain available unchanged.

Every player statistic is shifted within a season and uninterrupted team stint.
Team shares use full-game totals, not an estimate of who shared the court.
"""
import numpy as np
import pandas as pd
from population_features import player_features, attach_context, NUMERIC

SCHEMA = 'population-role-v2'
ROLE_NUMERIC = NUMERIC + [
    'role_games', 'role_current_weight', 'season_pts_avg', 'season_min_avg',
    'role_pts_avg', 'role_min_avg', 'role_pts_avg_5', 'role_min_avg_5',
    'role_fga_avg_5', 'role_fta_avg_5', 'role_fg3a_avg_5',
    'role_fg3a_per_min_10', 'role_opportunities_per_min_10',
    'team_season_games', 'opp_season_games',
]
SHARES = ['role_fga_share_10', 'role_opportunity_share_10']


def feature_columns(shares=True):
    return ROLE_NUMERIC + (SHARES if shares else []) + ['player_key']


def build_role_features(players, teams, prior_strength=2.5):
    if prior_strength <= 0:
        raise ValueError('prior_strength must be positive')
    # NaN outcomes identify inference rows; explicit nonappearances never count.
    players = players[players.MIN.isna() | players.MIN.gt(0)].copy()
    df = player_features(players)
    stint = df.groupby('PLAYER_ID').changed_team.cumsum()
    keys = [df.PLAYER_ID, df.SEASON, stint]
    group = df.groupby(keys, sort=False)
    df['role_games'] = group.cumcount()
    weight = df.role_games / (df.role_games + prior_strength)
    df['role_current_weight'] = weight
    for stat, prefix in [('PTS','pts'), ('MIN','min'), ('FGA','fga'), ('FTA','fta'), ('FG3A','fg3a')]:
        prior = ('prior_ppg' if stat == 'PTS' else 'prior_mpg' if stat == 'MIN' else None)
        fallback = df[prior] if prior else df[f'{prefix}_avg_10']
        fallback = fallback.fillna(df[f'{prefix}_avg_10'])
        df[f'role_{prefix}_avg'] = group[stat].transform(lambda s: s.shift().expanding().mean())
        for window in ([5,10,20] if stat == 'PTS' else [5,10]):
            value = group[stat].transform(lambda s: s.shift().rolling(window, min_periods=1).mean())
            df[f'role_{prefix}_avg_{window}'] = value
            # After an appearance, rolling features exclusively describe this role.
            df[f'{prefix}_avg_{window}'] = value.fillna(fallback)
        if prior:
            # Missing prior means observed current history gets all the weight.
            df[f'{prefix}_blend'] = (weight * df[f'role_{prefix}_avg'].fillna(fallback)
                                    + (1-weight) * fallback.fillna(df[f'role_{prefix}_avg']))
    for stat,prefix,span in [('PTS','pts',5),('PTS','pts',15),('MIN','min',10)]:
        value = group[stat].transform(lambda s: s.shift().ewm(span=span, adjust=False).mean())
        df[f'{prefix}_ewm_{span}'] = value.fillna(df[f'{prefix}_blend'])
    for stat,prefix in [('PTS','pts'),('MIN','min')]:
        df[f'{prefix}_std_10'] = group[stat].transform(lambda s: s.shift().rolling(10, min_periods=2).std()).fillna(0)
    # Prior shrinks the minutes estimate directly, rather than being counted again
    # through a cross-season EWMA. Recent five-appearance MPG defines the role.
    df['minutes_history_estimate'] = weight*df.min_avg_5 + (1-weight)*df.prior_mpg.fillna(df.min_avg_5)
    df['minutes_trend'] = df.min_avg_5 - df.min_avg_10
    for prefix in ['pts','fga','fta','fg3a']:
        rate = df[f'{prefix}_avg_10'] / df.min_avg_10.replace(0,np.nan)
        df[f'{prefix}_per_min_10' if prefix != 'fg3a' else 'role_fg3a_per_min_10'] = rate
    df['role_opportunities_per_min_10'] = df.fga_per_min_10 + .44*df.fta_per_min_10
    rate = weight*df.pts_per_min_10 + (1-weight)*df.prior_pts_per_min.fillna(df.pts_per_min_10)
    df['rate_points_baseline'] = df.minutes_history_estimate*rate

    totals = teams.set_index(['GAME_ID','TEAM_ABBREVIATION'])
    index = pd.MultiIndex.from_frame(df[['GAME_ID','TEAM_ABBREVIATION']])
    team_fga = totals.FGA.reindex(index).to_numpy()
    team_fta = totals.FTA.reindex(index).to_numpy()
    df['_fga_share'] = df.FGA / np.where(team_fga > 0, team_fga, np.nan)
    opportunities = team_fga + .44*team_fta
    df['_opportunity_share'] = (df.FGA + .44*df.FTA) / np.where(opportunities > 0, opportunities, np.nan)
    for source, dest in [('_fga_share',SHARES[0]),('_opportunity_share',SHARES[1])]:
        df[dest] = df.groupby(keys)[source].transform(lambda s: s.shift().rolling(10,min_periods=1).mean())

    # Three prior-season equivalent games for pace/defense; no last-season tail
    # in current-season rolling windows. As-of joins exclude the target date.
    from population_features import team_context
    ctx = team_context(teams)
    sources = ['allowed','total','possessions','def_rating']
    previous = ctx.groupby(['TEAM_ABBREVIATION','SEASON'])[sources].mean().reset_index()
    previous['SEASON'] = previous.SEASON.map(lambda s:f'{int(s[:4])+1}-{(int(s[:4])+2)%100:02d}')
    previous = previous.rename(columns={s:f'prior_{s}' for s in sources})
    ctx = ctx.merge(previous, on=['TEAM_ABBREVIATION','SEASON'],how='left',validate='many_to_one')
    tg = ctx.groupby(['TEAM_ABBREVIATION','SEASON'])
    ctx['context_games'] = tg.cumcount()+1
    for source in sources:
        recent = tg[source].transform(lambda s:s.rolling(10,min_periods=1).mean())
        n = ctx.context_games.clip(upper=10)
        ctx[f'opp_{source}_10'] = (n*recent+3*ctx[f'prior_{source}'].fillna(recent))/(n+3)
    # Start with original as-of context for zero-game season openers.
    result = attach_context(df,teams)
    for own in [False,True]:
        key = 'TEAM_ABBREVIATION' if own else 'opponent'
        cols = ['opp_possessions_10'] if own else [f'opp_{s}_10' for s in sources]
        rename = {'TEAM_ABBREVIATION':key, 'context_games':'team_season_games' if own else 'opp_season_games'}
        rename.update({col:f'new_{col}' for col in cols})
        right = ctx[['GAME_DATE','SEASON','TEAM_ABBREVIATION','context_games']+cols].rename(columns=rename)
        for col in [key,'SEASON']:
            result[col] = result[col].astype(pd.StringDtype(storage='python'))
            right[col] = right[col].astype(pd.StringDtype(storage='python'))
        result = pd.merge_asof(result.sort_values('GAME_DATE'),right.sort_values('GAME_DATE'),
                               on='GAME_DATE',by=[key,'SEASON'],allow_exact_matches=False)
        result[rename['context_games']] = result[rename['context_games']].fillna(0)
        for col in cols:
            dest = 'team_possessions_10' if own else col
            result[dest] = result.pop(f'new_{col}').fillna(result[dest])
    return result
