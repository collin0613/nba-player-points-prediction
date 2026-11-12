"""Point-in-time features shared by league-wide training and live inference.

Eligibility uses prior information. A game's actual minutes are never used to
select or describe that same game's prediction. Outcomes condition on appearance.
"""
import numpy as np
import pandas as pd
from config import ROOT
from data_io import read_logs, save_frame

SEASONS = ['2023-24', '2024-25', '2025-26']
NUMERIC = [
    'is_home','rest_days','days_since_appearance','season_games','prior_games',
    'pts_avg_5','pts_avg_10','pts_avg_20','pts_ewm_5','pts_ewm_15',
    'min_avg_5','min_avg_10','min_ewm_10','min_std_10','pts_std_10',
    'prior_ppg','prior_mpg','pts_blend','min_blend','minutes_history_estimate',
    'minutes_trend','fga_avg_10','fta_avg_10','fg3a_avg_10','pts_per_min_10',
    'fga_per_min_10','fta_per_min_10','prior_pts_per_min','rate_points_baseline',
    'changed_team','changed_team_since_prior','games_with_team',
    'opp_allowed_10','opp_total_10','opp_possessions_10','opp_def_rating_10',
    'team_possessions_10',
]
CATEGORICAL = ['player_key']
FEATURES = NUMERIC + CATEGORICAL
LEGACY_NUMERIC = ['is_home','rest_days','pts_avg_5','pts_avg_10','min_avg_5','opp_allowed_10','opp_total_10']

def load_history(seasons=SEASONS):
    player_frames, team_frames = [], []
    for season in seasons:
        directory=ROOT/'data'/'raw'/season
        players=read_logs(directory/'all_player_games.csv')
        players['SEASON']=season
        player_frames.append(players)
        teams=read_logs(directory/'league/gamelog.csv')
        teams['SEASON']=season
        team_frames.append(teams)
    players=pd.concat(player_frames,ignore_index=True)
    teams=pd.concat(team_frames,ignore_index=True)
    for df in (players,teams):
        df['GAME_DATE']=pd.to_datetime(df.GAME_DATE,format='mixed')
        df['GAME_ID']=df.GAME_ID.astype(str).str.zfill(10)
    players=players[pd.to_numeric(players.MIN,errors='coerce')>0].copy()
    if players.duplicated(['PLAYER_ID','GAME_ID']).any():
        raise ValueError('Duplicate historical player games')
    return players,teams

def player_features(frame):
    df=frame.sort_values(['PLAYER_ID','GAME_DATE','GAME_ID']).reset_index(drop=True).copy()
    df['GAME_DATE']=pd.to_datetime(df.GAME_DATE)
    g=df.groupby('PLAYER_ID',sort=False)
    # Previous season summaries are joined to the FOLLOWING season only.
    actual=df[df.PTS.notna() & df.MIN.notna()]
    prior=actual.groupby(['PLAYER_ID','SEASON']).agg(
        prior_ppg=('PTS','mean'),prior_mpg=('MIN','mean'),prior_games=('PTS','size'),
        prior_points=('PTS','sum'),prior_minutes=('MIN','sum'), prior_team=('TEAM_ID','last')).reset_index()
    prior['SEASON']=prior.SEASON.map(lambda s:f'{int(s[:4])+1}-{(int(s[:4])+2)%100:02d}')
    df=df.merge(prior,on=['PLAYER_ID','SEASON'],how='left',validate='many_to_one')
    g=df.groupby('PLAYER_ID',sort=False)
    df['history_games']=g.cumcount()
    df['player_key']=df.PLAYER_ID.astype(str)
    df['is_home']=df.MATCHUP.str.contains('vs.',regex=False).astype(int)
    df['opponent']=df.MATCHUP.str.split().str[-1].astype(str)
    df['days_since_appearance']=g.GAME_DATE.diff().dt.days.clip(0,60).fillna(5)
    df['rest_days']=df.days_since_appearance.clip(upper=5)
    previous_team=g.TEAM_ID.shift()
    df['changed_team']=(previous_team.notna() & previous_team.ne(df.TEAM_ID)).astype(int)
    df['changed_team_since_prior']=(df.prior_team.notna() & df.prior_team.ne(df.TEAM_ID)).astype(int)
    stint=df.groupby('PLAYER_ID').changed_team.cumsum()
    df['games_with_team']=df.groupby([df.PLAYER_ID,stint]).cumcount()
    for stat,prefix in [('PTS','pts'),('MIN','min'),('FGA','fga'),('FTA','fta'),('FG3A','fg3a')]:
        windows=[5,10,20] if stat=='PTS' else ([5,10] if stat=='MIN' else [10])
        for window in windows:
            df[f'{prefix}_avg_{window}']=g[stat].transform(lambda s:s.shift().rolling(window,min_periods=1).mean())
    for stat,prefix,span in [('PTS','pts',5),('PTS','pts',15),('MIN','min',10)]:
        df[f'{prefix}_ewm_{span}']=g[stat].transform(lambda s:s.shift().ewm(span=span,adjust=False).mean())
    for stat,prefix in [('PTS','pts'),('MIN','min')]:
        df[f'{prefix}_std_10']=g[stat].transform(lambda s:s.shift().rolling(10,min_periods=2).std()).fillna(0)
    sg=df.groupby(['PLAYER_ID','SEASON'],sort=False)
    df['season_games']=sg.cumcount()
    for stat,prefix in [('PTS','pts'),('MIN','min')]:
        sums=sg[stat].transform(lambda s:s.shift().fillna(0).cumsum())
        prior_values=df['prior_ppg' if stat=='PTS' else 'prior_mpg']
        prior_values=prior_values.fillna(df[f'{prefix}_avg_10'])
        df[f'{prefix}_blend']=(sums+10*prior_values)/(df.season_games+10)
        df[f'season_{prefix}_avg']=sums/df.season_games.replace(0,np.nan)
    df['minutes_history_estimate']=.5*df.min_ewm_10+.5*df.min_blend
    df['minutes_trend']=df.min_avg_5-df.min_avg_10
    for stat,prefix in [('PTS','pts'),('FGA','fga'),('FTA','fta')]:
        df[f'{prefix}_per_min_10']=df[f'{prefix}_avg_10']/df.min_avg_10.replace(0,np.nan)
    df['prior_pts_per_min']=df.prior_points/df.prior_minutes.replace(0,np.nan)
    rate=.5*df.pts_per_min_10+.5*df.prior_pts_per_min.fillna(df.pts_per_min_10)
    df['rate_points_baseline']=df.minutes_history_estimate*rate
    df['eligible']=(df.prior_mpg.ge(15) | (df.season_games.ge(5)&df.season_min_avg.ge(15))) & df.history_games.ge(5)
    return df

def team_context(frame):
    df=frame.sort_values(['TEAM_ABBREVIATION','GAME_DATE']).copy()
    if not df.groupby('GAME_ID').size().eq(2).all():
        raise ValueError('Missing team-game pair')
    df['allowed']=df.groupby('GAME_ID').PTS.transform('sum')-df.PTS
    df['total']=df.groupby('GAME_ID').PTS.transform('sum')
    df['possessions']=df.FGA+.44*df.FTA-df.OREB+df.TOV
    # Approximate shared game possessions, normalized to 48 minutes for overtime.
    shared=df.groupby('GAME_ID').possessions.transform('mean')
    df['possessions']=shared*240/df.MIN
    df['def_rating']=100*df.allowed/shared.replace(0,np.nan)
    for source in ['allowed','total','possessions','def_rating']:
        # Inclusive result used only by a strict earlier-date asof join below.
        df[f'opp_{source}_10']=df.groupby('TEAM_ABBREVIATION')[source].transform(lambda s:s.rolling(10,min_periods=1).mean())
    return df

def attach_context(players, teams):
    # API/CSV inputs and concatenated prediction rows can carry different pandas
    # string dtypes. merge_asof requires identical group-key dtypes on both sides.
    players=players.copy()
    teams=teams.copy()
    for frame,columns in [(players,['TEAM_ABBREVIATION','opponent']),
                          (teams,['TEAM_ABBREVIATION'])]:
        for column in columns:
            frame[column]=frame[column].astype(pd.StringDtype(storage='python'))
    ctx=team_context(teams)
    columns=['opp_allowed_10','opp_total_10','opp_possessions_10','opp_def_rating_10']
    left=players.sort_values('GAME_DATE')
    right=ctx[['GAME_DATE','TEAM_ABBREVIATION']+columns].rename(columns={'TEAM_ABBREVIATION':'opponent'})
    df=pd.merge_asof(left,right.sort_values('GAME_DATE'),on='GAME_DATE',by='opponent',allow_exact_matches=False)
    own=ctx[['GAME_DATE','TEAM_ABBREVIATION','opp_possessions_10']].rename(columns={'opp_possessions_10':'team_possessions_10'})
    return pd.merge_asof(df.sort_values('GAME_DATE'),own.sort_values('GAME_DATE'),on='GAME_DATE',by='TEAM_ABBREVIATION',allow_exact_matches=False)

def build_dataset():
    players,teams=load_history()
    df=attach_context(player_features(players),teams)
    # First season supplies priors only. No filtering on a target game's minutes.
    df=df[df.eligible & df.SEASON.ne(SEASONS[0]) & df.PTS.notna()].copy()
    df=df.sort_values(['GAME_DATE','PLAYER_ID']).reset_index(drop=True)
    output=ROOT/'data/processed/population/features.csv'
    save_frame(df,output)
    print(f'Population dataset: {len(df)} rows, {df.PLAYER_ID.nunique()} players, {df.GAME_DATE.min().date()}–{df.GAME_DATE.max().date()}',flush=True)
    return df

if __name__=='__main__': build_dataset()
