"""Download league-wide history and build a >=15 prior-season MPG watchlist."""
import argparse
import json
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, commonallplayers
from config import ROOT, raw_dir, season_value
from data_io import nba_call, read_logs, save_frame, fetch_league

def population_logs(season, refresh=False, allow_empty=False):
    path = raw_dir(season) / 'all_player_games.csv'
    if path.exists() and not refresh:
        df = read_logs(path)
    else:
        df = nba_call(leaguegamelog.LeagueGameLog, season=season, league_id='00',
                      player_or_team_abbreviation='P', season_type_all_star='Regular Season').get_data_frames()[0]
        if (df.empty and not allow_empty) or not {'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'GAME_ID', 'MIN', 'PTS'}.issubset(df):
            raise ValueError(f'No valid league-wide player logs for {season}')
        if df.duplicated(['PLAYER_ID', 'GAME_ID']).any():
            raise ValueError('Duplicate player/game rows')
        dates = pd.to_datetime(df.GAME_DATE, format='mixed')
        year = int(season[:4])
        if not dates.between(f'{year}-10-01', f'{year+1}-06-30').all():
            raise ValueError('Wrong-season data returned')
        save_frame(df, path, {'source': 'nba_api.LeagueGameLog PlayerOrTeam=P', 'season': season})
    df['SEASON'] = season
    df['GAME_ID'] = df.GAME_ID.astype(str).str.zfill(10)
    return df

def make_watchlist(history, roster, minimum=15):
    # Weighted across all team stints: one row per played game, not mean of team averages.
    games = history[pd.to_numeric(history.MIN, errors='coerce') > 0].copy()
    averages = games.groupby('PLAYER_ID').agg(prior_mpg=('MIN','mean'), prior_ppg=('PTS','mean'),
                                            prior_games=('GAME_ID','size')).reset_index()
    current = roster[(pd.to_numeric(roster.ROSTERSTATUS, errors='coerce') == 1)
                     & (pd.to_numeric(roster.TEAM_ID, errors='coerce') > 0)].copy()
    if current.PERSON_ID.duplicated().any():
        raise ValueError('Ambiguous current roster assignments')
    audit = current.merge(averages, left_on='PERSON_ID', right_on='PLAYER_ID', how='left', validate='one_to_one')
    audit['eligible'] = audit.prior_mpg.ge(minimum)
    audit['exclusion_reason'] = ''
    audit.loc[audit.prior_mpg.isna(), 'exclusion_reason'] = 'No prior-season NBA appearances (includes rookies)'
    audit.loc[audit.prior_mpg.lt(minimum), 'exclusion_reason'] = 'Prior-season MPG below threshold'
    audit['small_sample'] = audit.prior_games.fillna(0).lt(10)
    audit = audit.rename(columns={'PERSON_ID':'player_id', 'DISPLAY_FIRST_LAST':'player_name',
                                 'TEAM_ID':'team_id', 'TEAM_ABBREVIATION':'team'})
    cols = ['player_id','player_name','team_id','team','prior_mpg','prior_ppg','prior_games',
            'small_sample','eligible','exclusion_reason']
    audit = audit[cols].sort_values(['team','prior_mpg'], ascending=[True,False])
    watch = audit[audit.eligible].copy()
    watch['team_minutes_rank'] = watch.groupby('team').cumcount() + 1
    watch['top_five_by_prior_minutes'] = watch.team_minutes_rank.le(5)
    return watch, audit

def save_watchlist(watch, audit, prediction_season='2026-27', history_season='2025-26', minimum=15):
    directory=ROOT/'data'/'processed'/prediction_season
    save_frame(watch,directory/'player_watchlist.csv')
    save_frame(audit,directory/'roster_eligibility.csv')
    save_frame(watch[watch.top_five_by_prior_minutes],directory/'top_five_by_prior_minutes.csv')
    counts=watch.groupby('team').size()
    summary=watch.groupby('team').agg(players=('player_id','size'),minimum_mpg=('prior_mpg','min'),maximum_mpg=('prior_mpg','max')).reset_index()
    save_frame(summary,directory/'team_coverage.csv')
    if len(counts)!=30:
        raise ValueError(f'Watchlist covers only {len(counts)}/30 teams; inspect roster data')
    (directory/'watchlist_metadata.json').write_text(json.dumps({
        'history_season':history_season,'prediction_season':prediction_season,
        'minimum_prior_mpg':minimum,'players':len(watch),'teams':len(counts),
        'by_team':counts.to_dict(),'selection':'Active roster plus prior-season average minutes; not confirmed starters',
    },indent=2)+'\n')
    print(f'Watchlist: {len(watch)} players, {len(counts)} teams; range {counts.min()}–{counts.max()} per team',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seasons', nargs='+', type=season_value, default=['2023-24','2024-25','2025-26'])
    parser.add_argument('--prediction-season', type=season_value, default='2026-27')
    parser.add_argument('--min-mpg', type=float, default=15)
    parser.add_argument('--refresh', action='store_true')
    args=parser.parse_args()
    for season in args.seasons:
        df=population_logs(season,args.refresh)
        teams=fetch_league(season,args.refresh)
        print(f'{season}: {len(df)} player games, {df.PLAYER_ID.nunique()} players, {len(teams)} team games', flush=True)
    path=raw_dir(args.prediction_season)/'all_roster.csv'
    if args.refresh or not path.exists():
        roster=nba_call(commonallplayers.CommonAllPlayers, season=args.prediction_season,
                        is_only_current_season=1, league_id='00').get_data_frames()[0]
        if roster.empty:
            raise ValueError('Empty target-season roster')
        save_frame(roster,path,{'source':'nba_api.CommonAllPlayers','season':args.prediction_season})
    else:
        roster=pd.read_csv(path)
    history_season=max(args.seasons)
    watch,audit=make_watchlist(population_logs(history_season),roster,args.min_mpg)
    save_watchlist(watch,audit,args.prediction_season,history_season,args.min_mpg)

if __name__=='__main__': main()
