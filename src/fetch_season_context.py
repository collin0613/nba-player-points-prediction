"""Fetch current player/team assignments and the target season's NBA schedule."""
import pandas as pd
from nba_api.stats.endpoints import commonallplayers, scheduleleaguev2
from config import PLAYERS, PREDICT_SEASON, season_parser, raw_dir
from data_io import nba_call, read_logs, save_frame

def fetch_context(season, refresh=False):
    directory = raw_dir(season)
    roster_path, schedule_path = directory / 'roster.csv', directory / 'schedule.csv'
    if refresh or not roster_path.exists():
        roster = nba_call(commonallplayers.CommonAllPlayers, season=season,
                          is_only_current_season=1, league_id='00').get_data_frames()[0]
        required = {'PERSON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'ROSTERSTATUS'}
        if roster.empty or not required.issubset(roster.columns):
            raise ValueError(f'No valid current roster returned for {season}')
        # Do not substitute a team inferred from last season when a player is missing.
        roster = roster[roster.PERSON_ID.isin([p['player_id'] for p in PLAYERS])].copy()
        if roster.PERSON_ID.duplicated().any():
            raise ValueError('Ambiguous current team assignments')
        if len(roster) != len(PLAYERS):
            raise ValueError('Current roster is missing tracked players; verify NBA availability')
        if not (pd.to_numeric(roster.TEAM_ID).gt(0) & pd.to_numeric(roster.ROSTERSTATUS).eq(1)).all():
            raise ValueError('A tracked player has no active team; verify roster before prediction')
        save_frame(roster, roster_path, {'source': 'nba_api.CommonAllPlayers', 'season': season})
    else:
        roster = read_logs(roster_path)
    if refresh or not schedule_path.exists():
        schedule = nba_call(scheduleleaguev2.ScheduleLeagueV2, season=season, league_id='00').get_data_frames()[0]
        required = {'gameId', 'seasonYear', 'gameDateTimeUTC', 'gameStatus',
                    'homeTeam_teamId', 'awayTeam_teamId', 'homeTeam_teamTricode', 'awayTeam_teamTricode'}
        if schedule.empty or not required.issubset(schedule.columns):
            raise ValueError(f'No valid NBA schedule returned for {season}')
        if not schedule.seasonYear.astype(str).eq(season).all():
            raise ValueError(f'NBA returned a schedule for a different season than {season}')
        schedule['gameId'] = schedule.gameId.astype(str).str.zfill(10)
        # NBA regular-season game IDs begin 002; exclude preseason and playoffs.
        schedule = schedule[schedule.gameId.str.startswith('002')].copy()
        if schedule.empty or schedule.gameId.duplicated().any():
            raise ValueError('Empty or duplicate regular-season schedule')
        save_frame(schedule, schedule_path, {'source': 'nba_api.ScheduleLeagueV2', 'season': season})
    else:
        schedule = pd.read_csv(schedule_path, dtype={'gameId': str})
    return roster, schedule

def main():
    parser = season_parser(__doc__)
    parser.set_defaults(season=PREDICT_SEASON)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    roster, schedule = fetch_context(args.season, args.refresh)
    print(roster[['PERSON_ID', 'DISPLAY_FIRST_LAST', 'TEAM_ABBREVIATION']].to_string(index=False))
    print(f'{len(schedule)} scheduled regular-season games saved for {args.season}')

if __name__ == '__main__':
    main()
