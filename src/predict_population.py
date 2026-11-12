"""League-wide predictions with immutable local runs and default Django persistence."""
import argparse
import json
import uuid
from datetime import datetime, timezone, date
from zoneinfo import ZoneInfo
import joblib
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import commonallplayers,scheduleleaguev2
from config import ROOT,raw_dir,season_value
from data_io import nba_call,save_frame,fetch_league
from fetch_population import population_logs,make_watchlist,save_watchlist
from population_features import load_history,player_features,attach_context,SEASONS,FEATURES,NUMERIC
from predict import select_next_game

LOCAL_TZ=ZoneInfo('America/New_York')


def resolve_model(cutoff, model_version=None):
    """Choose a version without changing original artifacts or past-date defaults."""
    original = 'points_population_2026-27'
    pointer = ROOT/'models/active_population_2026-27.json'
    version = model_version or original
    if model_version is None and pointer.exists():
        active = json.loads(pointer.read_text())
        deployed = pd.Timestamp(active['deployed_at']).tz_convert(LOCAL_TZ).date()
        if cutoff.date() >= deployed:
            version = active['model_version']
    if not isinstance(version,str) or not version or '/' in version or '\\' in version or version in ('.','..'):
        raise ValueError('Invalid model version')
    artifact = ROOT/'models'/f'{version}.joblib'
    metadata = json.loads(artifact.with_suffix('.json').read_text())
    schema = metadata.get('feature_schema_version','population-v1')
    if schema == 'population-v1':
        columns = FEATURES
    elif schema == 'population-role-v2':
        from role_features import feature_columns
        columns = feature_columns(metadata['feature_config']['shares'])
    else:
        raise ValueError(f'Unknown feature schema: {schema}')
    if metadata['features'] != columns or pd.Timestamp(metadata['data_through']) >= cutoff:
        raise ValueError('Model schema mismatch or training cutoff is not before prediction date')
    return artifact, metadata, columns

def refresh_roster_schedule(season):
    roster=nba_call(commonallplayers.CommonAllPlayers,season=season,is_only_current_season=1,league_id='00').get_data_frames()[0]
    if roster.empty or roster.PERSON_ID.duplicated().any():
        raise ValueError('Empty or ambiguous current roster')
    schedule=nba_call(scheduleleaguev2.ScheduleLeagueV2,season=season,league_id='00').get_data_frames()[0]
    if schedule.empty or not schedule.seasonYear.astype(str).eq(season).all():
        raise ValueError('Empty or wrong-season schedule')
    schedule['gameId']=schedule.gameId.astype(str).str.zfill(10)
    schedule=schedule[schedule.gameId.str.startswith('002')].copy()
    if schedule.empty or schedule.gameId.duplicated().any():
        raise ValueError('Invalid regular-season schedule')
    save_frame(roster,raw_dir(season)/'all_roster.csv',{'source':'nba_api.CommonAllPlayers','season':season})
    save_frame(schedule,raw_dir(season)/'schedule.csv',{'source':'nba_api.ScheduleLeagueV2','season':season})

def parse_predict_date(value):
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError()
        return parsed
    except ValueError:
        raise argparse.ArgumentTypeError('predict-date must be YYYY-MM-DD') from None


def select_date_game(schedule, team_id, day, now):
    frame = schedule.copy()
    frame['start'] = pd.to_datetime(frame.gameDateTimeUTC, utc=True, errors='coerce')
    member = (pd.to_numeric(frame.homeTeam_teamId).eq(team_id)
              | pd.to_numeric(frame.awayTeam_teamId).eq(team_id))
    eligible = member & frame.start.dt.tz_convert(LOCAL_TZ).dt.date.eq(day)
    eligible &= pd.to_numeric(frame.gameStatus).isin([1, 2, 3])
    if 'gameStatusText' in frame:
        eligible &= ~frame.gameStatusText.fillna('').str.contains('postponed|cancel', case=False)
    if day >= now.astimezone(LOCAL_TZ).date():
        eligible &= frame.start.gt(now) & pd.to_numeric(frame.gameStatus).eq(1)
    games = frame[eligible]
    if games.empty:
        return None
    if len(games) != 1:
        raise ValueError(f'Ambiguous games for team {team_id} on {day}')
    return games.iloc[0]


def historical_watchlist(season, cutoff, schedule):
    """Use a complete archived selection observed before the target day.

    Older runs contain the eligible player/team list, not a full roster snapshot.
    Carry it forward explicitly; never select participants from target boxscores.
    """
    candidates = []
    for path in (ROOT/'data/predictions'/season).glob('*.json'):
        info = json.loads(path.read_text())
        if info.get('skipped') != [] or info.get('target_date') or not path.with_suffix('.csv').exists():
            continue
        observed = info.get('roster_metadata', {}).get('fetched_at')
        if not observed or pd.Timestamp(observed).tz_convert(LOCAL_TZ).date() >= cutoff.date():
            continue
        frame = pd.read_csv(path.with_suffix('.csv'))
        if (frame.empty or len(frame) != info.get('rows')
                or frame.player_id.duplicated().any()
                or pd.to_datetime(frame.predicted_at_utc, utc=True).max().tz_convert(LOCAL_TZ).date() >= cutoff.date()
                or info.get('run_type', 'live') != 'live'):
            continue
        candidates.append((pd.Timestamp(observed), path, frame))
    if not candidates:
        raise ValueError('No complete pre-date roster evidence; historical prediction requires an archived pre-date run')
    observed, path, frame = max(candidates, key=lambda x: (x[0], str(x[1])))
    ids = {}
    for side in ('home', 'away'):
        for row in schedule.itertuples():
            ids[str(getattr(row, side+'Team_teamTricode'))] = int(getattr(row, side+'Team_teamId'))
    watch = frame[['player_id','player_name','team']].copy()
    watch['team_id'] = watch.team.map(ids)
    if watch.team_id.isna().any():
        raise ValueError('Archived team not present in season schedule')
    return watch, {'source': 'archived eligible roster carried forward',
                   'path': str(path), 'fetched_at': observed.isoformat()}


def truncate_history(history, teams, cutoff):
    history, teams = history.copy(), teams.copy()
    for frame in (history, teams):
        frame['GAME_DATE'] = pd.to_datetime(frame.GAME_DATE, format='mixed')
    return (history[(history.GAME_DATE < cutoff) & (history.MIN > 0)].copy(),
            teams[teams.GAME_DATE < cutoff].copy())


def create_predictions(season='2026-27',offline=False,predict_date=None,model_version=None):
    if season!='2026-27':
        raise ValueError('This trained population model targets 2026-27; rebuild for a different season')
    now=datetime.now(timezone.utc)
    today=pd.Timestamp(now.astimezone(LOCAL_TZ).date())
    if predict_date and predict_date > today.date():
        raise ValueError('--predict-date must be today or a past date')
    cutoff = pd.Timestamp(predict_date) if predict_date else today
    historical = cutoff < today
    if not offline:
        refresh_roster_schedule(season)
    roster=pd.read_csv(raw_dir(season)/'all_roster.csv')
    schedule=pd.read_csv(raw_dir(season)/'schedule.csv',dtype={'gameId':str})
    history,teams=load_history()
    if historical:
        watch, roster_metadata = historical_watchlist(season, cutoff, schedule)
    else:
        watch,audit=make_watchlist(history[history.SEASON=='2025-26'],roster)
        save_watchlist(watch,audit,season)
        roster_metadata = json.loads((raw_dir(season)/'all_roster.meta.json').read_text())
    started=pd.to_datetime(schedule.gameDateTimeUTC,utc=True).min()<now
    if started:
        if offline and not all(p.exists() for p in [raw_dir(season)/'all_player_games.csv',raw_dir(season)/'league/gamelog.csv']):
            raise FileNotFoundError('Refresh current-season data online before using --offline')
        current=population_logs(season,refresh=not offline,allow_empty=True)
        current_teams=fetch_league(season,refresh=not offline,allow_empty=True)
        current_teams['SEASON']=season
        history=pd.concat([history,current],ignore_index=True)
        teams=pd.concat([teams,current_teams],ignore_index=True)
    history, teams = truncate_history(history, teams, cutoff)
    artifact,metadata,columns=resolve_model(cutoff,model_version)
    model=joblib.load(artifact)
    query=[]; skipped=[]
    games={}
    for row in watch.itertuples():
        try:
            game=(select_date_game(schedule,int(row.team_id),predict_date,now) if predict_date
                  else select_next_game(schedule,int(row.team_id),now))
            if game is None:
                continue
            if not history.PLAYER_ID.eq(row.player_id).any():
                raise ValueError('No historical appearances')
            home=int(game.homeTeam_teamId)==int(row.team_id)
            opponent=game.awayTeam_teamTricode if home else game.homeTeam_teamTricode
            game_id=str(game.gameId).zfill(10)
            query.append({'PLAYER_ID':row.player_id,'PLAYER_NAME':row.player_name,'TEAM_ID':row.team_id,
                          'TEAM_ABBREVIATION':row.team,'GAME_ID':game_id,
                          'GAME_DATE':pd.Timestamp(game.start.tz_convert(LOCAL_TZ).date()),'SEASON':season,
                          'MATCHUP':f"{row.team} {'vs.' if home else '@'} {opponent}",
                          'PTS':np.nan,'MIN':np.nan,'FGA':np.nan,'FTA':np.nan,'FG3A':np.nan,'IS_QUERY':True})
            games[(row.player_id,game_id)]=game
        except ValueError as exc:
            skipped.append({'player_id':row.player_id,'player_name':row.player_name,'reason':str(exc)})
    if not query:
        raise ValueError('No eligible games for the requested prediction date/window')
    history['IS_QUERY']=False
    combined=pd.concat([history,pd.DataFrame(query)],ignore_index=True)
    if metadata.get('feature_schema_version')=='population-role-v2':
        from role_features import build_role_features
        frame=build_role_features(combined,teams,prior_strength=metadata['feature_config']['prior_strength'])
    else:
        frame=attach_context(player_features(combined),teams)
    inputs=frame[frame.IS_QUERY.eq(True)].copy()
    predictions=np.maximum(0,model.predict(inputs[columns]))
    run_id=str(uuid.uuid4())
    rows=[]
    for (_,row),points in zip(inputs.iterrows(),predictions):
        game=games[(int(row.PLAYER_ID),str(row.GAME_ID))]
        feature_values={key:(None if pd.isna(row[key]) else (str(row[key]) if key=='player_key' else float(row[key]))) for key in columns}
        rows.append({'run_id':run_id,'player_id':int(row.PLAYER_ID),'player_name':row.PLAYER_NAME,
                     'team':row.TEAM_ABBREVIATION,'opponent':row.opponent,'nba_game_id':str(row.GAME_ID),
                     'home_team':game.homeTeam_teamTricode,'away_team':game.awayTeam_teamTricode,
                     'next_game_date_local':str(game.start.tz_convert(LOCAL_TZ).date()),
                     'next_game_start_utc':game.start.isoformat(),'prediction_season':season,
                     'predicted_at_utc':now.isoformat(),'feature_cutoff_date_exclusive':str(cutoff.date()),
                     'run_type':'historical_backfill' if historical else 'live',
                     'target_date':str(predict_date) if predict_date else '',
                     'roster_observed_at':roster_metadata['fetched_at'],
                     'roster_source':roster_metadata.get('path',roster_metadata['source']),
                     'model_version':artifact.stem,'selected_model':metadata['selected_model'],
                     'predicted_pts':float(points),'minutes_history_estimate':float(row.minutes_history_estimate),
                     'minutes_source':'historical estimate, not provider projection',
                     'prior_mpg':float(row.prior_mpg),'prior_games':int(row.prior_games),
                     'small_sample':bool(row.prior_games<10),'injuries_checked':False,
                     'features_json':json.dumps(feature_values,allow_nan=False)})
    return pd.DataFrame(rows),skipped,{**metadata, 'roster_provenance':roster_metadata}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season',default='2026-27',type=season_value)
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--model-version',help='Explicit saved model version; defaults to the deployed version for live dates')
    parser.add_argument('--predict-date',type=parse_predict_date,help='Exact New York game date; past dates reconstruct pre-date features and settle results')
    storage=parser.add_mutually_exclusive_group()
    storage.add_argument('--store',dest='store',action='store_true',help='Save predictions to Django (default)')
    storage.add_argument('--no-store',dest='store',action='store_false',help='Export files only')
    parser.set_defaults(store=True)
    args=parser.parse_args()
    frame,skipped,metadata=create_predictions(args.season,args.offline,args.predict_date,args.model_version)
    run_id=frame.run_id.iloc[0]
    directory=ROOT/'data/predictions'/args.season
    path=directory/f'{run_id}.csv'
    save_frame(frame,path)
    save_frame(frame,ROOT/(f'predictions_population_{args.season}_{args.predict_date}.csv' if args.predict_date
                          else f'predictions_population_{args.season}.csv'))
    path.with_suffix('.json').write_text(json.dumps({'run_id':run_id,'rows':len(frame),'skipped':skipped,
        'run_type':frame.run_type.iloc[0], 'target_date':str(args.predict_date) if args.predict_date else None,
        'model_metadata':metadata,'roster_metadata':metadata['roster_provenance']},indent=2)+'\n')
    print(f'{len(frame)} predictions, {frame.team.nunique()} teams; model={metadata["selected_model"]}; skipped={len(skipped)}')
    print(f'Saved run CSV: {path}')
    today = str(args.predict_date or datetime.now(LOCAL_TZ).date())
    preview = frame[frame.next_game_date_local.eq(today)]
    if not preview.empty:
        print(f'\nPredictions for {today}:')
        print(preview[['player_name', 'team', 'opponent', 'predicted_pts']].to_string(index=False, float_format=lambda value: f'{value:.1f}'))
    if skipped:
        print(json.dumps(skipped,indent=2))
    if args.store:
        import os,sys,django
        sys.path.insert(0,str(ROOT))
        os.environ.setdefault('DJANGO_SETTINGS_MODULE','tracker.settings')
        django.setup()
        from django.core.management import call_command
        from django.core.management.base import CommandError
        from django.db import DatabaseError
        try:
            call_command('sync_population',season=args.season,csv=str(path),historical=frame.run_type.iloc[0]=='historical_backfill')
        except (CommandError, DatabaseError) as exc:
            print(f'CSV saved, but database import failed ({type(exc).__name__}).')
            if isinstance(exc, CommandError):
                print(str(exc))
            print(f'Retry: python manage.py sync_population --season {args.season} --csv \"{path}\"')
            raise SystemExit(1) from None
        if frame.run_type.iloc[0] == 'historical_backfill':
            from core.services.historical import complete_historical_run
            if not complete_historical_run(frame.nba_game_id.unique().tolist(), str(args.predict_date), offline=args.offline):
                raise SystemExit(1)
    if skipped:
        raise SystemExit(2)

if __name__=='__main__': main()
