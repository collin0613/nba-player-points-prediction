"""Compare revised candidates with untouched stored pregame forecasts.

Uses original forecast-time teams and feature cutoffs. This is a reconstruction
using corrected cached boxscores, not a newly observed pregame track record.
"""
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from config import ROOT
from population_features import load_history
from role_features import build_role_features, feature_columns
from evaluate_models import metrics
from recalibrate import OUT, SPECS, segment_scores


def replay():
    original=pd.DataFrame(json.loads((OUT/'original_live.json').read_text()))
    players,teams=load_history(['2023-24','2024-25','2025-26','2026-27'])
    names={r.TEAM_ABBREVIATION:int(r.TEAM_ID) for r in teams.itertuples()}
    names.update({r.TEAM_ABBREVIATION:int(r.TEAM_ID) for r in pd.read_csv(ROOT/'data/raw/2026-27/all_roster.csv').itertuples()})
    selected=json.loads((OUT/'selection.json').read_text())['selected_model']
    # Selected model plus the explicitly requested >=2/3-weight challenger.
    challengers=list(dict.fromkeys([selected,'role_mae']))
    models={name:joblib.load(OUT/f'{name}.joblib') for name in challengers}
    rebuilt=[]
    for (cutoff,game_date),rows in original.groupby(['feature_cutoff','GAME_DATE']):
        queries=[]
        for row in rows.itertuples():
            opponent=row.away_team if row.team==row.home_team else row.home_team
            queries.append(dict(PLAYER_ID=row.PLAYER_ID,PLAYER_NAME=row.PLAYER_NAME,TEAM_ID=names[row.team],
                TEAM_ABBREVIATION=row.team,SEASON='2026-27',GAME_ID=row.GAME_ID,GAME_DATE=pd.Timestamp(row.GAME_DATE),
                MATCHUP=f"{row.team} {'vs.' if row.team==row.home_team else '@'} {opponent}",
                PTS=np.nan,MIN=np.nan,FGA=np.nan,FTA=np.nan,FG3A=np.nan,IS_QUERY=True))
        # Each player has one forecast per cutoff in retained data. Otherwise run
        # separate queries so an unplayed query cannot count as an appearance.
        query=pd.DataFrame(queries)
        if query.PLAYER_ID.duplicated().any():
            raise ValueError('Multiple future games per player/cutoff require separate replay batches')
        prior=players[players.GAME_DATE.lt(cutoff)&players.PLAYER_ID.isin(query.PLAYER_ID)].copy();prior['IS_QUERY']=False
        for name in challengers:
            spec=SPECS[name]
            frame=build_role_features(pd.concat([prior,query],ignore_index=True),teams[teams.GAME_DATE.lt(cutoff)],
                                      prior_strength=spec.get('prior_strength',2.5))
            inputs=frame[frame.IS_QUERY.eq(True)].copy()
            inputs['revised_prediction']=np.maximum(0,models[name].predict(inputs[feature_columns(spec.get('shares',True))]))
            merged=rows.merge(inputs[['GAME_ID','PLAYER_ID','revised_prediction','role_games','role_min_avg',
                                      'prior_mpg','season_games','changed_team_since_prior','pts_avg_5','pts_blend',
                                      'minutes_history_estimate']],on=['GAME_ID','PLAYER_ID'],validate='one_to_one')
            merged['model']=name
            rebuilt.append(merged)
        print(f'Replayed {cutoff} for {game_date}: {len(rows)} original forecasts',flush=True)
    paired=pd.concat(rebuilt,ignore_index=True)
    summaries=[]; segments=[]
    for name,g in paired.groupby('model'):
        summaries.append({'model':name,**metrics(g.PTS,g.revised_prediction)})
        segments.extend(segment_scores(g,g.revised_prediction.to_numpy(),name,'original_live_cutoffs'))
    g=paired[paired.model.eq(selected)]
    summaries.extend([{'model':'original_stored',**metrics(g.PTS,g.original_prediction)},
                      {'model':'original_rolling5',**metrics(g.PTS,g.features.map(lambda x:x['pts_avg_5']))},
                      {'model':'original_blend',**metrics(g.PTS,g.features.map(lambda x:x['pts_blend']))},
                      {'model':'role_rolling5',**metrics(g.PTS,g.pts_avg_5)},
                      {'model':'role_blend',**metrics(g.PTS,g.pts_blend)}])
    segments.extend(segment_scores(g,g.original_prediction.to_numpy(),'original_stored','original_live_cutoffs'))
    # Confidence interval for paired errors, resampling whole game dates.
    errors=pd.DataFrame({'date':g.GAME_DATE,'delta':abs(g.revised_prediction-g.PTS)-abs(g.original_prediction-g.PTS)})
    daily=errors.groupby('date').delta.agg(['sum','count'])
    rng=np.random.default_rng(42);draws=rng.integers(0,len(daily),size=(4000,len(daily)))
    boot=daily['sum'].to_numpy()[draws].sum(axis=1)/daily['count'].to_numpy()[draws].sum(axis=1)
    report={'selected_model':selected,'metrics':summaries,
            'selected_minus_original_mae_date_bootstrap_95pct':np.quantile(boot,[.025,.975]).tolist(),
            'interpretation':'Development replay, not an untouched test. Corrected raw logs; original cutoffs and teams. Negative MAE difference favors revised.'}
    (OUT/'paired_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    paired.drop(columns='features').to_csv(OUT/'paired_predictions.csv',index=False)
    pd.DataFrame(segments).to_csv(OUT/'paired_segments.csv',index=False)
    players=[]
    for pid,rows in g.groupby('PLAYER_ID'):
        if len(rows)>=5:
            players.append({'player_id':pid,'player_name':rows.PLAYER_NAME.iloc[0],'appearances':len(rows),
                'original_mae':float(abs(rows.original_prediction-rows.PTS).mean()),
                'revised_mae':float(abs(rows.revised_prediction-rows.PTS).mean()),
                'original_bias':float((rows.original_prediction-rows.PTS).mean()),
                'revised_bias':float((rows.revised_prediction-rows.PTS).mean()),
                'mean_minutes_change':float((rows.role_min_avg-rows.prior_mpg).mean())})
    pd.DataFrame(players).to_csv(OUT/'paired_players.csv',index=False)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        replay()
