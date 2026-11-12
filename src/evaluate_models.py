"""Select using three forward validation windows; reserve Feb–Apr 2026 for testing."""
import hashlib
import json
import platform
from datetime import datetime,timezone
from importlib.metadata import version
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,median_absolute_error
from threadpoolctl import threadpool_limits
from config import ROOT
from population_features import FEATURES
from model_candidates import candidate,CANDIDATES

OUT=ROOT/'reports/model_comparison'
FOLDS=[('winter_2024_25','2025-01-01','2025-05-01'),
       ('opening_2025_26','2025-10-01','2025-12-01'),
       ('winter_2025_26','2025-12-01','2026-02-01')]
TEST_START='2026-02-01'

def metrics(y,pred):
    errors=np.asarray(pred)-np.asarray(y)
    return {'mae':float(mean_absolute_error(y,pred)),
            'rmse':float(np.sqrt(mean_squared_error(y,pred))),
            'r2':float(r2_score(y,pred)), 'bias':float(errors.mean()),
            'median_absolute_error':float(median_absolute_error(y,pred)),
            'within_5_points':float((np.abs(errors)<=5).mean()),'rows':len(y)}

def prediction(model,df):
    return np.maximum(0,model.predict(df[FEATURES]))

def evaluate():
    path=ROOT/'data/processed/population/features.csv'
    df=pd.read_csv(path,dtype={'GAME_ID':str,'player_key':str},parse_dates=['GAME_DATE'])
    OUT.mkdir(parents=True,exist_ok=True)
    validation=[]
    validation_predictions=[]
    for name in CANDIDATES:
        for fold,start,end in FOLDS:
            train=df[df.GAME_DATE<pd.Timestamp(start)]
            val=df[(df.GAME_DATE>=pd.Timestamp(start))&(df.GAME_DATE<pd.Timestamp(end))]
            if train.empty or val.empty or train.GAME_DATE.max()>=val.GAME_DATE.min():
                raise ValueError('Invalid chronological partition')
            model=candidate(name)
            model.fit(train[FEATURES],train.PTS)
            pred=prediction(model,val)
            row={'model':name,'fold':fold,**metrics(val.PTS,pred),'train_rows':len(train)}
            validation.append(row)
            validation_predictions.extend(zip([name]*len(val),val.PTS.to_numpy(),pred))
            print(f"{name:16} {fold:18} MAE={row['mae']:.4f} RMSE={row['rmse']:.4f}",flush=True)
    results=pd.DataFrame(validation)
    results.to_csv(OUT/'validation_folds.csv',index=False)
    # Weighted mean absolute error across all validation predictions; no test access.
    vp=pd.DataFrame(validation_predictions,columns=['model','actual','predicted'])
    vp['absolute_error']=(vp.actual-vp.predicted).abs()
    scores=vp.groupby('model').absolute_error.mean().sort_values()
    selected=str(scores.index[0])
    # Freeze selection to disk before computing any held-out scores.
    selection={'selected_model':selected,'selection_metric':'pooled chronological validation MAE',
               'validation_mae':scores.to_dict(),'test_start':TEST_START,
               'folds':[{'name':a,'start':b,'end_exclusive':c} for a,b,c in FOLDS]}
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    train=df[df.GAME_DATE<pd.Timestamp(TEST_START)]
    test=df[df.GAME_DATE>=pd.Timestamp(TEST_START)]
    holdout=test[['GAME_DATE','GAME_ID','PLAYER_ID','PLAYER_NAME','TEAM_ABBREVIATION',
                  'SEASON','season_games','PTS','MIN','prior_mpg','changed_team_since_prior']].copy()
    test_results=[]
    for name in CANDIDATES:
        model=candidate(name)
        model.fit(train[FEATURES],train.PTS)
        pred=prediction(model,test)
        holdout[name]=pred
        result={'model':name,'selected_on_validation':name==selected,**metrics(test.PTS,pred)}
        test_results.append(result)
        print(f"TEST {name:16} MAE={result['mae']:.4f} RMSE={result['rmse']:.4f} R2={result['r2']:.4f}",flush=True)
    pd.DataFrame(test_results).to_csv(OUT/'test_metrics.csv',index=False)
    holdout.to_csv(OUT/'test_predictions.csv',index=False)
    per_player=[]
    for pid,g in holdout.groupby('PLAYER_ID'):
        per_player.append({'player_id':int(pid),'player_name':g.PLAYER_NAME.iloc[0],
                           **metrics(g.PTS,g[selected])} if len(g)>1 else
                          {'player_id':int(pid),'player_name':g.PLAYER_NAME.iloc[0],'rows':len(g),
                           'mae':float(abs(g.PTS.iloc[0]-g[selected].iloc[0]))})
    pd.DataFrame(per_player).to_csv(OUT/'selected_per_player.csv',index=False)
    groups=[]
    for label,mask in [('all',np.ones(len(test),dtype=bool)),('prior_mpg_15_25',test.prior_mpg.between(15,25)),
                       ('prior_mpg_over_25',test.prior_mpg.gt(25)),('changed_team',test.changed_team_since_prior.eq(1)),
                       ('actual_minutes_under_15',test.MIN.lt(15))]:
        g=holdout.loc[mask]
        if len(g)>1:
            groups.append({'segment':label,**metrics(g.PTS,g[selected]),'baseline_mae':mean_absolute_error(g.PTS,g.rolling5)})
    pd.DataFrame(groups).to_csv(OUT/'selected_segments.csv',index=False)
    # Paired date-cluster bootstrap: preserves player correlations within game dates.
    delta=(holdout[selected]-holdout.PTS).abs()-(holdout.rolling5-holdout.PTS).abs()
    daily=pd.DataFrame({'date':holdout.GAME_DATE,'delta':delta}).groupby('date').delta.agg(['sum','count'])
    rng=np.random.default_rng(42)
    draws=rng.integers(0,len(daily),size=(2000,len(daily)))
    bootstrap=daily['sum'].to_numpy()[draws].sum(axis=1)/daily['count'].to_numpy()[draws].sum(axis=1)
    interval=np.quantile(bootstrap,[.025,.975]).tolist()
    model=candidate(selected)
    model.fit(df[FEATURES],df.PTS)
    artifact=ROOT/'models/points_population_2026-27.joblib'
    joblib.dump(model,artifact)
    metadata={**selection,'model_version':artifact.stem,'fitted_rows':len(df),
              'training_players':int(df.PLAYER_ID.nunique()),'data_through':str(df.GAME_DATE.max().date()),
              'features':FEATURES,'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'created_at':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
              'packages':{k:version(k) for k in ['scikit-learn','pandas','numpy','joblib']},
              'test_metrics':test_results,'selected_minus_baseline_mae_95pct_date_bootstrap':interval,
              'interpretation':'Negative MAE difference favors selected model. Selection frozen before test; final artifact refitted on all history.',
              'minutes_source':'Historical estimate only; no provider projection or injury feed.'}
    artifact.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Selected {selected}; saved {artifact}; paired MAE delta interval {interval}',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):
        evaluate()
