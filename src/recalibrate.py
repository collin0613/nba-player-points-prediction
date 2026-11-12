"""Reproducible five-game checkpoint experiment; never writes prediction records.

Opening 2026-27 and the previously inspected 2026 holdout are development data.
Selection uses historical forward folds. Activation is an explicit separate step.
"""
import argparse
import hashlib
import json
import shutil
import platform
from importlib.metadata import version
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from config import ROOT
from data_io import read_logs
from population_features import load_history, player_features, attach_context, FEATURES
from role_features import build_role_features, feature_columns, SCHEMA
from model_candidates import candidate, preprocessing
from evaluate_models import FOLDS, metrics

OUT = ROOT/'reports/recalibration_2026-11-09'
ORIGINAL = 'points_population_2026-27'
SPECS = {
    'unchanged': {'original':True},
    'role_mae': {'loss':'absolute_error'},
    'role_mse': {'loss':'squared_error'},
    'role_ridge': {'ridge':True},
    'role_no_shares': {'loss':'absolute_error','shares':False},
    'role_prior5': {'loss':'absolute_error','prior_strength':5.0},
}


def make_model(spec):
    if spec.get('original'):
        return candidate('boosted_mae')
    columns = feature_columns(spec.get('shares',True))
    ridge = spec.get('ridge',False)
    model = (Ridge(alpha=1000,solver='lsqr',tol=1e-5) if ridge else
             HistGradientBoostingRegressor(loss=spec['loss'],learning_rate=.05,
                 max_iter=180,max_leaf_nodes=15,min_samples_leaf=40,
                 l2_regularization=10,early_stopping=False,random_state=42))
    return Pipeline([('prep',preprocessing([c for c in columns if c!='player_key'],ridge,ridge)),('model',model)])


def audit_data():
    p=read_logs(ROOT/'data/raw/2026-27/all_player_games.csv')
    t=read_logs(ROOT/'data/raw/2026-27/league/gamelog.csv')
    s=pd.read_csv(ROOT/'data/raw/2026-27/schedule.csv',dtype={'gameId':str})
    counts=t.groupby('TEAM_ABBREVIATION').GAME_ID.nunique()
    totals=p.groupby(['GAME_ID','TEAM_ID'])[['PTS','FGA','FTA','FG3A']].sum()
    expected=t.set_index(['GAME_ID','TEAM_ID'])[totals.columns]
    mismatch=(totals-expected).abs().fillna(np.inf).gt(0).any(axis=1)
    finals=set(s.loc[s.gameStatus.eq(3),'gameId'])
    logged=set(t.GAME_ID)
    team_keys=set(map(tuple,t[['GAME_ID','TEAM_ID','TEAM_ABBREVIATION']].to_numpy()))
    invalid_teams=sum(tuple(row) not in team_keys for row in p[['GAME_ID','TEAM_ID','TEAM_ABBREVIATION']].to_numpy())
    result={'team_games':counts.to_dict(),'completed_games':len(finals),
        'missing_games':sorted(finals-logged),'unexpected_games':sorted(logged-finals),
        'duplicate_player_games':int(p.duplicated(['PLAYER_ID','GAME_ID']).sum()),
        'duplicate_team_games':int(t.duplicated(['TEAM_ID','GAME_ID']).sum()),
        'invalid_team_assignments':invalid_teams,'mismatched_box_totals':int(mismatch.sum()),
        'zero_minute_rows_excluded':int(p.MIN.le(0).sum()),
        'appearances_by_player':p[p.MIN.gt(0)].groupby('PLAYER_ID').size().to_dict(),
        'data_through':str(p.GAME_DATE.max()),
        'source_metadata':json.loads((ROOT/'data/raw/2026-27/all_player_games.meta.json').read_text())}
    result['ready']=(len(counts)==30 and counts.min()>=5 and not any([
        result[k] for k in ['missing_games','unexpected_games','duplicate_player_games',
                           'duplicate_team_games','invalid_team_assignments','mismatched_box_totals']]))
    (OUT/'readiness.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['ready']:
        raise ValueError('Checkpoint data audit failed; inspect readiness.json')
    return result


def segment_scores(frame, pred, name, period):
    segments={'all':np.ones(len(frame),dtype=bool),
        'first_5_appearances':frame.season_games.lt(5),
        'role_5_plus':frame.role_games.ge(5),
        'limited_current_role':frame.role_games.lt(5),
        'changed_team':frame.changed_team_since_prior.eq(1),
        'minutes_role_shift_5_plus':frame.role_games.ge(5)&(frame.role_min_avg-frame.prior_mpg).abs().ge(5)}
    return [{'model':name,'period':period,'segment':label,**metrics(frame.loc[mask,'PTS'],pred[mask])}
            for label,mask in segments.items() if mask.sum()>1]


def run_experiment():
    OUT.mkdir(parents=True,exist_ok=True)
    audit=audit_data()
    p,t=load_history(['2023-24','2024-25','2025-26','2026-27'])
    original=attach_context(player_features(p),t)
    role=build_role_features(p,t)
    prior5=build_role_features(p,t,prior_strength=5)
    keys=['GAME_ID','PLAYER_ID']
    eligible=original[original.eligible & original.SEASON.ne('2023-24')].sort_values(['GAME_DATE','PLAYER_ID'])
    index=pd.MultiIndex.from_frame(eligible[keys])
    frames={}
    for label,frame in [('original',original),('role',role),('prior5',prior5)]:
        frames[label]=frame.set_index(keys).loc[index].reset_index()
    frame=frames['role']
    periods=FOLDS+[('reviewed_2026','2026-02-01','2026-07-01'),
                   ('opening_2026_27','2026-10-01','2026-11-09')]
    results=[]; segments=[]; predictions=frame[['GAME_ID','PLAYER_ID','PLAYER_NAME','GAME_DATE','PTS','MIN','role_games','role_min_avg','prior_mpg']].copy()
    selection=None
    for period,start,end in periods:
        train=frame.GAME_DATE.lt(start)
        valid=frame.GAME_DATE.ge(start)&frame.GAME_DATE.lt(end)
        for name,spec in SPECS.items():
            source=frames['original' if spec.get('original') else 'prior5' if spec.get('prior_strength') else 'role']
            cols=FEATURES if spec.get('original') else feature_columns(spec.get('shares',True))
            model=make_model(spec)
            model.fit(source.loc[train,cols],source.loc[train,'PTS'])
            pred=np.maximum(0,model.predict(source.loc[valid,cols]))
            row={'model':name,'period':period,**metrics(source.loc[valid,'PTS'],pred)}
            results.append(row)
            segments.extend(segment_scores(frame.loc[valid],pred,name,period))
            predictions.loc[valid,name]=pred
            print(f'{period:20} {name:16} MAE={row["mae"]:.4f} RMSE={row["rmse"]:.4f} bias={row["bias"]:+.4f}',flush=True)
        for name,column in [('role_rolling5','pts_avg_5'),('role_blend','pts_blend'),('role_minutes_rate','rate_points_baseline')]:
            pred=frame.loc[valid,column].to_numpy()
            results.append({'model':name,'period':period,**metrics(frame.loc[valid,'PTS'],pred)})
            predictions.loc[valid,name]=pred
        if period==FOLDS[-1][0]:
            scores=pd.DataFrame(results)
            scores['total_error']=scores.mae*scores.rows
            pooled=scores.groupby('model').total_error.sum()/scores.groupby('model').rows.sum()
            selected=pooled.loc[list(SPECS)].idxmin()
            selection={'selected_model':selected,'pooled_validation_mae':pooled.to_dict(),
                       'selection_rule':'Lowest pooled MAE across three historical forward folds; baseline challengers reported separately.',
                       'frozen_at':datetime.now(timezone.utc).isoformat()}
            (OUT/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    pd.DataFrame(results).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(segments).to_csv(OUT/'segments.csv',index=False)
    predictions.to_csv(OUT/'development_predictions.csv',index=False)
    # Train candidates only through the preceding season for paired replay of live
    # forecasts (their feature cutoffs may precede the actual game by several days).
    for name,spec in SPECS.items():
        source=frames['original' if spec.get('original') else 'prior5' if spec.get('prior_strength') else 'role']
        cols=FEATURES if spec.get('original') else feature_columns(spec.get('shares',True))
        fit=source.SEASON.ne('2026-27')
        model=make_model(spec);model.fit(source.loc[fit,cols],source.loc[fit,'PTS'])
        joblib.dump(model,OUT/f'{name}.joblib')
    # Persist new features for reproducibility; do not replace original training data.
    frame.to_csv(OUT/'features.csv',index=False)
    print(json.dumps(selection,indent=2),flush=True)


def activate():
    """Publish the reviewed candidate under a new, immutable version."""
    selection=json.loads((OUT/'selection.json').read_text())
    paired=json.loads((OUT/'paired_summary.json').read_text())
    selected=selection['selected_model']
    if selected=='unchanged' or paired['selected_model']!=selected:
        raise ValueError('No validated revised candidate to activate')
    scores={r['model']:r for r in paired['metrics']}
    if scores[selected]['mae']>=scores['original_stored']['mae']:
        raise ValueError('Selected candidate did not improve original-cutoff replay MAE')
    spec=SPECS[selected]
    artifact=ROOT/'models/points_population_2026-27_role_v2_20261109.joblib'
    if artifact.exists() or artifact.with_suffix('.json').exists():
        raise FileExistsError('Version already exists; never overwrite a deployed artifact')
    frame=pd.read_csv(OUT/'features.csv',parse_dates=['GAME_DATE'])
    fit=frame[frame.SEASON.ne('2026-27')]
    now=datetime.now(timezone.utc).isoformat()
    metadata={**selection,'model_version':artifact.stem,'feature_schema_version':SCHEMA,
        'feature_config':{'prior_strength':spec.get('prior_strength',2.5),'shares':spec.get('shares',True)},
        'features':feature_columns(spec.get('shares',True)), 'fitted_rows':len(fit),
        'training_players':int(fit.PLAYER_ID.nunique()), 'data_through':str(fit.GAME_DATE.max().date()),
        'created_at':now,'deployed_at':now,'python':platform.python_version(),
        'packages':{k:version(k) for k in ['scikit-learn','pandas','numpy','joblib']},
        'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'src/role_features.py',ROOT/'src/recalibrate.py',OUT/'features.csv',OUT/f'{selected}.joblib']},
        'evaluation_report':str(OUT.relative_to(ROOT)),'paired_development_replay':paired,
        'feature_definitions':'Current season and uninterrupted team stint; prior-only shifts; 5 prior-equivalent appearances; current-team rolling attempts/rates and team shares; 3-game team-context shrinkage.',
        'minutes_source':'Historical conditional-on-playing estimate; no injury or lineup feed.',
        'interpretation':'Trained through previous season. Historical folds selected candidate; opening-season replay is development evidence. Forward games after deployment are needed to validate.'}
    shutil.copyfile(OUT/f'{selected}.joblib',artifact)
    artifact.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    pointer=ROOT/'models/active_population_2026-27.json'
    temporary=pointer.with_suffix('.tmp')
    temporary.write_text(json.dumps({'model_version':artifact.stem,'deployed_at':now,
                                    'previous_model_version':ORIGINAL},indent=2)+'\n')
    temporary.replace(pointer)
    print(f'Activated {artifact.stem}; original artifact preserved',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activate',action='store_true',help='Activate reviewed outputs without rerunning experiments')
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        activate() if args.activate else run_experiment()
