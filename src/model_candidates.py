"""Fixed candidate set, selected using chronological validation only."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from population_features import NUMERIC, LEGACY_NUMERIC

class FeatureBaseline(RegressorMixin, BaseEstimator):
    def __init__(self, feature='pts_avg_5'):
        self.feature=feature
    def fit(self,X,y):
        return self
    def predict(self,X):
        return np.maximum(0, X[self.feature].fillna(X.pts_blend).to_numpy(dtype=float))

def preprocessing(columns, identity=False, scale=False):
    steps=[('impute',SimpleImputer(strategy='median',keep_empty_features=True))]
    if scale:
        steps.append(('scale',StandardScaler()))
    transformers=[('numeric',Pipeline(steps),columns)]
    if identity:
        transformers.append(('player',OneHotEncoder(handle_unknown='ignore'),['player_key']))
    return ColumnTransformer(transformers)

def candidate(name):
    baseline={'rolling5':'pts_avg_5','season_blend':'pts_blend','minutes_rate':'rate_points_baseline'}
    if name in baseline:
        return FeatureBaseline(baseline[name])
    if name.startswith('ridge'):
        old=name=='ridge_original'
        alpha=1 if old else float(name.split('_')[-1])
        return Pipeline([('prep',preprocessing(LEGACY_NUMERIC if old else NUMERIC,True,True)),
                         ('model',Ridge(alpha=alpha,solver='lsqr',tol=1e-5))])
    if name=='elastic_net':
        return Pipeline([('prep',preprocessing(NUMERIC,True,True)),
                         ('model',ElasticNet(alpha=.03,l1_ratio=.2,max_iter=5000,random_state=42))])
    if name=='random_forest':
        model=RandomForestRegressor(n_estimators=120,max_depth=14,min_samples_leaf=20,
                                     max_features=.8,n_jobs=2,random_state=42)
    else:
        model=HistGradientBoostingRegressor(loss='absolute_error' if name=='boosted_mae' else 'squared_error',
            learning_rate=.05,max_iter=180,max_leaf_nodes=15,min_samples_leaf=40,
            l2_regularization=10,early_stopping=False,random_state=42)
    return Pipeline([('prep',preprocessing(NUMERIC)),('model',model)])

CANDIDATES=['rolling5','season_blend','minutes_rate','ridge_original','ridge_1','ridge_100',
            'ridge_1000','elastic_net','random_forest','boosted_mse','boosted_mae']
