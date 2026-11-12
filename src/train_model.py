"""Evaluate the existing Ridge design chronologically, then refit on all history."""
import hashlib
import platform
from importlib.metadata import version
import json
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import NUM_FEATURES, CAT_FEATURES, season_parser, processed_dir, model_path

def chronological_split(df, date_col='GAME_DATE', train_frac=0.8):
    df = df.sort_values(date_col).reset_index(drop=True)
    cutoff = df.loc[int(len(df) * train_frac), date_col]
    train, test = df[df[date_col] <= cutoff].copy(), df[df[date_col] > cutoff].copy()
    if len(train) < 20 or len(test) < 20:
        raise ValueError('Insufficient rows for a date-separated evaluation (minimum 20 in each partition)')
    return train, test

def make_pipeline():
    prep = ColumnTransformer([
        ('num', StandardScaler(), NUM_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_FEATURES),
    ])
    return Pipeline([('prep', prep), ('model', Ridge(alpha=1.0, random_state=42))])

def train(season):
    path = processed_dir(season) / 'all_players_features.csv'
    df = pd.read_csv(path, parse_dates=['GAME_DATE'])
    features = NUM_FEATURES + CAT_FEATURES
    if df[features + ['label_pts']].isna().any().any():
        raise ValueError('Missing model inputs')
    train_df, test_df = chronological_split(df)
    pipeline = make_pipeline()
    pipeline.fit(train_df[features], train_df.label_pts)
    predictions = pipeline.predict(test_df[features])
    metrics = {
        'baseline_mae': mean_absolute_error(test_df.label_pts, test_df.pts_avg_5),
        'ridge_mae': mean_absolute_error(test_df.label_pts, predictions),
        'ridge_rmse': float(np.sqrt(mean_squared_error(test_df.label_pts, predictions))),
        'ridge_r2': r2_score(test_df.label_pts, predictions),
        'train_rows': len(train_df), 'test_rows': len(test_df),
        'train_through': str(train_df.GAME_DATE.max().date()),
        'test_from': str(test_df.GAME_DATE.min().date()),
    }
    # Preserve honest holdout metrics before fitting the deployable model on all data.
    evaluated = test_df[['GAME_DATE', 'player_name', 'label_pts', 'pts_avg_5']].copy()
    evaluated['ridge_predicted_pts'] = predictions
    evaluated.to_csv(processed_dir(season) / 'holdout_predictions.csv', index=False)
    pipeline.fit(df[features], df.label_pts)
    target = model_path(season)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, target)
    metadata = {
        'season': season, 'model_version': target.stem, 'features': features,
        'python_version': platform.python_version(),
        'packages': {name: version(name) for name in ['scikit-learn', 'pandas', 'numpy', 'joblib', 'nba_api']},
        'fitted_rows': len(df), 'data_through': str(df.GAME_DATE.max().date()),
        'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'created_at': datetime.now(timezone.utc).isoformat(),
        'evaluation': metrics,
        'note': 'Holdout metrics precede final refit on all rows. Not an opening-night backtest.',
    }
    target.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))
    if metrics['ridge_mae'] >= metrics['baseline_mae']:
        print('NOTE: Ridge did not beat the rolling-average baseline on this holdout.')
    return metadata

def main():
    train(season_parser(__doc__).parse_args().season)

if __name__ == '__main__':
    main()
