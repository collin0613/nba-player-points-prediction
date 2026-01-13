# src/train_model.py
# Trains regression model to predict player points using processed features.
#   Prints the results of model vs baseline for testing model accuracy, 
#   allowing for different sets of metrics to be tested to find the strongest performing combination.

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEASON_TAG = "2024-25" # update in the future; this was the last full NBA season at time of development
DATA_PATH = f"data/processed/all_players_features_{SEASON_TAG}.csv"
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/ridge_points_model_{SEASON_TAG}.joblib"

# Numeric features
NUM_FEATURES = [
    "is_home",
    "rest_days",
    "pts_avg_5",
    "pts_avg_10",
    "min_avg_5",
    "pts_allowed_avg_10",
    "pace_proxy_avg_10",
    "star_pts_allowed_avg_10",
]

# Removed features which did not help reduce mean absolute error in model performance:
    #"min_avg_10",
    #"fga_avg_5",
    #"fta_avg_5",
    #"usage_avg_5",

CAT_FEATURES = ["player_name"]
TARGET = "label_pts" # target variable to predict

def chronological_split(df: pd.DataFrame, date_col: str = "GAME_DATE", train_frac: float = 0.8):
    """
    Splits by time: earlier dates are train, later dates are test.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    cutoff_idx = int(len(df) * train_frac)
    cutoff_date = df.loc[cutoff_idx, date_col]

    train_df = df[df[date_col] <= cutoff_date].copy()
    test_df = df[df[date_col] > cutoff_date].copy()

    # If cutoff_date causes a very small test set (rare), fallback to index split
    if len(test_df) < 20:
        train_df = df.iloc[:cutoff_idx].copy()
        test_df = df.iloc[cutoff_idx:].copy()

    return train_df, test_df


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Run: python src/build_dataset.py"
        )

    df = pd.read_csv(DATA_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    needed = NUM_FEATURES + CAT_FEATURES + [TARGET, "GAME_DATE"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing columns: {missing}\n"
            f"Loaded from: {DATA_PATH}\n"
            f"Columns present: {list(df.columns)}\n"
            f"Next steps: re-run process_data.py then build_dataset.py."
        )

    df = df.dropna(subset=NUM_FEATURES + CAT_FEATURES + [TARGET]).reset_index(drop=True)

    train_df, test_df = chronological_split(df, train_frac=0.8)

    X_train = train_df[NUM_FEATURES + CAT_FEATURES]
    y_train = train_df[TARGET].astype(float)

    X_test = test_df[NUM_FEATURES + CAT_FEATURES]
    y_test = test_df[TARGET].astype(float)

    # Baseline prediction: just use pts_avg_5, average points scored over last 5 games
    # Goal is to beat the mean absolute error of this simple baseline
    baseline_pred = X_test["pts_avg_5"].to_numpy()
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    # Preprocess:
    # - scale numeric features (helps Ridge)
    # - one-hot encode player_name
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ]
    )

    # Ridge regression model used
    model = Ridge(alpha=1.0, random_state=42)
    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("=== Results ===")
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    print(f"Baseline MAE (predict pts_avg_5): {baseline_mae:.3f}")
    print(f"Model MAE (Ridge):               {mae:.3f}")
    print(f"Model RMSE:                      {rmse:.3f}")
    print(f"Model R^2:                       {r2:.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")

if __name__ == "__main__":
    main()
