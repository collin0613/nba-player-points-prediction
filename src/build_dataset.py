import os
import glob
import pandas as pd

SEASON_TAG = "2024-25" # update in the future; this was the last full NBA season at time of development
PROCESSED_DIR = "data/processed"
OUTPUT_PATH = f"{PROCESSED_DIR}/all_players_features_{SEASON_TAG}.csv"

def filename_to_player_name(path: str) -> str:
    """
    Example filename:
      data/processed/jalen_brunson_features_2024-25.csv
    -> player_name = "jalen_brunson"
    """
    base = os.path.basename(path)
    # remove suffix: _features_2024-25.csv
    suffix = f"_features_{SEASON_TAG}.csv"
    if not base.endswith(suffix):
        return base.replace(".csv", "")
    return base[: -len(suffix)]


def main():
    pattern = f"{PROCESSED_DIR}/*_features_{SEASON_TAG}.csv"
    paths = sorted(glob.glob(pattern))

    # Exclude the combined file if it exists
    paths = [p for p in paths if os.path.basename(p) != os.path.basename(OUTPUT_PATH)]

    if not paths:
        raise FileNotFoundError(
            f"No processed feature files found. Expected something like: {pattern}"
        )

    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df["player_name"] = filename_to_player_name(path)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # Sort for chronological splitting
    all_df = all_df.sort_values(["GAME_DATE", "player_name"]).reset_index(drop=True)
    required = {
        "GAME_DATE", "is_home", "rest_days",
        "pts_avg_5", "pts_avg_10", "min_avg_5", "fga_avg_5", "fta_avg_5",
        "label_pts", "player_name"
    }
    missing = required - set(all_df.columns)
    if missing:
        raise ValueError(f"Combined dataset is missing columns: {missing}")

    all_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(all_df)} rows -> {OUTPUT_PATH}")
    print(f"Players included: {sorted(all_df['player_name'].unique().tolist())}")


if __name__ == "__main__":
    main()
