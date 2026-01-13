import os
import glob
import pandas as pd

SEASON_TAG = "2024-25"
RAW_DIR = "data/raw/player"
OUT_DIR = "data/processed"
MIN_THRESHOLD = 15 # min minutes played to include row
TEAM_CONTEXT_PATH = f"{OUT_DIR}/team_context_{SEASON_TAG}.csv"


def extract_player_name(raw_path: str) -> str:
    """
    data/raw/jalen_brunson_gamelogs.csv -> jalen_brunson
    """
    base = os.path.basename(raw_path)
    return base.replace("_gamelogs.csv", "")


def load_team_context() -> pd.DataFrame:
    """
    team_context table with columns:
      TEAM_ABBREVIATION, GAME_DATE, pts_allowed_avg_10, pace_proxy_avg_10, star_pts_allowed_avg_10
    """
    if not os.path.exists(TEAM_CONTEXT_PATH):
        raise FileNotFoundError(
            f"Missing {TEAM_CONTEXT_PATH}. Build it first:\n"
            f"  python src/fetch_league_games.py\n"
            f"  python src/build_opponent_defense.py"
        )

    def_df = pd.read_csv(TEAM_CONTEXT_PATH)
    def_df["GAME_DATE"] = pd.to_datetime(def_df["GAME_DATE"])
    return def_df


def process_one_player(raw_path: str, def_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    # Date + sort
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    # Filter weird rows
    df = df.dropna(subset=["PTS", "MIN", "MATCHUP"])
    df = df[df["MIN"] >= MIN_THRESHOLD].copy()

    # Label
    df["label_pts"] = df["PTS"]

    # Matchup parsing
    df["is_home"] = df["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)
    df["opponent"] = df["MATCHUP"].apply(lambda x: x.split(" ")[-1])

    # Rest
    df["rest_days"] = df["GAME_DATE"].diff().dt.days.fillna(3).clip(upper=5)

    # Usage proxy (simple)
    df["usage_proxy"] = df["FGA"] + 0.44 * df["FTA"] + df["TOV"]

    # Rolling features (leakage-safe)
    df["pts_avg_5"] = df["PTS"].shift(1).rolling(5).mean()
    df["pts_avg_10"] = df["PTS"].shift(1).rolling(10).mean()

    df["min_avg_5"] = df["MIN"].shift(1).rolling(5).mean()
    df["min_avg_10"] = df["MIN"].shift(1).rolling(10).mean()

    df["fga_avg_5"] = df["FGA"].shift(1).rolling(5).mean()
    df["fta_avg_5"] = df["FTA"].shift(1).rolling(5).mean()

    df["usage_avg_5"] = df["usage_proxy"].shift(1).rolling(5).mean()

    # Merge opponent defense
    context_df = pd.read_csv("data/processed/team_context_2024-25.csv")
    context_df["GAME_DATE"] = pd.to_datetime(context_df["GAME_DATE"])

    df = df.merge(
        context_df,
        left_on=["opponent", "GAME_DATE"],
        right_on=["TEAM_ABBREVIATION", "GAME_DATE"],
        how="left"
    ).drop(columns=["TEAM_ABBREVIATION"])

    # Drop rows missing rolling features or defense
    df = df.dropna(subset=[
        "pts_avg_5", "pts_avg_10",
        "min_avg_5", "min_avg_10",
        "fga_avg_5", "fta_avg_5",
        "usage_avg_5",
        "pts_allowed_avg_10",
        "label_pts",
    ]).reset_index(drop=True)

    # Keep identifiers for debugging / analysis
    debug_cols = ["GAME_DATE", "MATCHUP", "opponent", "Player_ID", "Game_ID"]

    feature_cols = [
        "is_home",
        "rest_days",
        "pts_avg_5",
        "pts_avg_10",
        "min_avg_5",
        "min_avg_10",
        "fga_avg_5",
        "fta_avg_5",
        "usage_avg_5",
        "pts_allowed_avg_10",
        "pts_allowed_avg_10",
        "pace_proxy_avg_10",
        "star_pts_allowed_avg_10",
    ]

    final_cols = debug_cols + feature_cols + ["label_pts"]
    return df[final_cols]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load team context once
    def_df = load_team_context()

    raw_paths = sorted(glob.glob(f"{RAW_DIR}/*_gamelogs.csv"))
    if not raw_paths:
        raise FileNotFoundError(f"No raw files found matching: {RAW_DIR}/*_gamelogs.csv")

    for raw_path in raw_paths:
        player_name = extract_player_name(raw_path)
        out_path = f"{OUT_DIR}/{player_name}_features_{SEASON_TAG}.csv"

        df_model = process_one_player(raw_path, def_df)
        df_model.to_csv(out_path, index=False)

        print(f"[OK] {player_name}: {len(df_model)} rows -> {out_path}")


if __name__ == "__main__":
    main()
