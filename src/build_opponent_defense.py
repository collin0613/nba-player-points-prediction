# src/build_opponent_defense.py
# Build opponent defense context features from raw game logs.

import pandas as pd

SEASON = "2024-25"
INP = f"data/raw/league/gamelog_{SEASON}.csv"
OUT = f"data/processed/team_context_{SEASON}.csv"

def main():
    df = pd.read_csv(INP)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    # Each GAME_ID has 2 rows (one per team)
    total_pts = df.groupby("GAME_ID")["PTS"].transform("sum")
    df["PTS_ALLOWED"] = total_pts - df["PTS"]

    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
    df["pts_allowed_avg_10"] = (
        df.groupby("TEAM_ABBREVIATION")["PTS_ALLOWED"]
          .shift(1)
          .rolling(10)
          .mean()
    )

    # Total points in each game (pace ≈ possessions)
    game_totals = (
        df.groupby("GAME_ID")["PTS"]
          .sum()
          .reset_index(name="game_total_pts")
    )

    df = df.merge(game_totals, on="GAME_ID", how="left")

    df["pace_proxy_avg_10"] = (
        df.groupby("TEAM_ABBREVIATION")["game_total_pts"]
          .shift(1)
          .rolling(10)
          .mean()
    )

    # "Star scorer" proxy: high-usage, high-minutes games
    star_df = df[
        (df["FGA"] >= 15) &
        (df["MIN"] >= 30)
    ].copy()

    star_df = star_df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)

    star_df["star_pts_allowed_avg_10"] = (
        star_df.groupby("TEAM_ABBREVIATION")["PTS"]
               .shift(1)
               .rolling(10)
               .mean()
    )

    star_defense = star_df[
        ["TEAM_ABBREVIATION", "GAME_DATE", "star_pts_allowed_avg_10"]
    ].dropna()

    context = df[
        ["TEAM_ABBREVIATION", "GAME_DATE", "pts_allowed_avg_10", "pace_proxy_avg_10"]
    ].dropna()

    # Merge star defense onto context table
    context = context.merge(
        star_defense,
        on=["TEAM_ABBREVIATION", "GAME_DATE"],
        how="left"
    )

    context.to_csv(OUT, index=False)
    print(f"Saved {len(context)} rows -> {OUT}")

if __name__ == "__main__":
    main()
