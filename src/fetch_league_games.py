import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

SEASON = "2024-25"
OUT = f"data/raw/league/gamelog_{SEASON}.csv"

def main():
    print(f"Fetching league game log for {SEASON}...")
    time.sleep(1)

    lg = leaguegamelog.LeagueGameLog(season=SEASON, season_type_all_star="Regular Season")
    df = lg.get_data_frames()[0]
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows -> {OUT}")

if __name__ == "__main__":
    main()
