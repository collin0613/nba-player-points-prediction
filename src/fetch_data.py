# src/fetch_data.py
# Call nba_api, fetch player game logs, save them unchanged.

from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import pandas as pd
import time

def get_player_id(player_name: str) -> int:
    """
    Look up a player's NBA ID by full name.
    """
    nba_players = players.find_players_by_full_name(player_name)
    if not nba_players:
        raise ValueError(f"No player found with name: {player_name}")
    return nba_players[0]["id"]


def fetch_player_game_logs(player_id: int, season: str) -> pd.DataFrame:
    """
    Fetch all game logs for a player in a given NBA season.
    """
    gamelog = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season
    )
    df = gamelog.get_data_frames()[0]
    return df


def main():
    player_name = "Anthony Edwards" # update for different players
    safe_name = player_name.lower().replace(" ", "_")
    season = "2024-25"

    print(f"Fetching game logs for {player_name} ({season})")

    player_id = get_player_id(player_name)

    # Be polite to the NBA servers
    time.sleep(1)

    df = fetch_player_game_logs(player_id, season)
    output_path = f"data/raw/player/{safe_name}_gamelogs.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
