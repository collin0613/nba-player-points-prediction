"""Download all tracked players; reruns reuse validated season-specific files."""
from config import PLAYERS, season_parser, player_path
from data_io import fetch_player

def main():
    parser = season_parser(__doc__)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    for player in PLAYERS:
        df = fetch_player(player, args.season, refresh=args.refresh)
        print(f"{player['player_name']}: {len(df)} rows -> {player_path(args.season, player['player_name'])}", flush=True)

if __name__ == '__main__':
    main()
