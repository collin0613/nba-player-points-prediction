"""Download and validate completed regular-season team game logs."""
from config import season_parser, league_path
from data_io import fetch_league

def main():
    parser = season_parser(__doc__)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    df = fetch_league(args.season, refresh=args.refresh)
    print(f'Saved {len(df)} team rows -> {league_path(args.season)}')

if __name__ == '__main__':
    main()
