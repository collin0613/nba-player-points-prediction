"""Build strictly pregame team features; the invalid star-defense feature is removed."""
from config import season_parser, league_path, processed_dir
from data_io import read_logs, save_frame, validate_logs
from features import historical_team_context

def main():
    args = season_parser(__doc__).parse_args()
    raw = read_logs(league_path(args.season))
    validate_logs(raw, args.season, team=True)
    df = historical_team_context(raw)
    path = processed_dir(args.season) / 'team_context.csv'
    save_frame(df, path)
    print(f'Saved {len(df)} rows -> {path}')

if __name__ == '__main__':
    main()
