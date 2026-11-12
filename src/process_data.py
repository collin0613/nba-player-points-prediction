"""Build historical player features without rewriting historical team membership."""
import pandas as pd
from config import PLAYERS, NUM_FEATURES, season_parser, player_path, processed_dir
from data_io import read_logs, save_frame, validate_logs
from features import clean_player

def process_one_player(raw_path, context):
    df = clean_player(read_logs(raw_path))
    df['label_pts'] = df.PTS
    df['is_home'] = df.MATCHUP.str.contains('vs.', regex=False).astype(int)
    df['opponent'] = df.MATCHUP.str.split().str[-1]
    df['rest_days'] = df.GAME_DATE.diff().dt.days.fillna(3).clip(upper=5)
    df['usage_proxy'] = df.FGA + 0.44 * df.FTA + df.TOV
    for source, target, window in [
        ('PTS', 'pts_avg_5', 5), ('PTS', 'pts_avg_10', 10),
        ('MIN', 'min_avg_5', 5), ('MIN', 'min_avg_10', 10),
        ('FGA', 'fga_avg_5', 5), ('FTA', 'fta_avg_5', 5), ('usage_proxy', 'usage_avg_5', 5),
    ]:
        df[target] = df[source].shift(1).rolling(window).mean()
    df = df.merge(context, left_on=['opponent', 'GAME_DATE'],
                  right_on=['TEAM_ABBREVIATION', 'GAME_DATE'], how='left', validate='many_to_one')
    extra = ['min_avg_10', 'fga_avg_5', 'fta_avg_5', 'usage_avg_5']
    columns = ['GAME_DATE', 'MATCHUP', 'opponent', 'Player_ID', 'Game_ID'] + NUM_FEATURES + extra + ['label_pts']
    return df.dropna(subset=NUM_FEATURES + ['label_pts'])[columns].reset_index(drop=True)

def main():
    args = season_parser(__doc__).parse_args()
    directory = processed_dir(args.season)
    context = pd.read_csv(directory / 'team_context.csv', parse_dates=['GAME_DATE'])
    for player in PLAYERS:
        path = player_path(args.season, player['player_name'])
        validate_logs(read_logs(path), args.season)
        df = process_one_player(path, context)
        if df.empty:
            raise ValueError(f"No training rows for {player['player_name']}")
        output = directory / f"{player['player_name']}_features.csv"
        save_frame(df, output)
        print(f'Saved {len(df)} rows -> {output}')

if __name__ == '__main__':
    main()
