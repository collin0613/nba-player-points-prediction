"""Combine exactly the configured tracked players, failing on missing files."""
import pandas as pd
from config import PLAYERS, NUM_FEATURES, season_parser, processed_dir
from data_io import read_logs, save_frame

def main():
    args = season_parser(__doc__).parse_args()
    directory = processed_dir(args.season)
    frames = []
    for player in PLAYERS:
        df = read_logs(directory / f"{player['player_name']}_features.csv")
        df['player_name'] = player['player_name']
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df['GAME_DATE'] = pd.to_datetime(df.GAME_DATE)
    df = df.sort_values(['GAME_DATE', 'player_name']).reset_index(drop=True)
    if df.empty or df[NUM_FEATURES + ['label_pts']].isna().any().any():
        raise ValueError('Empty dataset or missing model features')
    if df.duplicated(['Player_ID', 'Game_ID']).any():
        raise ValueError('Duplicate player-game examples')
    output = directory / 'all_players_features.csv'
    save_frame(df, output)
    print(f'Saved {len(df)} rows, {df.player_name.nunique()} players -> {output}')

if __name__ == '__main__':
    main()
