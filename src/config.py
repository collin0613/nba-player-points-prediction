"""Shared season, player and feature definitions; paths never depend on cwd."""
import argparse
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
TRAIN_SEASON = '2025-26'
PREDICT_SEASON = '2026-27'
PLAYERS = [
    {'player_name': 'anthony_edwards', 'player_id': 1630162},
    {'player_name': 'cade_cunningham', 'player_id': 1630595},
    {'player_name': 'giannis_antetokounmpo', 'player_id': 203507},
    {'player_name': 'jalen_brunson', 'player_id': 1628973},
    {'player_name': 'lebron_james', 'player_id': 2544},
    {'player_name': 'shai_gilgeous-alexander', 'player_id': 1628983},
    {'player_name': 'stephen_curry', 'player_id': 201939},
]
NUM_FEATURES = ['is_home', 'rest_days', 'pts_avg_5', 'pts_avg_10',
                'min_avg_5', 'pts_allowed_avg_10', 'pace_proxy_avg_10']
CAT_FEATURES = ['player_name']
MIN_THRESHOLD = 15

def season_value(value):
    if not re.fullmatch(r'\d{4}-\d{2}', value) or int(value[-2:]) != (int(value[:4]) + 1) % 100:
        raise argparse.ArgumentTypeError('Expected a season such as 2025-26')
    return value

def season_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--season', type=season_value, default=TRAIN_SEASON)
    return parser

def raw_dir(season):
    return ROOT / 'data' / 'raw' / season

def processed_dir(season):
    return ROOT / 'data' / 'processed' / season

def player_path(season, name):
    return raw_dir(season) / 'players' / f'{name}_gamelogs.csv'

def league_path(season):
    return raw_dir(season) / 'league' / 'gamelog.csv'

def model_path(season):
    return ROOT / 'models' / f'ridge_points_model_{season}.joblib'
