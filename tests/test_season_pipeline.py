"""Temporal and season-transition regressions. No network or database required."""
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import ROOT, PLAYERS, NUM_FEATURES, player_path
from features import historical_team_context, recent_player_features, recent_team_features
from data_io import read_logs, validate_logs
from predict import select_next_game, combine_history, create_predictions
from train_model import chronological_split

class SeasonTests(unittest.TestCase):
    def test_team_context_is_pregame_and_grouped(self):
        raw = read_logs(ROOT / 'tests/fixtures/nba_team_gamelogs.csv')
        ctx = historical_team_context(raw)
        # Each team's first ten games must have no pregame ten-game context.
        for _, team in ctx.groupby('TEAM_ABBREVIATION'):
            self.assertTrue(team.iloc[:10].pts_allowed_avg_10.isna().all())
        first = raw.sort_values('GAME_DATE').iloc[0]
        original = ctx[(ctx.TEAM_ABBREVIATION == first.TEAM_ABBREVIATION)
                       & (ctx.GAME_DATE == pd.Timestamp(first.GAME_DATE))]
        changed = raw.copy()
        changed.loc[changed.GAME_ID == first.GAME_ID, 'PTS'] = 999
        rebuilt = historical_team_context(changed)
        same_game = rebuilt[(rebuilt.TEAM_ABBREVIATION == first.TEAM_ABBREVIATION)
                            & (rebuilt.GAME_DATE == pd.Timestamp(first.GAME_DATE))]
        pd.testing.assert_frame_equal(original, same_game)

    def test_training_and_live_team_values_agree_for_next_game(self):
        raw = read_logs(ROOT / 'tests/fixtures/nba_team_gamelogs.csv')
        ctx = historical_team_context(raw).dropna()
        self.assertGreater(len(ctx), 0)
        for row in ctx.itertuples():
            live = recent_team_features(raw, row.TEAM_ABBREVIATION, row.GAME_DATE)
            self.assertAlmostEqual(live['pts_allowed_avg_10'], row.pts_allowed_avg_10)
            self.assertAlmostEqual(live['pace_proxy_avg_10'], row.pace_proxy_avg_10)

    def test_opening_night_history_and_future_exclusion(self):
        dates = pd.date_range('2026-03-01', periods=10)
        prior = pd.DataFrame({'GAME_DATE': dates, 'PTS': range(10), 'MIN': [30]*10,
                              'MATCHUP': ['OLD vs. BOS']*10, 'Game_ID': [str(i) for i in range(10)],
                              'Player_ID': [1]*10})
        current = pd.DataFrame({'GAME_DATE': [pd.Timestamp('2026-10-20'), pd.Timestamp('2026-10-21')],
                                'PTS': [20, 999], 'MIN': [30, 30], 'MATCHUP': ['NEW vs. BOS']*2,
                                'Game_ID': ['10', '11'], 'Player_ID': [1, 1]})
        history = combine_history(prior, current)
        opening = recent_player_features(history, '2026-10-20')
        self.assertEqual(opening['pts_avg_10'], 4.5)
        next_day = recent_player_features(history, '2026-10-21')
        self.assertEqual(next_day['pts_avg_10'], 6.5)
        self.assertEqual(next_day['last_game_date'], pd.Timestamp('2026-10-20'))

    def test_schedule_rejects_started_final_and_other_team(self):
        now = datetime(2026, 10, 20, 20, tzinfo=timezone.utc)
        schedule = pd.DataFrame({
            'gameId': ['past', 'live', 'final', 'other', 'correct'],
            'homeTeam_teamId': [1, 1, 1, 3, 1], 'awayTeam_teamId': [2]*5,
            'gameStatus': [1, 2, 3, 1, 1],
            'gameDateTimeUTC': ['2026-10-20T19:00Z'] + ['2026-10-20T21:00Z']*4,
        })
        self.assertEqual(select_next_game(schedule, 1, now).gameId, 'correct')

    def test_split_keeps_dates_together(self):
        df = pd.DataFrame({'GAME_DATE': np.repeat(pd.date_range('2025-01-01', periods=30), 7)})
        train, test = chronological_split(df)
        self.assertLess(train.GAME_DATE.max(), test.GAME_DATE.min())
        self.assertFalse(set(train.GAME_DATE) & set(test.GAME_DATE))

    def test_wrong_season_and_missing_team_pair_rejected(self):
        raw = read_logs(ROOT / 'tests/fixtures/nba_team_gamelogs.csv')
        with self.assertRaises(ValueError):
            validate_logs(raw, '2025-26', team=True)
        with self.assertRaises(ValueError):
            validate_logs(raw.iloc[1:], '2024-25', team=True)

    def test_feature_schema_removes_invalid_star_defense(self):
        self.assertNotIn('star_pts_allowed_avg_10', NUM_FEATURES)
        self.assertEqual(len(NUM_FEATURES), len(set(NUM_FEATURES)))

    def test_season_paths_are_distinct_from_legacy(self):
        self.assertNotEqual(player_path('2025-26', 'anthony_edwards'),
                            player_path('2024-25', 'anthony_edwards'))
        self.assertNotEqual(player_path('2025-26', 'anthony_edwards'),
                            ROOT / 'data/raw/player/anthony_edwards_gamelogs.csv')

if __name__ == '__main__':
    unittest.main()
