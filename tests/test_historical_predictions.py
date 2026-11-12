import sys
import json
import tempfile
import unittest
from datetime import datetime, timezone, date
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from predict_population import select_date_game, truncate_history, historical_watchlist, parse_predict_date
from population_features import player_features, attach_context, FEATURES


class HistoricalFeatureTests(unittest.TestCase):
    def schedule(self):
        return pd.DataFrame([dict(gameId='0022600001',homeTeam_teamId=1,awayTeam_teamId=2,
            homeTeam_teamTricode='AAA',awayTeam_teamTricode='BBB',gameStatus=3,
            gameDateTimeUTC='2026-10-25T01:00:00Z'),
            dict(gameId='0022600002',homeTeam_teamId=1,awayTeam_teamId=2,
            homeTeam_teamTricode='AAA',awayTeam_teamTricode='BBB',gameStatus=3,
            gameDateTimeUTC='2026-10-26T01:00:00Z')])

    def test_exact_local_date_not_latest_or_utc_date(self):
        now=datetime(2026,10,27,tzinfo=timezone.utc)
        self.assertEqual(select_date_game(self.schedule(),1,date(2026,10,24),now).gameId,'0022600001')
        self.assertEqual(select_date_game(self.schedule(),1,date(2026,10,25),now).gameId,'0022600002')
        self.assertIsNone(select_date_game(self.schedule(),1,date(2026,10,23),now))
        self.assertEqual(parse_predict_date('2026-10-24'),date(2026,10,24))
        with self.assertRaises(Exception):parse_predict_date('20261024')

    def test_all_inputs_and_predictions_independent_of_target_and_future_stats(self):
        history=pd.DataFrame([dict(PLAYER_ID=1,PLAYER_NAME='One',TEAM_ID=1,TEAM_ABBREVIATION='AAA',
            SEASON='2026-27',GAME_ID=str(day),GAME_DATE=f'2026-10-{day}',MATCHUP='AAA vs. BBB',
            PTS=float(day),MIN=30.,FGA=15.,FTA=4.,FG3A=5.) for day in range(20,27)])
        teams=pd.DataFrame([dict(GAME_ID=str(day),GAME_DATE=f'2026-10-{day}',TEAM_ABBREVIATION=team,
            PTS=100.,FGA=80.,FTA=20.,OREB=10.,TOV=12.,MIN=240.) for day in range(20,27) for team in ['AAA','BBB']])
        query=history.iloc[[4]].copy();query[['PTS','MIN','FGA','FTA','FG3A']]=np.nan
        def inputs(h,t,day='2026-10-24'):
            h,t=truncate_history(h,t,pd.Timestamp(day))
            frame=attach_context(player_features(pd.concat([h,query])),t)
            return frame.iloc[[-1]][FEATURES]
        before=inputs(history,teams)
        changed=history.copy();changed.loc[4:,['PTS','MIN','FGA','FTA','FG3A']]=999.
        changed_teams=teams.copy();changed_teams.loc[changed_teams.GAME_DATE.ge('2026-10-24'),['PTS','FGA']]=999.
        pd.testing.assert_frame_equal(before,inputs(changed,changed_teams))
        # Sunday includes Saturday, but excludes Sunday and Monday.
        h,t=truncate_history(history,teams,pd.Timestamp('2026-10-25'))
        self.assertEqual(h.GAME_DATE.max(),pd.Timestamp('2026-10-24'))
        self.assertEqual(h.PTS.iloc[-1],24.)

    def test_roster_uses_complete_predate_archive_not_today_or_participants(self):
        with tempfile.TemporaryDirectory() as tmp, patch('predict_population.ROOT',Path(tmp)):
            folder=Path(tmp)/'data/predictions/2026-27';folder.mkdir(parents=True)
            for name,day,team,skips in [('old',23,'AAA',[]),('new',26,'BBB',[]),('partial',23,'BBB',['x'])]:
                pd.DataFrame([dict(player_id=1,player_name='One',team=team,
                    predicted_at_utc=f'2026-10-{day}T15:00:00Z')]).to_csv(folder/f'{name}.csv',index=False)
                (folder/f'{name}.json').write_text(json.dumps(dict(rows=1,skipped=skips,
                    roster_metadata={'fetched_at':f'2026-10-{day}T14:00:00Z'})))
            watch,provenance=historical_watchlist('2026-27',pd.Timestamp('2026-10-24'),self.schedule())
            self.assertEqual(watch.team.iloc[0],'AAA')
            self.assertIn('old.json',provenance['path'])
