import sys
import unittest
import tempfile
import json
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from role_features import build_role_features, feature_columns
from predict_population import resolve_model
from population_features import FEATURES


class RoleFeaturesTests(unittest.TestCase):
    def fixture(self):
        players=[]; teams=[]
        for season,year in [('2025-26',2025),('2026-27',2026)]:
            for i in range(9):
                gid=f'002{year}{i:03}'; day=pd.Timestamp(year,10,20)+pd.Timedelta(days=2*i)
                players.append(dict(PLAYER_ID=1,PLAYER_NAME='One',TEAM_ID=10,TEAM_ABBREVIATION='AAA',
                    SEASON=season,GAME_ID=gid,GAME_DATE=day,MATCHUP='AAA vs. BBB',
                    PTS=30. if year==2025 else 12.,MIN=36. if year==2025 else 18.,
                    FGA=20. if year==2025 else 10.,FTA=4.,FG3A=5.))
                for tid,team in [(10,'AAA'),(20,'BBB')]:
                    teams.append(dict(TEAM_ID=tid,TEAM_ABBREVIATION=team,SEASON=season,
                        GAME_ID=gid,GAME_DATE=day,PTS=100.,FGA=80.,FTA=20.,OREB=10.,TOV=12.,MIN=240.))
        return pd.DataFrame(players),pd.DataFrame(teams)

    def test_five_appearances_have_two_thirds_weight_and_no_old_role_in_rolling(self):
        p,t=self.fixture(); f=build_role_features(p,t).reset_index(drop=True)
        row=f.iloc[14]
        self.assertEqual(row.role_games,5)
        self.assertAlmostEqual(row.role_current_weight,2/3)
        self.assertEqual(row.role_min_avg_5,18)
        self.assertEqual(row.pts_avg_5,12)
        self.assertEqual(row.min_blend,24)
        self.assertAlmostEqual(row.minutes_history_estimate,24)
        self.assertAlmostEqual(row.role_fga_share_10,10/80)

    def test_trade_and_return_reset_stint_and_season(self):
        p,t=self.fixture()
        p.loc[12:14,['TEAM_ID','TEAM_ABBREVIATION','MATCHUP']]=[20,'BBB','BBB @ AAA']
        f=build_role_features(p,t).reset_index(drop=True)
        self.assertEqual(f.iloc[9].role_games,0)
        self.assertEqual(f.iloc[12].role_games,0)
        self.assertEqual(f.iloc[15].role_games,0)
        self.assertTrue(pd.isna(f.iloc[15].role_min_avg))
        self.assertEqual(f.iloc[16].role_games,1)

    def test_target_and_future_boxscores_do_not_change_features(self):
        p,t=self.fixture(); before=build_role_features(p,t).reset_index(drop=True)
        cutoff=p.iloc[14].GAME_DATE
        p.loc[p.GAME_DATE.ge(cutoff),['PTS','MIN','FGA','FTA','FG3A']]=999.
        t.loc[t.GAME_DATE.ge(cutoff),['PTS','FGA','FTA','OREB','TOV']]=999.
        after=build_role_features(p,t).reset_index(drop=True)
        pd.testing.assert_frame_equal(before.loc[:14,feature_columns()],after.loc[:14,feature_columns()])

    def test_live_query_matches_training_row_and_dnp_does_not_count(self):
        p,t=self.fixture(); cutoff=p.iloc[14].GAME_DATE
        expected=build_role_features(p,t).reset_index(drop=True).iloc[[14]][feature_columns()]
        query=p.iloc[[14]].copy();query[['PTS','MIN','FGA','FTA','FG3A']]=np.nan
        dnp=p.iloc[[13]].copy();dnp['GAME_ID']='dnp';dnp['GAME_DATE']+=pd.Timedelta(days=1)
        dnp[['PTS','MIN','FGA','FTA','FG3A']]=0.
        actual=build_role_features(pd.concat([p.iloc[:14],dnp,query]),t[t.GAME_DATE.lt(cutoff)])
        pd.testing.assert_frame_equal(expected.reset_index(drop=True),actual.iloc[[-1]][feature_columns()].reset_index(drop=True))

    def test_no_prior_and_no_current_appearances_are_safe(self):
        p,t=self.fixture();p=p[p.SEASON.eq('2026-27')]
        f=build_role_features(p,t).reset_index(drop=True)
        self.assertTrue(pd.isna(f.iloc[0].min_blend))
        self.assertEqual(f.iloc[5].min_blend,18)
        self.assertEqual(f.iloc[5].role_games,5)

    def test_version_dispatch_preserves_historical_defaults_and_validates_schema(self):
        with tempfile.TemporaryDirectory() as tmp, patch('predict_population.ROOT',Path(tmp)):
            models=Path(tmp)/'models';models.mkdir()
            old='points_population_2026-27'; new='points_population_test_v2'
            (models/f'{old}.json').write_text(json.dumps({'features':FEATURES,'data_through':'2026-04-12'}))
            metadata={'features':feature_columns(),'data_through':'2026-04-12',
                'feature_schema_version':'population-role-v2','feature_config':{'shares':True,'prior_strength':5}}
            (models/f'{new}.json').write_text(json.dumps(metadata))
            (models/'active_population_2026-27.json').write_text(json.dumps({'model_version':new,'deployed_at':'2026-11-09T19:00:00Z'}))
            self.assertEqual(resolve_model(pd.Timestamp('2026-11-08'))[0].stem,old)
            self.assertEqual(resolve_model(pd.Timestamp('2026-11-09'))[0].stem,new)
            self.assertEqual(resolve_model(pd.Timestamp('2026-11-09'),old)[0].stem,old)
            with self.assertRaisesRegex(ValueError,'cutoff'):
                resolve_model(pd.Timestamp('2026-04-12'),new)
            metadata['features']=FEATURES;(models/f'{new}.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'schema'):
                resolve_model(pd.Timestamp('2026-11-09'))


if __name__=='__main__': unittest.main()
