import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from population_features import player_features,attach_context,NUMERIC,FEATURES
from fetch_population import make_watchlist

class PopulationFeaturesTest(unittest.TestCase):
    def test_context_accepts_mixed_string_types_without_future_leakage(self):
        teams=pd.DataFrame([
            {'GAME_ID':str(day),'GAME_DATE':pd.Timestamp(2026,10,day),
             'TEAM_ABBREVIATION':team,'PTS':points,'FGA':80,'FTA':20,
             'OREB':10,'TOV':12,'MIN':240}
            for day in [19,20] for team,points in [('AAA',100),('BBB',110)]])
        teams['TEAM_ABBREVIATION']=teams.TEAM_ABBREVIATION.astype('str')
        players=pd.DataFrame({'GAME_DATE':[pd.Timestamp(2026,10,20)],
                              'TEAM_ABBREVIATION':['AAA'],'opponent':['BBB']})
        players['TEAM_ABBREVIATION']=players.TEAM_ABBREVIATION.astype(object)
        original_players=players.copy();original_teams=teams.copy()
        result=attach_context(players,teams)
        self.assertEqual(result.opp_allowed_10.iloc[0],100)
        self.assertAlmostEqual(result.team_possessions_10.iloc[0],90.8)
        teams.loc[teams.GAME_DATE.eq(pd.Timestamp(2026,10,20)),'PTS']=999
        pd.testing.assert_frame_equal(result,attach_context(players,teams))
        pd.testing.assert_frame_equal(players,original_players)
        self.assertEqual(original_teams.TEAM_ABBREVIATION.dtype,teams.TEAM_ABBREVIATION.dtype)

    def fixture(self):
        rows=[]
        for season,year in [('2023-24',2023),('2024-25',2024)]:
            for i in range(12):
                rows.append({'PLAYER_ID':1,'PLAYER_NAME':'Test Player','TEAM_ID':10,
                    'TEAM_ABBREVIATION':'AAA','SEASON':season,'GAME_ID':f'{year}{i:06}',
                    'GAME_DATE':pd.Timestamp(year,11,1)+pd.Timedelta(days=2*i),
                    'MATCHUP':'AAA vs. BBB','PTS':10.+i,'MIN':30.,'FGA':12.,'FTA':4.,'FG3A':5.})
        return pd.DataFrame(rows)

    def test_target_and_future_outcomes_do_not_change_pregame_features(self):
        raw=self.fixture()
        before=player_features(raw)
        changed=raw.copy()
        changed.loc[changed.index>=18,['PTS','MIN','FGA','FTA','FG3A']]=999.
        after=player_features(changed)
        fields=[x for x in NUMERIC if x in before]
        pd.testing.assert_frame_equal(before.loc[:18,fields],after.loc[:18,fields])

    def test_short_target_appearance_remains_eligible(self):
        raw=self.fixture()
        raw.loc[15,'MIN']=2
        result=player_features(raw)
        self.assertTrue(result.loc[15,'eligible'])
        self.assertEqual(result.loc[15,'MIN'],2)
        self.assertGreater(result.loc[15,'min_avg_5'],15)

    def test_opening_prior_and_historical_team_preserved(self):
        raw=self.fixture()
        raw.loc[12:,'TEAM_ID']=20
        result=player_features(raw)
        opening=result.iloc[12]
        self.assertEqual(opening.season_games,0)
        self.assertEqual(opening.prior_mpg,30)
        self.assertEqual(opening.pts_blend,raw.iloc[:12].PTS.mean())
        self.assertEqual(opening.changed_team_since_prior,1)
        self.assertTrue(result.iloc[:12].TEAM_ID.eq(10).all())

    def test_full_vs_truncated_history_has_same_features(self):
        raw=self.fixture()
        target=raw.iloc[[18]].copy()
        target[['PTS','MIN','FGA','FTA','FG3A']]=np.nan
        truncated=pd.concat([raw.iloc[:18],target],ignore_index=True)
        expected=player_features(raw).iloc[[18]]
        actual=player_features(truncated).iloc[[18]]
        columns=[x for x in FEATURES if x in expected]
        pd.testing.assert_frame_equal(expected[columns],actual[columns])

    def test_watchlist_uses_prior_average_and_current_roster(self):
        raw=self.fixture().iloc[:3].copy()
        raw['PLAYER_ID']=[1,1,2];raw['MIN']=[10.,30.,14.]
        roster=pd.DataFrame({'PERSON_ID':[1,2,3],'DISPLAY_FIRST_LAST':['One','Two','Rookie'],
                             'TEAM_ID':[99]*3,'TEAM_ABBREVIATION':['NEW']*3,'ROSTERSTATUS':[1]*3})
        watch,audit=make_watchlist(raw,roster)
        self.assertEqual(watch.player_id.tolist(),[1])
        self.assertEqual(watch.prior_mpg.iloc[0],20)
        self.assertEqual(watch.team.iloc[0],'NEW')
        self.assertTrue(audit[audit.player_id==3].exclusion_reason.iloc[0].startswith('No prior'))

if __name__=='__main__': unittest.main()
