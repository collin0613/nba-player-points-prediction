from datetime import timedelta
from decimal import Decimal
from io import StringIO
from unittest.mock import patch
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase, SimpleTestCase
from core.models import Game,Player,PredictionRun,ModelPrediction,DataImport,SportsbookLine
from core.services.history import date_range
from src.market_mapping import aware_datetime


class HistoryTests(TestCase):
    def setUp(self):
        self.start=aware_datetime('2026-10-21T01:30:00Z')  # October 20 in New York
        self.game=Game.objects.create(nba_game_id='0022600001',season='2026-27',home_team='DET',away_team='BOS',scheduled_at=self.start,status='final')
        self.run=PredictionRun.objects.create(season='2026-27',generated_at=self.start-timedelta(hours=4),feature_cutoff='2026-10-20',model_version='test')
        self.source=DataImport.objects.create(source='nba',kind='results',request_fingerprint='fixture',observed_at=self.start+timedelta(hours=3))
        # First three picks: over win, under loss, push. Fourth falls below threshold.
        for pid,pred,actual,line in [(1,24,25,20),(2,16,25,20),(3,24,20,20),(4,21,23,20)]:
            self.add(pid,pred,actual,line)

    def add(self,pid,pred,actual,line):
        player=Player.objects.create(nba_player_id=pid,name=f'Player {pid}')
        p=ModelPrediction.objects.create(game=self.game,player=player,prediction_run=self.run,team_at_prediction='BOS',predicted_points=pred,
            actual_points=actual,participation='played',result_verified_at=self.start+timedelta(hours=3),result_import=self.source)
        SportsbookLine.objects.create(game=self.game,player=player,bookmaker='draftkings',line=Decimal(line),over_price=-110,under_price=102,
            over_available=True,under_available=True,market_status='available',observed_at=self.start-timedelta(hours=2),source_import=self.source)
        return p

    def output(self,*args,**kwargs):
        out=StringIO()
        with patch('requests.sessions.Session.get',side_effect=AssertionError('evaluation must not use HTTP')):
            call_command('evaluate_history',*args,stdout=out,stored_only=True,**kwargs)
        return out.getvalue()

    def test_date_threshold_push_and_metrics_scope(self):
        before=list(ModelPrediction.objects.values())
        out=self.output('--date','2026-10-20','--diff-threshold','4','--show-rows','--per-player')
        self.assertIn('n=4 MAE=',out);self.assertIn('N=3 | Wins: 1 | Losses: 1 | Pushes: 1',out)
        self.assertIn('O/U Win Rate: 50.0%',out);self.assertIn('OVER  PUSH',out)
        self.assertIn('Prediction model: n=3',out);self.assertEqual(before,list(ModelPrediction.objects.values()))
        self.assertNotIn('All eligible prediction rows',out)
        self.assertNotIn('Line policy:',out)
        self.assertNotIn('Player 4',out)
        self.assertEqual(out.count('✅') + out.count('❌') + out.count('OVER  PUSH'),3)
        self.assertIn('Optimal Diff (2026-10-20): 1.0',out)
        self.assertIn('N=0',self.output(date='2026-10-21'))

    def test_model_version_filter_keeps_original_and_revised_results_separate(self):
        self.assertIn('n=4 MAE=',self.output(model_version='test'))
        self.assertNotIn('n=4 MAE=',self.output(model_version='other-version'))

    def test_underscore_alias_bookmaker_name_player_and_range(self):
        out=self.output('--diff_threshold','4',since='2026-10-20',until='2026-10-20',bookmaker='DRAFTKINGS',player='player_1')
        self.assertIn('N=1 | Wins: 1',out)
        self.assertIn('missing_line',self.output(bookmaker='FanDuel'))

    def test_yesterday_uses_new_york_calendar(self):
        with patch('core.services.history.timezone.now',return_value=aware_datetime('2026-10-22T02:00:00Z')):
            self.assertIn('N=4',self.output(yesterday=True))  # NY Oct21 -> yesterday Oct20

    def test_dnp_unverified_invalid_context_and_played_zero(self):
        ModelPrediction.objects.filter(player_id=1).update(participation='dnp',actual_points=None)
        ModelPrediction.objects.filter(player_id=2).update(result_verified_at=None)
        ModelPrediction.objects.filter(player_id=3).update(prediction_status='withdrawn')
        ModelPrediction.objects.filter(player_id=4).update(predicted_points=0,actual_points=0)
        out=self.output();self.assertIn('n=1 MAE=0.000',out);self.assertIn('N=1 | Wins: 1',out)
        self.assertIn('DNP_or_unknown',out);self.assertIn('awaiting_verified_final_result',out)

    def test_postgame_predictions_and_nonfinal_games_excluded(self):
        self.run.generated_at=self.start;self.run.save();self.assertIn('N=0',self.output())
        self.run.generated_at=self.start-timedelta(hours=1);self.run.save()
        self.game.status='live';self.game.save();self.assertIn('N=0',self.output())

    def test_optional_age_guard_and_bad_market_postgame_lines(self):
        self.assertIn('stale_observation',self.output(max_age_minutes=30))
        SportsbookLine.objects.filter(player_id=1).update(observed_at=self.start)
        SportsbookLine.objects.filter(player_id=2).update(market_status='suspended')
        SportsbookLine.objects.filter(player_id=3).update(line=None,market_status='missing')
        SportsbookLine.objects.filter(player_id=4).update(over_updated_at=self.start+timedelta(seconds=1))
        out=self.output();self.assertIn('N=0',out);self.assertIn('post_tipoff_line',out);self.assertIn('stale_bookmaker_quote',out)

    def test_zero_difference_and_invalid_options(self):
        ModelPrediction.objects.update(predicted_points=20)
        self.assertIn('zero_diff_no_pick',self.output())
        for opts in [dict(date='bad'),dict(date='2026-10-20',since='2026-10-19'),dict(since='2026-10-21',until='2026-10-20'),
                     dict(diff_threshold=-1),dict(diff_threshold=float('nan')),dict(diff_threshold=float('inf')),dict(min_samples=0),dict(bookmaker='bad'),dict(max_age_minutes=0)]:
            with self.subTest(opts=opts),self.assertRaises(CommandError):self.output(**opts)

    def test_forecast_evaluation_date_filters(self):
        out=StringIO();call_command('evaluate_forecasts',date='2026-10-20',stdout=out);self.assertIn('n=4',out.getvalue())
        out=StringIO();call_command('evaluate_forecasts',date='2026-10-21',stdout=out);self.assertNotIn('n=4',out.getvalue())

    def test_backfill_date_scope_and_repeat_skip(self):
        ModelPrediction.objects.update(result_verified_at=None,actual_points=None,participation='pending')
        future=Game.objects.create(nba_game_id='0022600002',season='2026-27',home_team='DET',away_team='BOS',scheduled_at=self.start+timedelta(days=1))
        ModelPrediction.objects.create(game=future,player_id=1,prediction_run=self.run,team_at_prediction='BOS',predicted_points=22)
        with patch('core.services.results.fetch_final_boxscore',return_value=({'boxScoreSummary':{'gameStatus':2}},None)) as fetch:
            call_command('update_actual_points',date='2026-10-20',stdout=StringIO())
        fetch.assert_called_once_with(self.game.nba_game_id)
        ModelPrediction.objects.filter(game=self.game).update(result_verified_at=self.start+timedelta(hours=3),participation='dnp')
        with patch('core.services.results.fetch_final_boxscore') as fetch:
            call_command('update_actual_points',date='2026-10-20',stdout=StringIO())
        fetch.assert_not_called()
        with self.assertRaises(CommandError):call_command('update_actual_points',date='2026-10-20',game_id=self.game.nba_game_id)

    def test_displayed_optimum_reproduces_count_and_rate(self):
        # Two distinct raw magnitudes share the displayed 5.7 boundary.
        ModelPrediction.objects.filter(player_id=1).update(predicted_points=25.66)
        ModelPrediction.objects.filter(player_id=2).update(predicted_points=14.26)
        ModelPrediction.objects.filter(player_id=3).update(predicted_points=26, actual_points=25)
        ModelPrediction.objects.filter(player_id=4).update(predicted_points=25.64, actual_points=19)
        out = self.output(diff_threshold=5.7, show_rows=True)
        self.assertIn('Optimal Diff (All time): 5.7', out)
        self.assertIn('N=3 | Wins: 2 | Losses: 1 | Pushes: 0', out)
        self.assertIn('Win rate = 66.7%, O/U: n=3 | Wins: 2 | Losses: 1 | Pushes: 0', out)
        self.assertIn('Player 1', out)
        self.assertIn('Player 2', out)
        self.assertNotIn('Player 4', out)

    def test_exact_threshold_is_inclusive(self):
        out = self.output(diff_threshold=4)
        self.assertIn('N=3 | Wins: 1 | Losses: 1 | Pushes: 1', out)


class LocalDateTests(SimpleTestCase):
    def test_dst_day_is_25_hours_and_end_is_exclusive(self):
        start,end=date_range({'date':'2026-11-01'})
        from datetime import timezone
        self.assertEqual((end.astimezone(timezone.utc)-start.astimezone(timezone.utc)).total_seconds(),25*3600)

class CleanupTests(HistoryTests):
    def test_confirmed_dnp_cutoff_and_zero(self):
        from core.services.results import cleanup_dnps
        ModelPrediction.objects.filter(player_id=1).update(participation='dnp', actual_points=None)
        ModelPrediction.objects.filter(player_id=2).update(actual_points=None, result_verified_at=None)
        ModelPrediction.objects.filter(player_id=3).update(actual_points=0)
        qs = ModelPrediction.objects.all()
        self.assertEqual(cleanup_dnps(qs, self.start+timedelta(hours=24)), 0)
        self.assertEqual(cleanup_dnps(qs, self.start+timedelta(hours=25), dry_run=True), 1)
        self.assertEqual(qs.count(),4)
        self.assertEqual(cleanup_dnps(qs, self.start+timedelta(hours=25)),1)
        self.assertTrue(qs.filter(player_id=2).exists())
        self.assertEqual(qs.get(player_id=3).actual_points,0)

    def test_settled_is_evaluated(self):
        ModelPrediction.objects.update(prediction_status='settled')
        self.assertIn('N=4', self.output())

    def test_optimum_independent_of_display_threshold(self):
        out = self.output(diff_threshold=100)
        self.assertIn('N=0', out)
        self.assertIn('Optimal Diff (All time): 1.0', out)
