from datetime import timedelta
from io import StringIO
from unittest.mock import patch
from django.test import TestCase
from django.core.management import call_command
from django.core.management.base import CommandError
from core.models import Game, Player, PredictionRun, ModelPrediction
from core.services.predictions import upsert_prediction
from core.services.results import cleanup_dnps, save_results
from core.services.history import prediction_exclusion
from core.services.historical import complete_historical_run
from src.market_mapping import aware_datetime


class HistoricalPredictionTests(TestCase):
    def setUp(self):
        self.start = aware_datetime('2026-10-25T01:00:00Z')
        self.now = aware_datetime('2026-10-27T02:00:00Z')
        self.game = Game.objects.create(nba_game_id='0022600001', season='2026-27',
            home_team='BOS', away_team='DET', scheduled_at=self.start, status='final')
        Player.objects.create(nba_player_id=1, name='Test Player')
        self.run = PredictionRun.objects.create(run_type='historical_backfill', target_date='2026-10-24',
            generated_at=self.now, feature_cutoff='2026-10-24', season='2026-27', model_version='test',
            metadata={'data_through':'2026-04-12'})
        self.run.refresh_from_db()
        self.row = dict(next_game_start_utc=self.start.isoformat(), predicted_pts='22', home_team='BOS',
            away_team='DET', team='BOS', nba_game_id=self.game.nba_game_id, player_id='1', features_json='{}',
            roster_observed_at='2026-10-23T12:00:00Z')

    def create(self):
        return upsert_prediction(self.run,self.row,now=self.now)

    def test_insert_locked_game_retry_preserves_prediction_and_live_records(self):
        self.assertEqual(self.create(),'created')
        p = ModelPrediction.objects.get()
        self.assertEqual(p.locked_at,self.start)
        self.row['predicted_pts']='999'
        self.assertEqual(self.create(),'existing_preserved')
        p.refresh_from_db();self.assertEqual(p.predicted_points,22)
        self.run.run_type='live';self.run.save()
        other = PredictionRun.objects.create(run_type='historical_backfill', target_date='2026-10-24',
            generated_at=self.now,feature_cutoff='2026-10-24', season='2026-27',metadata={'data_through':'2026-04-12'})
        other.refresh_from_db()
        self.assertEqual(upsert_prediction(other,self.row,now=self.now),'existing_preserved')
        p.refresh_from_db();self.assertEqual(p.prediction_run_id,self.run.pk)

    def test_wrong_date_training_or_roster_cutoff_rejected(self):
        for field,value in [('next_game_start_utc','2026-10-26T01:00:00Z'),
                            ('roster_observed_at','2026-10-24T12:00:00Z')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                upsert_prediction(self.run,{**self.row,field:value},now=self.now)
        self.run.metadata={'data_through':'2026-10-24'}
        with self.assertRaises(ValueError):self.create()
        self.assertFalse(ModelPrediction.objects.exists())

    def test_live_cannot_use_post_tipoff_bypass(self):
        self.run.run_type='live'
        self.assertEqual(self.create(),'non_pregame')

    def test_settlement_keeps_dnp_and_requires_explicit_backfill_evaluation(self):
        self.create()
        save_results(self.game,{1:{'points':None,'played':False}},self.now)
        self.assertEqual(cleanup_dnps(ModelPrediction.objects.all(), self.now),0)
        p=ModelPrediction.objects.get()
        self.assertEqual(p.participation,'dnp');self.assertIsNone(p.actual_points)
        save_results(self.game,{1:{'points':25,'played':True}},self.now+timedelta(seconds=1))
        p=ModelPrediction.objects.select_related('game','prediction_run').get()
        self.assertEqual(prediction_exclusion(p),'historical_backfill_excluded')
        self.assertIsNone(prediction_exclusion(p,include_backfills=True))
        out=StringIO()
        call_command('evaluate_history',since='2026-10-24',until='2026-10-25',
                     prediction_kind='backfill',stored_only=True,show_points=True,stdout=out)
        self.assertIn('historical_backfill: n=1 MAE=3.000',out.getvalue())
        self.assertIn('actual=25',out.getvalue())
        default=StringIO()
        call_command('evaluate_history',date='2026-10-24',stored_only=True,show_rows=True,stdout=default)
        self.assertIn('historical_backfill: n=1 MAE=3.000',default.getvalue())
        live=StringIO()
        call_command('evaluate_history',date='2026-10-24',prediction_kind='live',stored_only=True,stdout=live)
        self.assertNotIn('historical_backfill: n=1',live.getvalue())


    def test_odds_failure_does_not_prevent_scoped_results_and_unresolved_failure(self):
        self.create()
        other=Game.objects.create(nba_game_id='0022600002',season='2026-27',home_team='BOS',
                                  away_team='DET',scheduled_at=self.start+timedelta(days=1))
        with patch('core.services.historical.call_command',side_effect=CommandError('Unavailable')) as odds, \
             patch('core.services.historical.backfill_games',return_value={}) as fetch:
            self.assertFalse(complete_historical_run([self.game.nba_game_id],'2026-10-24'))
            self.assertEqual(list(fetch.call_args.args[0].values_list('pk',flat=True)),[self.game.pk])
            odds.assert_called_once_with('fetch_odds',date='2026-10-24',historical=True)

    def test_historical_odds_archived_without_mutating_locked_lines(self):
        from core.test_sportsbook import load_event
        from core.services.sportsbook import event_identity, import_event
        from core.models import SportsbookLine, DataImport
        _, event = load_event()
        home, away, start = event_identity(event)
        game=Game.objects.create(nba_game_id='0022500001',season='2025-26',home_team=home,
                                 away_team=away,scheduled_at=start,status='final')
        source=DataImport.objects.create(source='sportsgameodds',kind='odds',
            request_fingerprint='fixture',observed_at=start-timedelta(hours=1))
        line=SportsbookLine.objects.create(source_import=source,game=game,player_id=1,bookmaker='draftkings',
            line=20,observed_at=start-timedelta(hours=1),locked_at=start)
        before=SportsbookLine.objects.values().get(pk=line.pk)
        with patch('core.services.sportsbook.timezone.now',return_value=self.now):
            _, item, counts=import_event(event,self.now,['draftkings','fanduel'],historical=True)
        self.assertGreater(counts['historical_quote_rows'],0)
        self.assertFalse(item.diagnostics['pregame_provenance_verified'])
        self.assertTrue(item.request_scope['historical'])
        self.assertEqual(SportsbookLine.objects.values().get(pk=line.pk),before)
        self.assertEqual(SportsbookLine.objects.count(),1)

    def test_explicit_closing_props_are_scored_without_faking_observation_time(self):
        from core.test_sportsbook import load_event
        from core.services.sportsbook import event_identity, import_event
        from core.services.history import line_exclusion
        from core.models import SportsbookLine
        _,event=load_event()
        home,away,start=event_identity(event)
        game=Game.objects.create(nba_game_id='0022500001',season='2025-26',home_team=home,
                                 away_team=away,scheduled_at=start,status='final')
        player=Player.objects.create(nba_player_id=2,name='Jayson Tatum',current_team='LAL')
        ModelPrediction.objects.create(game=game,player=player,prediction_run=self.run,
                                       team_at_prediction='BOS',predicted_points=27)
        event['status'].update(finalized=True,started=True,ended=True)
        for side,price in [('over','-110'),('under','-115')]:
            odd=event['odds'][f'points-JAYSON_TATUM_1_NBA-game-ou-{side}']
            odd['started']=True
            odd['byBookmaker']={'draftkings':dict(overUnder='99.5',odds='-999',available=False,
                closeOverUnder='25.5',closeOdds=price,lastUpdatedAt=self.now.isoformat())}
        with patch('core.services.sportsbook.timezone.now',return_value=self.now):
            _,item,counts=import_event(event,self.now,['draftkings'],historical=True)
        line=SportsbookLine.objects.get(player=player)
        self.assertEqual(float(line.line),25.5)
        self.assertEqual(line.over_price,-110)
        self.assertEqual(line.observed_at,self.now)
        self.assertIsNone(line.over_updated_at)
        self.assertFalse(line.over_available)
        self.assertEqual(line.line_type,'provider_closing')
        self.assertIsNone(line_exclusion(line))
        self.assertEqual(line_exclusion(line,30),'closing_quote_age_unknown')
        before=SportsbookLine.objects.values().get(pk=line.pk)
        with patch('core.services.sportsbook.timezone.now',return_value=self.now):
            import_event(event,self.now,['draftkings'],historical=True)
        self.assertEqual(SportsbookLine.objects.values().get(pk=line.pk),before)

    def test_closing_parser_never_uses_last_live_values(self):
        from core.test_sportsbook import load_event
        from src.odds_api import extract_closing_points
        _,event=load_event()
        rows,_=extract_closing_points(event)
        self.assertEqual(rows,[])
        for side in ['over','under']:
            odd=event['odds'][f'points-JAYSON_TATUM_1_NBA-game-ou-{side}']
            odd['byBookmaker']={'draftkings':dict(closeOverUnder='25.5' if side=='over' else '26.5',closeOdds='-110',
                                                overUnder='20.5',odds='-105')}
        rows,_=extract_closing_points(event,['draftkings'])
        self.assertEqual(len(rows),2)
        self.assertTrue(all(r['over_price'] is None or r['under_price'] is None for r in rows))
