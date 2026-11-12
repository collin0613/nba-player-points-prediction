import copy
import json
import tempfile
from datetime import timedelta
from decimal import Decimal
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch
from django.test import TestCase, SimpleTestCase
from django.core.management import call_command
from django.core.management.base import CommandError
from core.models import Game, Player, PredictionRun, ModelPrediction, SportsbookLine, DataImport
from core.services.sportsbook import match_event, import_event, PlayerResolver, MappingError
from core.services.comparisons import collect_comparisons
from core.services.results import parse_final, save_results
from src.market_mapping import aware_datetime, normalize_player_name
from src.odds_api import SportsGameOddsClient, OddsAPIError, extract_points

FIXTURES = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures'


def load_event():
    saved = json.loads((FIXTURES/'sportsgameodds/event.json').read_text())
    return saved, saved['response']['data'][0]


class APIClientTests(SimpleTestCase):
    def test_auth_and_safe_errors(self):
        for code in [401, 403, 429, 500]:
            session = Mock()
            session.get.return_value.status_code = code
            session.get.return_value.text = 'do not print secret-key'
            client = SportsGameOddsClient(api_key='secret-key', session=session)
            with self.assertRaises(OddsAPIError) as raised:
                client.page({'leagueID':'NBA'})
            self.assertNotIn('secret-key', str(raised.exception))
            self.assertEqual(session.get.call_count, 1)
            self.assertEqual(session.get.call_args.kwargs['headers'], {'x-api-key':'secret-key'})
            self.assertNotIn('apiKey', session.get.call_args.kwargs['params'])

    def test_invalid_json_and_shape(self):
        session = Mock()
        session.get.return_value.status_code = 200
        session.get.return_value.json.side_effect = ValueError('bad json')
        with self.assertRaises(OddsAPIError):
            SportsGameOddsClient(api_key='test',session=session).page({})
        session.get.return_value.json.side_effect = None
        session.get.return_value.json.return_value = {'success':False,'data':[]}
        with self.assertRaises(OddsAPIError):
            SportsGameOddsClient(api_key='test',session=session).page({})

    def test_cache_preserves_observation_time_and_pagination(self):
        session = Mock()
        session.get.return_value.status_code = 200
        session.get.return_value.json.return_value = {'data':[], 'nextCursor':None}
        with tempfile.TemporaryDirectory() as cache:
            client = SportsGameOddsClient(api_key='test',cache_dir=cache,session=session)
            first = client.page({'leagueID':'NBA'})
            self.assertEqual(first, client.page({'leagueID':'NBA'}))
            self.assertEqual(session.get.call_count,1)
            self.assertNotIn('x-api-key', next(Path(cache).glob('*.json')).read_text())
        client = SportsGameOddsClient(api_key='test')
        with patch.object(client,'page',side_effect=[{'response':{'data':[], 'nextCursor':'two'}}, {'response':{'data':[]}}]) as page:
            self.assertEqual(len(list(client.events({'leagueID':'NBA'}))),2)
            self.assertEqual(page.call_args.args[0]['cursor'],'two')
        with patch.object(client,'page',return_value={'response':{'data':[],'nextCursor':'loop'}}):
            with self.assertRaises(OddsAPIError):list(client.events({},max_pages=3))

    def test_players_cache_is_separate_and_names_are_preserved(self):
        session = Mock()
        session.get.return_value.status_code = 200
        session.get.return_value.json.return_value = {'data': [
            {'playerID': 'JAYSON_TATUM_1_NBA', 'names': {'display': 'Jayson Tatum'}}]}
        with tempfile.TemporaryDirectory() as cache:
            client = SportsGameOddsClient(api_key='test', cache_dir=cache, session=session)
            client.last_request = -100
            players = client.players('event')
            self.assertEqual(players['JAYSON_TATUM_1_NBA']['names']['display'], 'Jayson Tatum')
            self.assertEqual(client.players('event'), players)
            self.assertEqual(session.get.call_count, 1)
            self.assertTrue(session.get.call_args.args[0].endswith('/players'))
            self.assertEqual(session.get.call_args.kwargs['params']['eventID'], 'event')

    def test_strict_filter_pairing_and_american_prices(self):
        _,event = load_event()
        rows, diagnostics = extract_points(event)
        tatum = [r for r in rows if r['provider_player_id']=='JAYSON_TATUM_1_NBA']
        fd = next(r for r in tatum if r['bookmaker']=='fanduel')
        dk = next(r for r in tatum if r['bookmaker']=='draftkings')
        self.assertEqual(fd['line'],Decimal('26.5'))
        self.assertEqual(fd['over_price'],-114)
        self.assertTrue(fd['over_available']);self.assertFalse(dk['over_available'])
        self.assertGreater(diagnostics['excluded_market'],0)
        odd = event['odds']['points-JAYSON_TATUM_1_NBA-game-ou-under']
        odd['byBookmaker']['fanduel']['overUnder']='28.5'
        rows,_ = extract_points(event)
        fd = [r for r in rows if r['provider_player_id']=='JAYSON_TATUM_1_NBA' and r['bookmaker']=='fanduel']
        self.assertEqual(len(fd),2)
        self.assertTrue(all(r['over_price'] is None or r['under_price'] is None for r in fd))

    def test_alts_bad_values_missing_available_and_conflicts(self):
        _,event = load_event()
        key='points-JAYSON_TATUM_1_NBA-game-ou-over'
        quote=event['odds'][key]['byBookmaker']['fanduel']
        quote['altLines']=[{'overUnder':'99.5','odds':'+100','available':True}]
        quote.pop('available');quote['odds']='1.91'
        rows,_=extract_points(event)
        row=next(r for r in rows if r['provider_player_id']=='JAYSON_TATUM_1_NBA' and r['bookmaker']=='fanduel')
        self.assertFalse(row['over_available']);self.assertIsNone(row['over_price'])
        self.assertFalse(any(r['line']==Decimal('99.5') for r in rows))
        quote['overUnder']='NaN'
        _,diag=extract_points(event);self.assertEqual(diag['invalid_line'],1)

    def test_punctuation_accents_suffixes(self):
        self.assertEqual(normalize_player_name('Luka Dončić'),normalize_player_name('Luka Doncic'))
        self.assertEqual(normalize_player_name('Gary Trent Jr.'),normalize_player_name('Gary Trent'))
        self.assertEqual(normalize_player_name("D’Angelo Russell"),normalize_player_name("D'Angelo Russell"))


class SportsbookWorkflowTests(TestCase):
    def setUp(self):
        saved,self.event=load_event()
        self.observed=aware_datetime(saved['fetched_at'])
        self.game=Game.objects.create(nba_game_id='0022600001', season='2026-27',home_team='DET',away_team='BOS',scheduled_at=aware_datetime(self.event['status']['startsAt']))
        self.player=Player.objects.create(nba_player_id=1628369,name='Jayson Tatum',current_team='BOS',roster_season='2026-27',eligible=True)
        self.run=PredictionRun.objects.create(model_version='test',season='2026-27',generated_at=self.observed-timedelta(hours=1),feature_cutoff=self.observed.date())
        self.prediction=ModelPrediction.objects.create(prediction_run=self.run,game=self.game,player=self.player,predicted_points=28.384729,team_at_prediction='BOS')
        for odd in self.event['odds'].values():
            for quote in odd.get('byBookmaker',{}).values():quote['lastUpdatedAt']=self.observed.isoformat()
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        from django.test import override_settings
        self.settings_override=override_settings(BASE_DIR=Path(self.temp.name));self.settings_override.enable();self.addCleanup(self.settings_override.disable)

    def store(self, event=None, observed=None):
        when=observed or self.observed
        with patch('core.services.sportsbook.timezone.now',return_value=when):
            return import_event(event if event is not None else self.event,when,['draftkings','fanduel'])

    def test_repeats_and_changed_line_update_same_rows(self):
        self.store();pks=list(SportsbookLine.objects.order_by('pk').values_list('pk',flat=True));self.store()
        self.assertEqual(DataImport.objects.count(),1)
        changed=copy.deepcopy(self.event)
        for odd in changed['odds'].values():
            if odd.get('playerID')=='JAYSON_TATUM_1_NBA':
                for q in odd.get('byBookmaker',{}).values(): q['overUnder']='27.5'
        self.store(changed,self.observed+timedelta(minutes=1))
        self.assertEqual(list(SportsbookLine.objects.order_by('pk').values_list('pk',flat=True)),pks)
        rows,_,_=collect_comparisons([self.game],['fanduel'],self.observed+timedelta(minutes=2))
        self.assertEqual(len(rows),1)
        self.assertEqual(ModelPrediction.objects.count(),1)

    def test_sparse_event_retries_failed_import_with_player_metadata(self):
        info = self.event['players']['JAYSON_TATUM_1_NBA']
        for key in ('name', 'names', 'firstName', 'lastName'):
            info.pop(key, None)
        _, item, diag = self.store()
        self.assertGreater(diag['missing_player_name_or_team'], 0)
        self.assertFalse(SportsbookLine.objects.exists())
        lookup = Mock(return_value={'playerID': 'JAYSON_TATUM_1_NBA',
                                   'names': {'display': 'Jayson Tatum'}})
        with patch('core.services.sportsbook.timezone.now', return_value=self.observed):
            import_event(self.event, self.observed, ['draftkings', 'fanduel'], player_lookup=lookup)
        self.assertEqual(DataImport.objects.count(), 1)
        self.assertEqual(SportsbookLine.objects.get(bookmaker='fanduel').line, Decimal('26.5'))
        self.assertEqual(SportsbookLine.objects.get(bookmaker='fanduel').observed_at, self.observed)
        self.assertEqual(SportsbookLine.objects.get(bookmaker='fanduel').source_import_id, item.pk)
        self.player.refresh_from_db()
        self.assertEqual(self.player.sportsgameodds_player_id, 'JAYSON_TATUM_1_NBA')

    def test_known_sparse_player_needs_no_lookup_but_still_checks_roster(self):
        self.store()
        info = self.event['players']['JAYSON_TATUM_1_NBA']
        for key in ('name', 'names', 'firstName', 'lastName'):
            info.pop(key, None)
        lookup = Mock(side_effect=AssertionError('Known players need no HTTP'))
        resolver = PlayerResolver(self.game, self.event, self.observed, lookup)
        self.assertEqual(resolver.resolve('JAYSON_TATUM_1_NBA').pk, self.player.pk)
        self.player.refresh_from_db()
        self.player.current_team = 'LAL'; self.player.save()
        with self.assertRaisesRegex(MappingError, 'roster_conflict'):
            resolver.resolve('JAYSON_TATUM_1_NBA')

    def test_nested_names_and_mismatched_metadata_identity(self):
        info = self.event['players']['JAYSON_TATUM_1_NBA']
        for key in ('name', 'names', 'firstName', 'lastName'):
            info.pop(key, None)
        resolver = PlayerResolver(self.game, self.event, self.observed,
            lambda _: {'playerID': 'WRONG', 'names': {'display': 'Jayson Tatum'}})
        with self.assertRaisesRegex(MappingError, 'invalid_player_identity'):
            resolver.resolve('JAYSON_TATUM_1_NBA')
        info['names'] = {'display': 'Jayson Tatum'}
        self.assertEqual(resolver.resolve('JAYSON_TATUM_1_NBA').pk, self.player.pk)

    def test_latest_missing_market_invalidates_old_and_old_import_cannot_restore(self):
        self.store();changed=copy.deepcopy(self.event);changed['odds']={}
        self.store(changed,self.observed+timedelta(minutes=2))
        self.store(self.event,self.observed+timedelta(minutes=1))
        self.assertEqual(SportsbookLine.objects.get(bookmaker='fanduel').market_status,'missing')
        rows,excluded,_=collect_comparisons([self.game],['fanduel'],self.observed+timedelta(minutes=3))
        self.assertFalse(rows);self.assertEqual(excluded['missing'],1)

    def test_suspension_and_conflicting_sides(self):
        self.store()
        changed=copy.deepcopy(self.event)
        changed['odds']['points-JAYSON_TATUM_1_NBA-game-ou-under']['byBookmaker']['fanduel']['overUnder']='28.5'
        self.store(changed,self.observed+timedelta(minutes=1))
        q=SportsbookLine.objects.get(bookmaker='fanduel');self.assertEqual(q.market_status,'conflict');self.assertIsNone(q.line)
        changed=copy.deepcopy(self.event);changed['status']['oddsAvailable']=False
        self.store(changed,self.observed+timedelta(minutes=2))
        self.assertEqual(SportsbookLine.objects.get(bookmaker='fanduel').market_status,'suspended')

    def test_started_game_freezes_lines(self):
        self.store();q=SportsbookLine.objects.get(bookmaker='fanduel');before=q.line
        self.store(self.event,self.game.scheduled_at+timedelta(seconds=1))
        q.refresh_from_db();self.assertEqual(q.line,before);self.assertIsNotNone(q.locked_at)
        self.assertEqual(DataImport.objects.count(),1)

    def test_provider_start_before_scheduled_tipoff_locks_both_tables(self):
        self.store();changed=copy.deepcopy(self.event);changed['status']['started']=True
        self.store(changed,self.observed+timedelta(minutes=1))
        self.game.refresh_from_db();self.prediction.refresh_from_db()
        self.assertEqual(self.game.status,'live');self.assertIsNotNone(self.prediction.locked_at)
        self.assertTrue(self.game.is_locked(self.observed+timedelta(minutes=2)))
        self.assertEqual(DataImport.objects.count(),1)

    def test_roster_conflict_is_not_mapped(self):
        self.store();self.player.refresh_from_db();self.player.current_team='LAL';self.player.save()
        _,_,diag=self.store(observed=self.observed+timedelta(minutes=1))
        self.assertEqual(diag['mapped_player_roster_conflict'],1)
        self.assertEqual(SportsbookLine.objects.get(bookmaker='fanduel').market_status,'missing')

    def test_ambiguous_game_not_guessed(self):
        Game.objects.create(nba_game_id='0022600002',home_team='DET',away_team='BOS',scheduled_at=self.game.scheduled_at)
        with self.assertRaisesRegex(MappingError,'ambiguous'):match_event(self.event)

    def test_default_comparison_selects_one_and_explicit_books_keep_both(self):
        self.store()
        fd=SportsbookLine.objects.get(bookmaker='fanduel')
        SportsbookLine.objects.filter(bookmaker='draftkings').update(
            line=fd.line,over_price=fd.over_price,under_price=fd.under_price,
            over_available=True,under_available=True,market_status='available',
            observed_at=fd.observed_at,over_updated_at=fd.over_updated_at,under_updated_at=fd.under_updated_at)
        def output(*args):
            out=StringIO()
            with patch('core.management.commands.compare_predictions_to_props.timezone.now',return_value=self.observed+timedelta(seconds=1)):
                call_command('compare_predictions_to_props',*args,date='2026-10-20',stored_only=True,stdout=out)
            return out.getvalue()
        both=output('--bookmakers','draftkings,fanduel')
        default=output()
        self.assertEqual(sum('Jayson Tatum' in line for line in default.splitlines()),1)
        self.assertEqual(sum('Jayson Tatum' in line for line in both.splitlines()),2)
        self.assertIn('FanDuel',output('--bookmaker','fanduel'))
        self.assertNotIn('DraftKings',output('--bookmaker','fanduel'))

    def test_comparison_freshness_and_format(self):
        self.store();out=StringIO()
        with patch('core.management.commands.compare_predictions_to_props.timezone.now',return_value=self.observed+timedelta(seconds=1)),patch('requests.sessions.Session.get',side_effect=AssertionError('no HTTP')):
            call_command('compare_predictions_to_props',date='2026-10-20',bookmakers='FanDuel',stored_only=True,stdout=out)
        self.assertIn('Predictions vs. Props: 2026-10-20',out.getvalue());self.assertIn('28.4',out.getvalue());self.assertIn('+1.9',out.getvalue());self.assertIn('3:00pm EDT',out.getvalue())
        rows,excluded,_=collect_comparisons([self.game],['fanduel'],self.observed+timedelta(hours=1))
        self.assertFalse(rows);self.assertEqual(excluded['stale_observation'],1)
        rows,excluded,_=collect_comparisons([self.game],['fanduel'],self.game.scheduled_at)
        self.assertFalse(rows);self.assertEqual(excluded['game_not_pregame'],1)

    def test_admin_pages_and_old_routes(self):
        self.store()
        from django.contrib.auth import get_user_model
        self.client.force_login(get_user_model().objects.create_superuser('testadmin',password='test-only'))
        for model in ('game','modelprediction','sportsbookline','player','predictionrun','dataimport'):
            self.assertEqual(self.client.get(f'/admin/core/{model}/').status_code,200,model)
        for route in ('games','predictions','lines','system/players','system/prediction-runs','system/data-imports','core/playerforecast','core/sportsbookquote'):
            self.assertEqual(self.client.get('/admin/'+route+'/',follow=True).status_code,200,route)
        response=self.client.get('/admin/');self.assertContains(response,'System');self.assertNotContains(response,'Player forecasts')
        response=self.client.get(f'/admin/core/modelprediction/{self.prediction.pk}/change/')
        self.assertContains(response,'Model tracking and features')
        self.assertNotContains(response,'name="_save"')


class FinalResultsTests(TestCase):
    def setUp(self):
        self.summary=json.loads((FIXTURES/'nba_final_summary.json').read_text());self.stats=json.loads((FIXTURES/'nba_final_boxscore.json').read_text())
        self.game=Game.objects.create(nba_game_id='0022500002',home_team='LAL',away_team='GSW',season='2025-26',scheduled_at=aware_datetime('2025-10-22T02:00:00Z'))
        self.run=PredictionRun.objects.create(model_version='test',season='2025-26',generated_at=self.game.scheduled_at-timedelta(hours=1),feature_cutoff='2025-10-21')
        for pid,name in [(1630559,'Austin Reaves'),(1642355,'Bronny James')]:
            player=Player.objects.create(nba_player_id=pid,name=name)
            ModelPrediction.objects.create(prediction_run=self.run,game=self.game,player=player,predicted_points=25.12345,team_at_prediction='LAL')
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        from django.test import override_settings
        self.override=override_settings(BASE_DIR=Path(self.temp.name));self.override.enable();self.addCleanup(self.override.disable)

    def test_final_update_keeps_forecast_and_dnp_distinct(self):
        with patch('core.services.results.fetch_final_boxscore',return_value=(self.summary,self.stats)) as fetch:
            call_command('update_actual_points',date='2025-10-21',stdout=StringIO())
            call_command('update_actual_points',date='2025-10-21',stdout=StringIO())
        self.assertEqual(fetch.call_count,1)
        p=ModelPrediction.objects.get(player_id=1630559);self.assertEqual(p.actual_points,26);self.assertEqual(p.predicted_points,25.12345)
        self.assertFalse(ModelPrediction.objects.filter(player_id=1642355).exists())
        out=StringIO();call_command('evaluate_forecasts',season='2025-26',stdout=out);self.assertIn('n=1',out.getvalue())

    def test_invalid_nonfinal_and_partial_data_rejected(self):
        summary=copy.deepcopy(self.summary);summary['boxScoreSummary']['gameStatus']=2
        self.assertIsNone(parse_final(self.game,summary,self.stats))
        stats=copy.deepcopy(self.stats);stats['boxScoreTraditional']['gameId']='wrong'
        with self.assertRaisesRegex(ValueError,'ID mismatch'):parse_final(self.game,self.summary,stats)
        stats=copy.deepcopy(self.stats);stats['boxScoreTraditional']['homeTeam']['players'][0]['statistics']['points']=0
        with self.assertRaisesRegex(ValueError,'totals disagree'):parse_final(self.game,self.summary,stats)

    def test_dry_run_writes_nothing(self):
        with patch('core.services.results.fetch_final_boxscore',return_value=(self.summary,self.stats)):
            call_command('update_actual_points',game_id=self.game.nba_game_id,dry_run=True,stdout=StringIO())
        self.assertEqual(DataImport.objects.count(),0)
        self.assertEqual(ModelPrediction.objects.filter(result_verified_at__isnull=False).count(),0)

    def test_history_backfills_before_evaluating_and_skips_settled_games(self):
        out=StringIO()
        with patch('core.services.results.fetch_final_boxscore',return_value=(self.summary,self.stats)) as fetch:
            call_command('evaluate_history',date='2025-10-21',stdout=out)
            call_command('evaluate_history',date='2025-10-21',stdout=StringIO())
        self.assertEqual(fetch.call_count,1)
        self.assertIn('Backfill:',out.getvalue())
        self.assertIn('n=1 MAE=0.877',out.getvalue())
        p=ModelPrediction.objects.get(player_id=1630559)
        self.assertEqual(p.actual_points,26);self.assertEqual(p.predicted_points,25.12345)
        self.assertFalse(ModelPrediction.objects.filter(player_id=1642355).exists())

    def test_history_stored_only_does_not_backfill(self):
        with patch('core.services.results.fetch_final_boxscore') as fetch:
            call_command('evaluate_history',date='2025-10-21',stored_only=True,stdout=StringIO())
        fetch.assert_not_called()
        self.assertFalse(ModelPrediction.objects.filter(result_verified_at__isnull=False).exists())

    def test_history_failure_is_visible_and_does_not_fabricate_results(self):
        out,err=StringIO(),StringIO()
        with patch('core.services.results.fetch_final_boxscore',side_effect=ValueError('NBA request failed')):
            call_command('evaluate_history',date='2025-10-21',stdout=out,stderr=err)
        self.assertIn('NBA request failed',err.getvalue());self.assertIn("'skipped': 1",out.getvalue())
        self.assertFalse(ModelPrediction.objects.filter(result_verified_at__isnull=False).exists())

    def test_history_validates_filters_before_backfill(self):
        with patch('core.services.results.fetch_final_boxscore') as fetch:
            with self.assertRaises(CommandError):
                call_command('evaluate_history',date='2025-10-21',diff_threshold=-1,stdout=StringIO())
        fetch.assert_not_called()

    def test_history_backfill_respects_date_season_player_and_future_games(self):
        from django.utils import timezone
        future=Game.objects.create(nba_game_id='0029900001',home_team='LAL',away_team='GSW',season='2025-26',scheduled_at=timezone.now()+timedelta(days=3))
        ModelPrediction.objects.create(game=future,player_id=1630559,prediction_run=self.run,team_at_prediction='LAL',predicted_points=30)
        with patch('core.services.results.fetch_final_boxscore',return_value=({'boxScoreSummary':{'gameStatus':2}},None)) as fetch:
            call_command('evaluate_history',date='2025-10-22',stdout=StringIO())
            call_command('evaluate_history',season='2026-27',stdout=StringIO())
            call_command('evaluate_history',player='Nonexistent Player',stdout=StringIO())
            fetch.assert_not_called()
            call_command('evaluate_history',player='Austin Reaves',stdout=StringIO())
            fetch.assert_called_once_with(self.game.nba_game_id)
        self.assertFalse(ModelPrediction.objects.filter(result_verified_at__isnull=False).exists())


class PreferredQuoteTests(TestCase):
    def test_edge_coverage_and_draftkings_tiebreakers(self):
        from types import SimpleNamespace
        from collections import Counter
        from core.services.comparisons import select_preferred_rows
        from django.utils import timezone
        start=timezone.now()
        day=timezone.localtime(start).date()
        p=SimpleNamespace(game_id=1,player_id=1,game=SimpleNamespace(scheduled_at=start))
        dk=(p,SimpleNamespace(bookmaker='draftkings'),-4.)
        fd=(p,SimpleNamespace(bookmaker='fanduel'),3.)
        coverage=Counter({(day,'fanduel'):100,(day,'draftkings'):10})
        self.assertEqual(select_preferred_rows([fd,dk],coverage),[dk])
        fd=(p,fd[1],4.)
        self.assertEqual(select_preferred_rows([dk,fd],coverage),[fd])
        coverage[(day,'draftkings')]=100
        self.assertEqual(select_preferred_rows([fd,dk],coverage),[dk])
        self.assertEqual(select_preferred_rows([fd],coverage),[fd])
        p2=SimpleNamespace(game_id=2,player_id=1,game=p.game)
        self.assertEqual(len(select_preferred_rows([dk,fd,(p2,fd[1],4.)],coverage)),2)
