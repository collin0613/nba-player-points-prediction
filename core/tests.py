import json
from datetime import timedelta
from django.test import TestCase
from django.db import IntegrityError, transaction
from django.utils import timezone
from core.models import Game, Player, ModelPrediction, PredictionRun
from core.services.predictions import upsert_prediction


class PredictionPersistenceTests(TestCase):
    def setUp(self):
        self.now=timezone.now();self.start=self.now+timedelta(days=2)
        self.player=Player.objects.create(nba_player_id=1,name='Player',current_team='BOS',eligible=True)
        self.run=self.new_run(self.now-timedelta(hours=1))
        self.row={'player_id':1,'nba_game_id':'0022600001','home_team':'DET','away_team':'BOS','team':'BOS',
                  'next_game_start_utc':self.start.isoformat(),'features_json':json.dumps({'prior':12.34}),'predicted_pts':12.34567}

    def new_run(self,when):
        return PredictionRun.objects.create(season='2026-27',generated_at=when,feature_cutoff=timezone.localtime(when).date(),model_version='test')

    def test_same_run_is_idempotent_and_newer_replaces_same_pk(self):
        self.assertEqual(upsert_prediction(self.run,self.row,self.now),'created')
        original=ModelPrediction.objects.get();self.assertEqual(upsert_prediction(self.run,self.row,self.now),'unchanged')
        newer=self.new_run(self.now);row={**self.row,'predicted_pts':14.1,'features_json':'{"prior":15}'}
        self.assertEqual(upsert_prediction(newer,row,self.now),'updated')
        current=ModelPrediction.objects.get();self.assertEqual(current.pk,original.pk);self.assertEqual(current.features,{'prior':15});self.assertEqual(current.prediction_run_id,newer.pk)
        self.assertEqual(upsert_prediction(self.run,self.row,self.now),'older_ignored')
        self.assertEqual(ModelPrediction.objects.get().predicted_points,14.1)

    def test_same_value_updates_provenance_without_changing_value_timestamp(self):
        upsert_prediction(self.run,self.row,self.now);old=ModelPrediction.objects.get()
        newer=self.new_run(self.now);self.assertEqual(upsert_prediction(newer,self.row,self.now),'provenance_updated')
        new=ModelPrediction.objects.get();self.assertEqual(old.value_changed_at,new.value_changed_at);self.assertEqual(new.prediction_run_id,newer.pk)

    def test_tipoff_prevents_replacement_and_postgame_creation(self):
        upsert_prediction(self.run,self.row,self.now)
        run=self.new_run(self.start-timedelta(minutes=1))
        self.assertEqual(upsert_prediction(run,{**self.row,'predicted_pts':99},self.start),'locked')
        self.assertEqual(ModelPrediction.objects.get().predicted_points,12.34567)
        self.assertIsNotNone(ModelPrediction.objects.get().locked_at)
        post=self.new_run(self.start+timedelta(seconds=1))
        self.assertEqual(upsert_prediction(post,self.row,self.start+timedelta(hours=1)),'non_pregame')

    def test_unique_constraint_prevents_duplicate_even_outside_service(self):
        upsert_prediction(self.run,self.row,self.now);p=ModelPrediction.objects.get()
        with self.assertRaises(IntegrityError),transaction.atomic():
            ModelPrediction.objects.create(game=p.game,player=p.player,prediction_run=self.run,predicted_points=5,team_at_prediction='BOS')

    def test_same_timestamp_conflict_is_rejected(self):
        upsert_prediction(self.run,self.row,self.now)
        with self.assertRaisesRegex(ValueError,'identical generation'):
            upsert_prediction(self.new_run(self.run.generated_at),{**self.row,'predicted_pts':55},self.now)

    def test_invalid_values_team_and_cutoff_rejected(self):
        for row in [{**self.row,'predicted_pts':float('nan')},{**self.row,'predicted_pts':-1},{**self.row,'team':'NYK'}]:
            with self.assertRaises(ValueError):upsert_prediction(self.run,row,self.now)
        self.run.feature_cutoff=self.start.date()
        with self.assertRaisesRegex(ValueError,'cutoff'):upsert_prediction(self.run,self.row,self.now)

    def test_rescheduled_game_requires_new_input(self):
        upsert_prediction(self.run,self.row,self.now)
        Game.objects.update(scheduled_at=self.start+timedelta(hours=1))
        with self.assertRaisesRegex(ValueError,'refreshed schedule'):upsert_prediction(self.run,self.row,self.now)

class ImportBoundaryTests(TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        from django.test import override_settings
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.override=override_settings(BASE_DIR=self.root);self.override.enable();self.addCleanup(self.override.disable)
        self.now=timezone.now()

    def test_schedule_archives_unassigned_cup_games_without_creating_invalid_games(self):
        import csv
        from core.services.schedule import sync_schedule
        from core.models import DataImport
        root=self.root/'data/raw/2026-27';root.mkdir(parents=True)
        (root/'schedule.meta.json').write_text(json.dumps({'fetched_at':self.now.isoformat()}))
        valid=dict(gameId='0022600001',seasonYear='2026-27',homeTeam_teamTricode='DET',awayTeam_teamTricode='BOS',homeTeam_teamId='1',awayTeam_teamId='2',gameDateTimeUTC=(self.now+timedelta(days=1)).isoformat(),gameStatus='1')
        pending={**valid,'gameId':'0022601201','homeTeam_teamTricode':'','awayTeam_teamTricode':'','homeTeam_teamId':'0','awayTeam_teamId':'0'}
        with (root/'schedule.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=valid);writer.writeheader();writer.writerows([valid,pending])
        self.assertEqual(sync_schedule('2026-27'),1);self.assertEqual(sync_schedule('2026-27'),1)
        self.assertEqual(Game.objects.count(),1);self.assertEqual(DataImport.objects.count(),1)
        self.assertEqual(DataImport.objects.get().diagnostics['unassigned_matchups'],['0022601201'])

    def test_import_csv_replays_without_duplicates_and_rejects_changed_uuid(self):
        import csv,uuid
        from core.services.predictions import import_predictions
        (self.root/'models').mkdir();(self.root/'models/test.json').write_text('{"feature_schema_version":"population-role-v2"}')
        Player.objects.create(nba_player_id=1,name='Player',eligible=True)
        row=dict(run_id=str(uuid.uuid4()),model_version='test',prediction_season='2026-27',predicted_at_utc=(self.now-timedelta(minutes=1)).isoformat(),feature_cutoff_date_exclusive=str(timezone.localtime(self.now-timedelta(minutes=1)).date()),player_id=1,nba_game_id='0022600001',home_team='DET',away_team='BOS',team='BOS',next_game_start_utc=(self.now+timedelta(days=1)).isoformat(),predicted_pts=12.34567,features_json='{"prior":12}')
        path=self.root/'run.csv'
        def write():
            with path.open('w') as f:
                writer=csv.DictWriter(f,fieldnames=row);writer.writeheader();writer.writerow(row)
        write();run,counts=import_predictions(path,'2026-27');self.assertEqual(counts,{'created':1})
        self.assertEqual(run.feature_schema_version,'population-role-v2')
        _,counts=import_predictions(path,'2026-27');self.assertEqual(counts,{'unchanged':1})
        row['predicted_pts']=90;write()
        with self.assertRaisesRegex(ValueError,'changed CSV'):import_predictions(path,'2026-27')
        self.assertEqual(ModelPrediction.objects.get().predicted_points,12.34567)
