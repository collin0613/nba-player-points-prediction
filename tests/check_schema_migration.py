"""Runs the real migrations forward in an isolated SQLite database."""
import sys, tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import django
django.setup()
from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import override_settings
from django.utils.dateparse import parse_datetime

with tempfile.TemporaryDirectory() as directory, override_settings(BASE_DIR=Path(directory)):
    executor=MigrationExecutor(connection)
    executor.migrate([('core','0007_forecast_team_context')])
    apps=executor.loader.project_state([('core','0007_forecast_team_context')]).apps
    get=lambda name:apps.get_model('core',name).objects
    g=get('Game').create(nba_game_id='0022600001',home_team='DET',away_team='BOS',season='2026-27',start_time_utc=parse_datetime('2026-10-20T19:00:00Z'),sportsgameodds_event_id='YcuJMzH3KxfKzxVKzSQs')
    for number in range(24):
        p=get('Player').create(nba_player_id=number+1,name=f'Preserved player {number}')
        for hour,value in [(10,1.234567),(11,19.734567)]:
            r=get('PredictionRun').create(season='2026-27',generated_at=parse_datetime(f'2026-10-19T{hour}:00:00Z'),feature_cutoff='2026-10-19',model_version='test')
            get('PlayerForecast').create(player=p,game=g,run=r,predicted_points=value,team='BOS',opponent='DET',features={'prior':21.12345})
    executor=MigrationExecutor(connection);executor.migrate([('core','0011_retire_redundant_tables')])
    from core.models import Game,ModelPrediction,DataImport
    assert ModelPrediction.objects.count()==24
    assert set(ModelPrediction.objects.values_list('predicted_points',flat=True))=={19.734567}
    assert all(p.features=={'prior':21.12345} and p.locked_at for p in ModelPrediction.objects.all())
    assert Game.objects.get().sportsgameodds_event_id=='YcuJMzH3KxfKzxVKzSQs'
    audit=DataImport.objects.get();assert len(audit.diagnostics['forecast_redirects'])==48
    assert (Path(directory)/audit.payload_path).exists()
    assert len([t for t in connection.introspection.table_names() if t.startswith('core_')])==6
    print('Migration: 48 forecasts -> 24 exact, locked predictions; six core tables.')
