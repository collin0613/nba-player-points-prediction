"""Migration exercised independently against a legacy fixture, without production DB access."""
import os
import sys
import subprocess
from pathlib import Path
from django.test import SimpleTestCase


class ConsolidationMigrationTests(SimpleTestCase):
    def test_legacy_predictions_consolidate_and_started_game_stays_exact(self):
        root=Path(__file__).resolve().parents[1]
        result=subprocess.run([sys.executable,str(root/'tests/check_schema_migration.py')],cwd=root,
                              env={**os.environ,'DJANGO_SETTINGS_MODULE':'tracker.test_settings'},capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
