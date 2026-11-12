from django.db import migrations


def require_game_identity(apps, schema_editor):
    Game = apps.get_model('core', 'Game')
    if Game.objects.using(schema_editor.connection.alias).filter(nba_game_id__isnull=True).exists():
        raise RuntimeError('Resolve legacy Games without NBA IDs before consolidation; no data was deleted.')


def release_legacy_index_names(apps, schema_editor):
    # Renaming a table leaves index names behind. Free the old names before the
    # replacement tables are created, including PostgreSQL PK/unique indexes.
    connection = schema_editor.connection
    qn = schema_editor.quote_name
    for name in ('LegacyModelPrediction', 'LegacySportsbookLine'):
        table = apps.get_model('core', name)._meta.db_table
        with connection.cursor() as cursor:
            constraints = connection.introspection.get_constraints(cursor, table)
        for index, info in constraints.items():
            if connection.vendor == 'postgresql' and (info.get('index') or info.get('primary_key') or info.get('unique')):
                schema_editor.execute(f'ALTER INDEX {qn(index)} RENAME TO {qn("legacy_" + index[:55])}')
            elif connection.vendor == 'sqlite' and info.get('index') and not index.startswith('sqlite_autoindex'):
                schema_editor.execute(f'DROP INDEX {qn(index)}')
                unique = 'UNIQUE ' if info.get('unique') else ''
                columns = ', '.join(qn(c) for c in info['columns'])
                schema_editor.execute(f'CREATE {unique}INDEX {qn("legacy_" + index[:55])} ON {qn(table)} ({columns})')


class Migration(migrations.Migration):
    dependencies = [('core', '0007_forecast_team_context')]
    operations = [
        migrations.RunPython(require_game_identity, migrations.RunPython.noop),
        migrations.RenameModel('ModelPrediction', 'LegacyModelPrediction'),
        migrations.RenameModel('SportsbookLine', 'LegacySportsbookLine'),
        migrations.RenameField('Game', 'start_time_utc', 'scheduled_at'),
        migrations.RunPython(release_legacy_index_names),
    ]
