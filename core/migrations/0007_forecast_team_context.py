from django.db import migrations


def fill_team_context(apps, schema_editor):
    Forecast = apps.get_model('core', 'PlayerForecast')
    Roster = apps.get_model('core', 'RosterSnapshot')
    alias = schema_editor.connection.alias
    for forecast in Forecast.objects.using(alias).select_related('run', 'game').iterator():
        roster = Roster.objects.using(alias).filter(player_id=forecast.player_id,
            season=forecast.run.season, observed_at__lte=forecast.run.generated_at).order_by('-observed_at').first()
        if roster and roster.team in (forecast.game.home_team, forecast.game.away_team):
            forecast.team = roster.team
            forecast.opponent = forecast.game.away_team if roster.team == forecast.game.home_team else forecast.game.home_team
            forecast.save(using=alias, update_fields=['team', 'opponent'])


class Migration(migrations.Migration):
    dependencies = [('core', '0006_sportsgameodds_and_results')]
    operations = [migrations.RunPython(fill_team_context, migrations.RunPython.noop)]
