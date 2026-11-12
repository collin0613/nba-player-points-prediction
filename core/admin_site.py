"""Analysis-first navigation, with compatibility redirects for retired tables."""
from django.contrib.admin import AdminSite
from django.http import HttpResponseRedirect
from django.urls import path, reverse


class NBAAdminSite(AdminSite):
    site_header = 'NBA points predictions'
    site_title = 'NBA analysis'
    index_title = 'Analysis'

    def get_app_list(self, request, app_label=None):
        apps = super().get_app_list(request, app_label)
        result = []
        for app in apps:
            if app['app_label'] != 'core':
                result.append(app)
                continue
            for name, models in [('Analysis', ['Game', 'ModelPrediction', 'SportsbookLine']),
                                 ('System', ['Player', 'PredictionRun', 'DataImport'])]:
                group = [m for key in models for m in app['models'] if m['object_name'] == key]
                if group:
                    result.append({**app, 'name': name, 'models': group})
        return sorted(result, key=lambda a: {'Analysis': 0, 'System': 1}.get(a['name'], 2))

    def get_urls(self):
        aliases = {'games': 'game', 'predictions': 'modelprediction', 'lines': 'sportsbookline',
                   'system/players': 'player', 'system/prediction-runs': 'predictionrun', 'system/data-imports': 'dataimport'}
        def redirect_to(model, mapping=None):
            def view(request, old_id=None):
                from core.models import DataImport
                destination = reverse(f'admin:core_{model}_changelist')
                if old_id and mapping:
                    audit = DataImport.objects.filter(request_scope__operation='six_table_consolidation').first()
                    new_id = audit.diagnostics.get(mapping, {}).get(str(old_id)) if audit else None
                    if new_id:
                        destination = reverse(f'admin:core_{model}_change', args=[new_id])
                return HttpResponseRedirect(destination)
            return self.admin_view(view)
        urls = [path(route+'/', redirect_to(model)) for route, model in aliases.items()]
        retired = {'playerforecast': ('modelprediction', 'forecast_redirects'),
                   'sportsbookquote': ('sportsbookline', 'quote_redirects'),
                   'forecastcomparison': ('sportsbookline', 'comparison_redirects'),
                   'playergameresult': ('modelprediction', 'result_redirects'),
                   'rostersnapshot': ('player', 'roster_redirects'),
                   'providerplayer': ('player', 'provider_redirects'),
                   'oddssnapshot': ('dataimport', 'snapshot_redirects')}
        for old, (new, mapping) in retired.items():
            urls += [path(f'core/{old}/', redirect_to(new)),
                     path(f'core/{old}/<str:old_id>/change/', redirect_to(new, mapping))]
        return urls + super().get_urls()
