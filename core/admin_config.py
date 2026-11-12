from django.contrib.admin.apps import AdminConfig


class NBAAdminConfig(AdminConfig):
    default_site = 'core.admin_site.NBAAdminSite'
