from django.contrib import admin
from .models import Game, ModelPrediction, SportsbookLine

@admin.register(Game)
class GameAdmin(admin.ModelAdmin):
    list_display = ("odds_event_id", "away_team", "home_team", "start_time_utc")
    search_fields = ("home_team", "away_team")
    ordering = ("-start_time_utc",)


@admin.register(ModelPrediction)
class ModelPredictionAdmin(admin.ModelAdmin):
    list_display = (
        "player_name",
        "game",
        "predicted_pts",
        "actual_points",
        "model_version",
    )
    list_filter = ("model_version",)
    search_fields = ("player_name",)
    ordering = ("-created_at",)


@admin.register(SportsbookLine)
class SportsbookLineAdmin(admin.ModelAdmin):
    list_display = (
        "player_name",
        "game",
        "bookmaker",
        "stat",
        "line",
    )
    list_filter = ("bookmaker", "stat")
    search_fields = ("player_name",)
    ordering = ("-fetched_at",)
