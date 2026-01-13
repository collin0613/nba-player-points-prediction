# core/models.py
# Django models for NBA player points prediction app.

from django.db import models


# One row per NBA game, keyed by Odds API event_id
class Game(models.Model):
    odds_event_id = models.BigIntegerField(unique=True)
    home_team = models.CharField(max_length=50)
    away_team = models.CharField(max_length=50)
    start_time_utc = models.DateTimeField()

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.away_team} @ {self.home_team}"


# One row per player per game prediction. Extension of Game.
class ModelPrediction(models.Model):
    game = models.ForeignKey(Game, on_delete=models.CASCADE, related_name="predictions")

    player_id = models.BigIntegerField()
    player_name = models.CharField(max_length=100)

    predicted_pts = models.FloatField()

    # filled later (after game is played)
    actual_points = models.FloatField(null=True, blank=True)

    model_version = models.CharField(max_length=50)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("game", "player_id")

    def __str__(self):
        return f"{self.player_name} ({self.predicted_pts})"


# One row per player per bookmaker per game. Extension of Game.
class SportsbookLine(models.Model):
    game = models.ForeignKey(Game, on_delete=models.CASCADE)

    bookmaker = models.CharField(max_length=50)

    player_name = models.CharField(max_length=100)
    stat = models.CharField(max_length=20, default="points")

    line = models.FloatField()
    over_price = models.FloatField(null=True, blank=True)
    under_price = models.FloatField(null=True, blank=True)

    fetched_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("game", "bookmaker", "player_name", "stat")

    def __str__(self):
        return f"{self.player_name} {self.stat} {self.line} ({self.bookmaker})"
