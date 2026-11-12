"""Three analysis tables and three supporting tables; latest pregame state only."""
import uuid
from django.db import models
from django.utils import timezone


def choices(*values):
    return [(value, value.replace('_', ' ').title()) for value in values]


class DataImport(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    source = models.CharField(max_length=30, choices=choices('nba', 'sportsgameodds'))
    kind = models.CharField(max_length=30, choices=choices('schedule', 'roster', 'odds', 'results'))
    request_fingerprint = models.CharField(max_length=64)
    observed_at = models.DateTimeField()
    imported_at = models.DateTimeField(default=timezone.now)
    status = models.CharField(max_length=20, choices=choices('success', 'partial', 'failed'), default='success')
    request_scope = models.JSONField(default=dict)
    row_count = models.PositiveIntegerField(default=0)
    diagnostics = models.JSONField(default=dict)
    payload_path = models.TextField(blank=True, default='')
    payload_sha256 = models.CharField(max_length=64, blank=True, default='')

    class Meta:
        constraints = [models.UniqueConstraint(fields=['source', 'kind', 'request_fingerprint', 'observed_at'], name='import_observation_identity')]
        ordering = ['-observed_at']

    def __str__(self):
        return f'{self.source} {self.kind} — {timezone.localtime(self.observed_at):%Y-%m-%d %H:%M %Z}'


class Game(models.Model):
    nba_game_id = models.CharField(max_length=10, unique=True)
    season = models.CharField(max_length=7, blank=True, default='')
    home_team = models.CharField(max_length=5)
    away_team = models.CharField(max_length=5)
    scheduled_at = models.DateTimeField()
    status = models.CharField(max_length=20, choices=choices('scheduled', 'postponed', 'live', 'final', 'cancelled'), default='scheduled')
    actual_started_at = models.DateTimeField(null=True, blank=True)
    sportsgameodds_event_id = models.CharField(max_length=128, unique=True, null=True, blank=True)
    schedule_checked_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(default=timezone.now)

    class Meta:
        constraints = [models.CheckConstraint(condition=~models.Q(home_team=models.F('away_team')), name='game_distinct_teams')]

    def __str__(self):
        return f'{self.away_team} @ {self.home_team} — {timezone.localtime(self.scheduled_at):%Y-%m-%d %H:%M %Z}'

    def is_locked(self, now=None):
        return (self.actual_started_at is not None or self.scheduled_at <= (now or timezone.now())
                or self.status in ('live', 'final', 'cancelled'))


class Player(models.Model):
    nba_player_id = models.BigIntegerField(primary_key=True)
    name = models.CharField(max_length=128)
    current_team = models.CharField(max_length=5, null=True, blank=True)
    roster_season = models.CharField(max_length=7, null=True, blank=True)
    prior_mpg = models.FloatField(null=True, blank=True)
    prior_games = models.PositiveIntegerField(default=0)
    eligible = models.BooleanField(default=False)
    roster_observed_at = models.DateTimeField(null=True, blank=True)
    roster_import = models.ForeignKey(DataImport, null=True, blank=True, on_delete=models.PROTECT, related_name='+')
    sportsgameodds_player_id = models.CharField(max_length=128, unique=True, null=True, blank=True)
    sportsgameodds_name = models.CharField(max_length=128, blank=True, default='')
    mapping_method = models.CharField(max_length=32, blank=True, default='')
    mapping_status = models.CharField(max_length=20, choices=choices('unmapped', 'verified', 'conflict'), default='unmapped')
    mapping_validated_team = models.CharField(max_length=5, blank=True, default='')
    mapping_validated_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return self.name


class PredictionRun(models.Model):
    run_type = models.CharField(max_length=24, choices=choices('live', 'historical_backfill'), default='live')
    target_date = models.DateField(null=True, blank=True)
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    season = models.CharField(max_length=7)
    generated_at = models.DateTimeField()
    feature_cutoff = models.DateField()
    model_version = models.CharField(max_length=100)
    model_sha256 = models.CharField(max_length=64, blank=True, default='')
    feature_schema_version = models.CharField(max_length=50, blank=True, default='')
    metadata = models.JSONField(default=dict)
    output_path = models.TextField(blank=True, default='')
    output_sha256 = models.CharField(max_length=64, blank=True, default='')
    row_count = models.PositiveIntegerField(default=0)
    status = models.CharField(max_length=20, choices=choices('success', 'partial', 'failed'), default='success')
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f'{self.model_version} — {timezone.localtime(self.generated_at):%Y-%m-%d %H:%M %Z}'


class ModelPrediction(models.Model):
    game = models.ForeignKey(Game, on_delete=models.PROTECT, related_name='model_predictions')
    player = models.ForeignKey(Player, on_delete=models.PROTECT)
    predicted_points = models.FloatField()
    team_at_prediction = models.CharField(max_length=5)
    prediction_run = models.ForeignKey(PredictionRun, on_delete=models.PROTECT, related_name='predictions')
    features = models.JSONField(default=dict)
    minutes_history_estimate = models.FloatField(null=True, blank=True)
    participation_assumption = models.TextField(default='Conditional on appearance; injuries not modeled')
    prediction_status = models.CharField(max_length=20, choices=choices('current', 'settled', 'withdrawn', 'needs_review'), default='current')
    status_reason = models.TextField(blank=True, default='')
    last_checked_at = models.DateTimeField(default=timezone.now)
    value_changed_at = models.DateTimeField(default=timezone.now)
    locked_at = models.DateTimeField(null=True, blank=True)
    actual_points = models.PositiveIntegerField(null=True, blank=True)
    participation = models.CharField(max_length=20, choices=choices('pending', 'played', 'dnp', 'unknown'), default='pending')
    result_verified_at = models.DateTimeField(null=True, blank=True)
    result_import = models.ForeignKey(DataImport, null=True, blank=True, on_delete=models.PROTECT, related_name='+')
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(default=timezone.now)

    class Meta:
        constraints = [models.UniqueConstraint(fields=['game', 'player'], name='one_prediction_per_player_game'),
                       models.CheckConstraint(condition=models.Q(predicted_points__gte=0, predicted_points__lte=1.7976931348623157e308), name='prediction_finite_nonnegative')]

    @property
    def opponent(self):
        if self.team_at_prediction == self.game.home_team:
            return self.game.away_team
        if self.team_at_prediction == self.game.away_team:
            return self.game.home_team
        return ''

    def __str__(self):
        return f'{self.player} — {self.game} — {self.predicted_points:.1f} pts'


class SportsbookLine(models.Model):
    line_type = models.CharField(max_length=24, choices=choices('observed', 'provider_closing'), default='observed')
    game = models.ForeignKey(Game, on_delete=models.PROTECT, related_name='sportsbook_lines')
    player = models.ForeignKey(Player, on_delete=models.PROTECT)
    bookmaker = models.CharField(max_length=50, choices=choices('draftkings', 'fanduel'))
    line = models.DecimalField(max_digits=7, decimal_places=2, null=True, blank=True)
    over_price = models.IntegerField(null=True, blank=True)
    under_price = models.IntegerField(null=True, blank=True)
    over_available = models.BooleanField(default=False)
    under_available = models.BooleanField(default=False)
    market_status = models.CharField(max_length=20, choices=choices('available', 'suspended', 'incomplete', 'missing', 'conflict'), default='missing')
    observed_at = models.DateTimeField()
    value_changed_at = models.DateTimeField(default=timezone.now)
    over_updated_at = models.DateTimeField(null=True, blank=True)
    under_updated_at = models.DateTimeField(null=True, blank=True)
    source_import = models.ForeignKey(DataImport, on_delete=models.PROTECT)
    locked_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(default=timezone.now)

    class Meta:
        constraints = [models.UniqueConstraint(fields=['game', 'player', 'bookmaker'], name='one_line_per_player_game_book'),
                       models.CheckConstraint(condition=models.Q(line__isnull=True) | models.Q(line__gte=0), name='line_nonnegative')]

    def __str__(self):
        return f'{self.player} — {self.bookmaker} {self.line if self.line is not None else self.market_status}'
