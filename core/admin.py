from django.contrib import admin
from django.db.models import Count, OuterRef, Subquery, FloatField, F, ExpressionWrapper
from django.db.models.functions import Abs, Cast
from django.urls import reverse
from django.utils import timezone
from django.utils.html import format_html
from .models import Game, ModelPrediction, SportsbookLine, Player, PredictionRun, DataImport


class PredictionPresentation:
    presentation_fields = ('team_display', 'points_display', 'actual_display')

    @admin.display(description='Team', ordering='team_at_prediction')
    def team_display(self, obj): return obj.team_at_prediction

    @admin.display(description='Predicted Pts', ordering='predicted_points')
    def points_display(self, obj): return f'{obj.predicted_points:.1f}'

    @admin.display(description='Actual Pts', ordering='actual_points')
    def actual_display(self, obj): return obj.actual_points if obj.actual_points is not None else '—'


class LinePresentation:
    presentation_fields = ('line_display', 'over_display', 'under_display')

    @admin.display(description='Line', ordering='line')
    def line_display(self, obj): return f'{obj.line:.1f}' if obj.line is not None else '—'

    @admin.display(description='Over price', ordering='over_price')
    def over_display(self, obj): return f'{obj.over_price:+d}' if obj.over_price is not None else '—'

    @admin.display(description='Under price', ordering='under_price')
    def under_display(self, obj): return f'{obj.under_price:+d}' if obj.under_price is not None else '—'


class ReadOnlyAdmin(admin.ModelAdmin):
    actions = None
    list_per_page = 100

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def get_readonly_fields(self, request, obj=None):
        return [f.name for f in self.model._meta.fields] + list(getattr(self, 'presentation_fields', ()))


class PredictionInline(PredictionPresentation, admin.TabularInline):
    model = ModelPrediction
    fields = ('player', 'team_display', 'points_display', 'actual_display', 'participation')
    readonly_fields = fields
    extra = 0
    can_delete = False
    show_change_link = True

    def has_add_permission(self, request, obj=None): return False
    def has_change_permission(self, request, obj=None): return False


class LineInline(LinePresentation, PredictionInline):
    model = SportsbookLine
    fields = ('player', 'bookmaker', 'line_display', 'over_display', 'under_display')
    readonly_fields = fields


@admin.register(Game)
class GameAdmin(ReadOnlyAdmin):
    list_display = ('matchup', 'scheduled_local', 'status', 'prediction_count', 'line_count')
    list_filter = ('season', 'status', 'home_team', 'away_team')
    search_fields = ('home_team', 'away_team', 'nba_game_id', 'sportsgameodds_event_id')
    date_hierarchy = 'scheduled_at'
    ordering = ('scheduled_at',)
    inlines = (PredictionInline, LineInline)
    fieldsets = ((None, {'fields': ('home_team','away_team','scheduled_at','status','season')}),
                 ('Identifiers and schedule tracking', {'classes': ('collapse',), 'fields': ('nba_game_id','sportsgameodds_event_id','actual_started_at','schedule_checked_at','created_at','updated_at')}))

    def get_queryset(self, request):
        return super().get_queryset(request).annotate(num_predictions=Count('model_predictions', distinct=True), num_lines=Count('sportsbook_lines', distinct=True))

    @admin.display(description='Game')
    def matchup(self, obj): return f'{obj.away_team} @ {obj.home_team}'

    @admin.display(description='Scheduled (New York)', ordering='scheduled_at')
    def scheduled_local(self, obj): return timezone.localtime(obj.scheduled_at).strftime('%Y-%m-%d %I:%M %p %Z')

    @admin.display(description='Predictions')
    def prediction_count(self, obj):
        return format_html('<a href="{}?game__id__exact={}">{}</a>', reverse('admin:core_modelprediction_changelist'), obj.pk, obj.num_predictions)

    @admin.display(description='Sportsbook lines')
    def line_count(self, obj):
        return format_html('<a href="{}?game__id__exact={}">{}</a>', reverse('admin:core_sportsbookline_changelist'), obj.pk, obj.num_lines)


class GameRelatedAdmin(ReadOnlyAdmin):
    list_select_related = ('player', 'game')
    search_fields = ('player__name', 'game__home_team', 'game__away_team')
    date_hierarchy = 'game__scheduled_at'
    ordering = ('game__scheduled_at', 'player__name')

    @admin.display(description='Player', ordering='player__name')
    def player_name(self, obj): return obj.player.name

    @admin.display(description='Game')
    def matchup(self, obj):
        return format_html('<a href="{}">{} @ {}</a>', reverse('admin:core_game_change', args=[obj.game_id]), obj.game.away_team, obj.game.home_team)

    @admin.display(description='Scheduled (New York)', ordering='game__scheduled_at')
    def scheduled_local(self, obj): return timezone.localtime(obj.game.scheduled_at).strftime('%Y-%m-%d %I:%M %p %Z')


@admin.register(ModelPrediction)
class ModelPredictionAdmin(PredictionPresentation, GameRelatedAdmin):
    list_display = ('player_name', 'team_display', 'opponent', 'matchup', 'scheduled_local', 'points_display', 'actual_display', 'error')
    list_filter = ('game__season', 'prediction_run__run_type', 'team_at_prediction', 'prediction_status', 'participation')
    fieldsets = ((None, {'fields': ('player','game','team_display','points_display','actual_display','participation','prediction_status','status_reason')}),
                 ('Model tracking and features', {'classes': ('collapse',), 'fields': ('prediction_run','features','minutes_history_estimate','participation_assumption','last_checked_at','value_changed_at','locked_at','result_verified_at','result_import','created_at','updated_at')}))

    @admin.display(description='Error')
    def error(self, obj): return f'{obj.predicted_points-obj.actual_points:+.1f}' if obj.actual_points is not None else '—'


@admin.register(SportsbookLine)
class SportsbookLineAdmin(LinePresentation, GameRelatedAdmin):
    list_display = ('player_name', 'prediction', 'line_display', 'difference', 'bookmaker', 'matchup', 'scheduled_local', 'over_display', 'under_display')
    list_filter = ('game__season', 'bookmaker', 'line_type', 'market_status')
    fieldsets = ((None, {'fields': ('player','game','bookmaker','line_display','over_display','under_display')}),
                 ('Observation tracking', {'classes': ('collapse',), 'fields': ('line_type','market_status','over_available','under_available','source_import','observed_at','over_updated_at','under_updated_at','value_changed_at','locked_at','created_at','updated_at')}))

    def get_queryset(self, request):
        prediction = ModelPrediction.objects.filter(game_id=OuterRef('game_id'),player_id=OuterRef('player_id'),prediction_status='current').values('predicted_points')[:1]
        return super().get_queryset(request).annotate(predicted=Subquery(prediction), diff=ExpressionWrapper(Subquery(prediction)-Cast(F('line'),FloatField()),output_field=FloatField())).annotate(abs_diff=Abs('diff')).order_by(F('abs_diff').desc(nulls_last=True), 'player__name')

    @admin.display(description='Pred', ordering='predicted')
    def prediction(self, obj): return f'{obj.predicted:.1f}' if obj.predicted is not None else '—'

    @admin.display(description='Diff', ordering='abs_diff')
    def difference(self, obj): return f'{obj.diff:+.1f}' if obj.diff is not None else '—'


@admin.register(Player)
class PlayerAdmin(ReadOnlyAdmin):
    list_display = ('name','current_team','eligible','prior_mpg','prior_games','mapping_status')
    list_filter = ('current_team','eligible','mapping_status','roster_season')
    search_fields = ('name','sportsgameodds_player_id')
    fieldsets = ((None, {'fields': ('name','current_team','eligible','prior_mpg','prior_games','roster_season')}),
                 ('Roster and provider identity', {'classes': ('collapse',), 'fields': ('nba_player_id','roster_observed_at','roster_import','sportsgameodds_player_id','sportsgameodds_name','mapping_method','mapping_status','mapping_validated_team','mapping_validated_at')}))


@admin.register(PredictionRun)
class PredictionRunAdmin(ReadOnlyAdmin):
    list_display = ('generated_at','run_type','target_date','season','model_version','row_count','status')
    list_filter = ('run_type','season','model_version','status')
    ordering = ('-generated_at',)


@admin.register(DataImport)
class DataImportAdmin(ReadOnlyAdmin):
    list_display = ('observed_at','source','kind','row_count','status')
    list_filter = ('source','kind','status')
    ordering = ('-observed_at',)
