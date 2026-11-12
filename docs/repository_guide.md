# Repository guide

## Application code

Keep all Python modules below. A module can be required through a Django registry,
management command, migration dependency, or subprocess even if a normal import
search finds no callers. Generated files are discussed separately.

### Project configuration

- `manage.py`: Django command entry point.
- `tracker/settings.py`: local Django, PostgreSQL, timezone, and application settings.
- `tracker/test_settings.py`: isolated SQLite configuration for tests.
- `tracker/urls.py`: admin URL routing.
- `tracker/asgi.py` and `tracker/wsgi.py`: Django server entry points.
- `core/apps.py`: Django application registration.
- `core/models.py`: games, players, predictions, sportsbook lines, runs, and imports.
- `core/admin.py`: analysis pages, filters, and record presentation.
- `core/admin_site.py`: navigation and redirects for historical admin routes.
- `core/admin_config.py`: registers the custom admin site.
- All `__init__.py` files: package markers; retain them even when empty.
- All `core/migrations/0001_*.py` through `0014_*.py`: ordered schema and data
  migrations. Keep the full chain, including migrations for retired tables.

### Current ML pipeline and shared utilities

- `src/config.py`: shared paths, seasons, and legacy feature/player definitions.
- `src/data_io.py`: NBA download retries, validation, and file serialization.
- `src/fetch_population.py`: league-wide logs and roster-based watchlist selection.
- `src/population_features.py`: original population features and historical dataset.
- `src/role_features.py`: current-season/team-stint features for the active model.
- `src/model_candidates.py`: candidate models and preprocessing configuration.
- `src/evaluate_models.py`: chronological evaluation and training of the original
  population model; overwrites its artifact, not the active role-model version.
- `src/predict.py`: public prediction entry point; routes to population inference
  by default and provides the explicit `--legacy` implementation and shared helpers.
- `src/predict_population.py`: version selection, input refresh, inference, immutable
  run exports, and database import.
- `src/market_mapping.py`: player/team identity normalization and timestamp parsing.
- `src/odds_api.py`: SportsGameOdds client, caching, retries, and market parsing.
  Its generic filename does not mean it uses the previous odds provider.

### Experiments and compatibility workflows

These are outside the daily workflow but still have a purpose:

- `src/recalibrate.py`: role-feature candidate evaluation and guarded activation.
  It is tied to the recorded season/checkpoint and requires the corresponding local
  data. Review its paths and dates before reusing it for a new experiment.
- `src/replay_recalibration.py`: paired replay against saved original forecasts.
  Requires `reports/recalibration_2026-11-09/original_live.json` and experiment
  outputs; a fresh clone cannot reconstruct that original forecast evidence.
- `src/improve_season.py`: wrapper for population ingestion, original feature
  generation, original-model retraining, and prediction. It does not retrain the
  model selected by the active role-model pointer.
- `src/rebuild_season.py`: orchestrates the seven-player historical training pipeline.
- `src/fetch_data.py`: downloads the configured seven players' logs.
- `src/fetch_league_games.py`: downloads team-game logs for that pipeline.
- `src/build_opponent_defense.py`: builds its historical opponent context.
- `src/process_data.py`: builds its per-player features.
- `src/build_dataset.py`: combines those player feature files.
- `src/train_model.py`: trains and evaluates the legacy Ridge model.
- `src/features.py`: shared legacy rolling calculations, including those exercised
  by temporal regression tests.
- `src/fetch_season_context.py`: roster/schedule inputs for legacy inference.

Do not remove these legacy modules individually: `predict.py --legacy`, the rebuild
wrapper, and tests reference them. A future retirement should remove that interface
and its dependencies together. Legacy Ridge artifacts are generated locally rather
than committed; `python src/rebuild_season.py` rebuilds the default legacy model.

### Django commands

- `fetch_events.py`: match sportsbook events to stored NBA games.
- `fetch_odds.py`: ingest supported player-points markets.
- `compare_predictions_to_props.py`: refresh and compare lines with forecasts.
- `sync_population.py`: import roster/schedule data and optionally a saved run.
- `update_actual_points.py`: verify final NBA games and backfill results.
- `evaluate_history.py`: date-filtered result and prop evaluation, with optional
  result backfill and live/backfill selection.
- `evaluate_forecasts.py`: forecast-level scoring and grouped metrics; also supplies
  metric functions used by `evaluate_history.py`. It is not a redundant duplicate.

All these files live under `core/management/commands/`.

### Django services

- `comparisons.py`: prediction/line joins and quote eligibility.
- `historical.py`: completion of reconstructed runs and their result/odds processing.
- `history.py`: shared evaluation dates, cutoffs, and provenance checks.
- `imports.py`: durable source payloads and ingestion audit records.
- `odds_fetch.py`: bounded sportsbook fetch orchestration.
- `predictions.py`: roster imports and guarded prediction upserts.
- `results.py`: official final boxscores and result persistence.
- `schedule.py`: NBA schedule synchronization.
- `sportsbook.py`: provider identity resolution and guarded line upserts.

All these files live under `core/services/`. In particular, `history.py` and
`historical.py` have different responsibilities.

### Tests and fixtures

- `tests/test_population.py`: watchlist and original population feature behavior.
- `tests/test_recalibration.py`: role features and model-version selection.
- `tests/test_historical_predictions.py`: historical selection and feature cutoffs.
- `tests/test_season_pipeline.py`: rolling windows, season boundaries, and scheduling.
- `tests/check_schema_migration.py`: isolated migration rehearsal invoked by a
  Django test; keep it despite its nonstandard test filename.
- `core/tests.py`: prediction persistence and import boundaries.
- `core/test_sportsbook.py`: provider parsing, matching, quote selection, and results.
- `core/test_history.py`: historical evaluation and local-date filtering.
- `core/test_historical.py`: reconstructed-run persistence and completion.
- `core/test_migrations.py`: runs the migration rehearsal in a subprocess.
- `tests/fixtures/README.md`: fixture provenance and scope.
- `tests/fixtures/nba_team_gamelogs.csv`: compact paired team logs for temporal tests.
- `tests/fixtures/nba_final_summary.json`, `nba_final_boxscore.json`, and
  `sportsgameodds/event.json`: trimmed provider responses for offline tests.

## Models and local data

Keep these five files versioned as one set:

- `models/active_population_2026-27.json`
- `models/points_population_2026-27.joblib`
- `models/points_population_2026-27.json`
- `models/points_population_2026-27_role_v2_20261109.joblib`
- `models/points_population_2026-27_role_v2_20261109.json`

The pointer selects the active version. The original artifact/metadata support
fallback and earlier-date selection. Metadata defines the expected feature schema
and training cutoff. The filenames are version identifiers, not interchangeable
exports. Only load trusted joblib artifacts.

Keep these outside Git:

- `data/raw/`: downloaded logs, rosters, schedules, and observation metadata.
- `data/processed/`: derived features, watchlists, and coverage summaries.
- `data/predictions/`: original per-run CSVs and metadata. Retain for provenance.
- `data/archive/`: source responses and migration records. Retain for auditing.
- `reports/`: generated evaluations, experiment models, forecast exports, and
  Snowflake parity bundles. Some are original evidence, not reproducible scratch.
- `.cache/backups/`: private database and migration backups; never publish or treat
  them as disposable API cache.
- `.cache/sportsgameodds/`: replaceable response cache; deleting it causes refetches.
- Root `predictions_*.csv`: convenience exports, distinct from per-run archives.
- `models/ridge_points_model_*.joblib` and their JSON metadata: optional legacy
  artifacts, rebuilt by the legacy training workflow.
- `.env`, virtual environments, bytecode, logs, local SQLite files, and personal TODOs.

The old flat `data/processed/*_2024-25.csv` files, unseasoned `data/raw/player/`
and `data/raw/league/` directories, and `models/ridge_points_model_2024-25.joblib`
are no longer inputs to the current code or tests. They can be removed if their
historical snapshots are not needed. Their removal does not replace the separate
season-scoped 2024–25 history required by population inference.

## Supporting files and cloud work

- `README.md`: local installation, dataset building, inference, and evaluation.
- `.env.example`: configuration template with placeholders only.
- `.gitignore`: excludes credentials and local/generated artifacts while retaining
  the selected population models and test fixtures.
- `requirements.txt`: human-readable direct dependencies.
- `requirements-lock.txt`: exact versions from the tested local environment.
- `docs/proposed_database_structure.md`: historical database design rationale.
- `docs/database_implementation_2026-10-20.md`: historical migration verification
  and backup record; not a current runbook or current database row count.
- `docs/projected_minutes_sources.md`: historical research on possible minutes feeds;
  provider availability and terms require a new review before integration.
- `scripts/check_snowflake_parity.py`: reusable notebook parity checker, preserved
  outside the ignored report directory. Its uploaded manifest, fixture, and model
  bundle remain local/staged artifacts; it is not the daily automation pipeline.
- `.github/workflows/api-connectivity.yml`, `scripts/check_cloud_apis.py`, and
  `requirements-connectivity.txt`: the separate GitHub Actions connectivity setup.
  Commit these together after the project update. Connectivity success alone does
  not establish end-to-end pipeline reliability.
