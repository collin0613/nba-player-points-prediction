# NBA Player Points Prediction

A machine learning project for predicting NBA player points, comparing forecasts
with sportsbook props, and evaluating predictions against verified game results.
Python and scikit-learn handle data preparation and inference; Django and PostgreSQL
store players, games, predictions, sportsbook lines, and ingestion history. Results
are available through management commands and the Django admin.

The current pipeline targets the 2026–27 regular season. It selects players on
current NBA rosters with at least 15 average minutes per appearance in the previous
season. Features use prior-game scoring, minutes, shot attempts, team context, and
opponent history. Predictions assume the player participates; no injury feed or
provider-supplied minutes projections are integrated.

## Local setup

Prerequisites: Git, Python, and a running PostgreSQL server. The local environment
was tested with Python 3.14; `requirements-lock.txt` records its dependency versions.
`requirements.txt` lists the direct dependencies.

```bash
git clone https://github.com/collin0613/nba-player-points-prediction.git
cd nba-player-points-prediction
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

For a new installation, create a PostgreSQL login and database using an account
with permission to create them:

```bash
createuser --pwprompt nba_ml
createdb --owner=nba_ml ml_sports
cp .env.example .env
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

Edit `.env` with the generated `DJANGO_SECRET_KEY`, your PostgreSQL connection
settings, and `SPORTSGAMEODDS_API_KEY`. NBA Stats does not require an API key.
SportsGameOdds access is required for sportsbook ingestion and comparisons.
Existing installations should retain their database and `.env`.

```bash
python manage.py migrate
python manage.py check
python manage.py createsuperuser
```

## Build the dataset

Run from the repository root with the virtual environment activated:

```bash
python src/fetch_population.py
python src/population_features.py
```

The first command downloads league-wide player and team logs for 2023–24 through
2025–26, fetches the current roster, and builds the eligible-player watchlist.
Historical downloads are reused unless `--refresh` is supplied. The second command
builds the original population training feature dataset at
`data/processed/population/features.csv`, using information available before each
historical game.

The repository includes versioned population models and their JSON metadata.
`models/active_population_2026-27.json` selects the current model for live dates;
the original model is retained for earlier-date requests. Current-model inference
uses the feature schema recorded in that model's metadata.

To retrain and evaluate the **original population model**:

```bash
python src/evaluate_models.py
```

This compares candidates using chronological validation and overwrites the original
population artifact and comparison reports. It does **not** retrain or activate the
current role-feature model, and is not part of the daily prediction workflow.
Model experiments and the legacy seven-player pipeline are described in the
[repository guide](docs/repository_guide.md).

## Daily prediction and evaluation pipeline

```bash
# Backfill verified final results for unresolved games in the lookback window.
python manage.py update_actual_points --days 10

# Evaluate yesterday's stored live predictions and eligible sportsbook lines.
python manage.py evaluate_history --yesterday --stored-only --prediction-kind live --diff-threshold 3 --show-rows

# Refresh NBA inputs, predict upcoming games, and store the forecasts.
python src/predict.py --store

# Refresh sportsbook lines and compare them with stored predictions.
python manage.py compare_predictions_to_props
```

Prediction generation refreshes rosters, schedules, and current-season game logs.
By default it predicts each eligible player's next scheduled game within 14 days.
To restrict generation to today's New York game date, replace the prediction
command with:

```bash
python src/predict.py --predict-date "$(TZ=America/New_York date +%F)" --store
```

Comparison defaults to today's New York date and refreshes odds itself; separate
`fetch_events` and `fetch_odds` commands are available for ingestion troubleshooting.
`--diff-threshold 3` filters evaluated prop comparisons by the absolute difference
between prediction and line. That difference is not a calibrated win probability.

Predictions are stored by default. Use `--no-store` for file export only, or
`--offline --no-store` to reuse existing NBA snapshots without writing to PostgreSQL.
Offline inference requires previously downloaded inputs. Use each command's `--help`
for date filters and other options.

## View results and retain history

```bash
python manage.py runserver
```

Open <http://127.0.0.1:8000/admin/> and sign in with the admin account. The primary
analysis pages show games, model predictions, and sportsbook lines.

PostgreSQL retains the latest pregame prediction per player/game and locks it at
tipoff. Unique run CSVs and metadata are saved under `data/predictions/`; root-level
`predictions_*.csv` files are convenience exports. Preserve run files, source
archives, and database backups locally. Recreating a past-date prediction does not
recover the original forecast or establish live performance.

Credentials, downloaded datasets, prediction exports, reports, and private archives
are excluded from Git. Small test fixtures and the selected model artifacts are
versioned. A fresh clone does not include your local database or forecast history.

## Tests

```bash
python -m unittest discover -s tests -v
python manage.py test core --settings=tracker.test_settings --noinput
```

Tests use local fixtures and an isolated SQLite database; they do not require live
API requests or modify the configured PostgreSQL database. The Django settings are
for local development; cloud scheduling and production hosting require additional
configuration.
