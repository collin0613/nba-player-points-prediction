# Database simplification — approved and implemented

Implemented 2026-10-20 by migrations 0008–0011. The schemas and policy below
record the approved design. See README.md for current commands and
`database_implementation_2026-10-20.md` for historical preservation results and
backup locations. The original implementation plan is retained below; its row counts
and proposed fields are historical, not a description of the current database.

## Recommendation

Use six application tables, but expose only three primary analysis pages:
**Games**, **Model Predictions**, and **Sportsbook Lines**. Keep **Players**,
**Prediction Runs**, and **Data Imports** under a separate System area.
Django's Users and Groups remain separate administration records.

Use one prediction per player/game and one main points market per player/game/book.
Update existing rows in place before tipoff. Do not create another prediction merely
because the same matchup appears in tomorrow's prediction run.

The current code was reviewed in core/models.py, core/admin.py, sync_population,
sportsbook import/matching, comparison, result backfill, evaluation, and tracker/urls.py.
A read-only PostgreSQL inspection found:

- 18 Games, 528 Players, 3 PredictionRuns.
- 1,045 PlayerForecasts for 374 distinct player/game pairs; 348 pairs have multiple forecasts.
- 284 SportsbookQuotes for 85 distinct player/game/bookmaker combinations.
- 1,564 RosterSnapshots, 43 ProviderPlayers, 13 OddsSnapshots, 273 ForecastComparisons.
- ModelPrediction, SportsbookLine, and PlayerGameResult currently have zero rows.

These repeated forecast/quote rows are deliberate historical observations in the
current design. The proposed design changes that retention policy to latest pregame
state, rather than simply hiding repeated rows in the admin.

## Table schemas

PK = primary key; FK = foreign key; ? = nullable. Timestamps are timezone-aware and
stored in UTC. The UI renders America/New_York time, including EST/EDT correctly.
Existing integer Game IDs can remain stable through migration.

### 1. Game — one NBA game

```text
id                          bigint PK
nba_game_id                 varchar(10) UNIQUE NOT NULL
season                      varchar(7)
home_team                   varchar(5), validated NBA code
away_team                   varchar(5), validated NBA code
scheduled_at                timestamp
status                      enum(scheduled, postponed, live, final, cancelled)
actual_started_at           timestamp ?
sportsgameodds_event_id      varchar(128) UNIQUE ?
schedule_checked_at         timestamp ?
created_at                  timestamp
updated_at                  timestamp
```

Require different home/away teams. The official NBA ID determines identity; a
rescheduled start does not create another game. External IDs stay hidden by default.

Main columns: **Game | Date | Time | Status | Predictions | Sportsbook Lines**.
Counts link to the relevant filtered lists. Game detail contains both related lists.
No separate Team table is needed for the current scope; validate the team codes.

### 2. ModelPrediction — one latest prediction per player/game

```text
id                          bigint PK
game_id                     FK Game, PROTECT
player_id                   FK Player, PROTECT
predicted_points            double precision, finite and >= 0
team_at_prediction          varchar(5)
prediction_run_id           FK PredictionRun, PROTECT
features                    JSON object
minutes_history_estimate    double precision ?
participation_assumption    text
prediction_status           enum(current, withdrawn, needs_review)
status_reason               text ?
last_checked_at             timestamp
value_changed_at            timestamp
locked_at                   timestamp ?
actual_points               integer >= 0 ?
participation               enum(pending, played, dnp, unknown)
result_verified_at          timestamp ?
result_import_id            FK DataImport ?
created_at                  timestamp
updated_at                  timestamp
UNIQUE(game_id, player_id)
```

Game supplies date, time, and matchup. Opponent is derived from Game and the saved
team_at_prediction; a later trade must not rewrite that historical team. A player's
current team in Player is not used to display a past prediction.

Generation time, model identity, and feature cutoff are obtained through the run
FK. Keep that relation, features, and minutes in a collapsed **Model details** section.
The prediction value, feature JSON, and run FK must always be updated together.
Features explain the model inputs; this does not itself calculate SHAP contributions
or prove why any particular input raised or lowered the output.

Main columns: **Player | Team | Opponent | Game date | Time | Pred | Actual | Error**.
Error = prediction minus actual; show blank until a verified played result exists.
Display one decimal, retaining full precision in storage. Filters: game date, team,
status, and model/run as an advanced filter.

Because there is now exactly one prediction per player/game, its actual result can
live here too. DNP has NULL points; a played zero is 0. Unknown/missing results are
not automatically zero. Actual-result updates must not modify forecast provenance.
Results for players without predictions are outside this analysis table's scope.

### 3. SportsbookLine — one latest full-game points market per player/game/book

```text
id                          bigint PK
game_id                     FK Game, PROTECT
player_id                   FK Player, PROTECT
bookmaker                   enum(draftkings, fanduel)
line                        decimal(7,2) ?
over_price                  integer ?   # American odds
under_price                 integer ?   # American odds
over_available              boolean
under_available             boolean
market_status               enum(available, suspended, incomplete, missing, conflict)
observed_at                 timestamp
value_changed_at            timestamp
over_updated_at             timestamp ?
under_updated_at            timestamp ?
source_import_id            FK DataImport, PROTECT
locked_at                   timestamp ?
created_at                  timestamp
updated_at                  timestamp
UNIQUE(game_id, player_id, bookmaker)
```

This table is specifically for individual, full-game, main points over/under markets.
Consequently stat and period are fixed by the importer, not free-text dimensions.
Adding other statistics or alternate lines later would require an explicit schema
extension and revised unique constraint.

**The line value is not part of the unique key.** A move from 22.5 to 23.5 replaces
the same row's values. Availability is not inferred from a numerical price. Both
sides must have the same threshold and valid prices to show an actionable pair.
If thresholds conflict, mark incomplete/conflict and clear the normalized paired
line/prices rather than combining them; original side data is available through the
import artifact. A successful complete refresh that omits a previously present
market marks it missing. Failed/partial refreshes cannot claim a market disappeared.

This table relates directly to Game and Player, rather than requiring a prediction
row to exist first. Comparing uses the same game/player keys.

Main columns: **Player | Pred | Line | Diff | Bookmaker | Game | Time | Over | Under**.
For historical views also show/filter the game date. Add availability/freshness
indicators, and offer completed-game Actual/Outcome columns as a separate view.
Pred and Diff are computed by joining ModelPrediction, not copied into another table.
The latest comparison page and CLI sort by absolute Diff, with negative/positive
signs displayed. A missing prediction yields a blank Pred/Diff in the all-lines view
and is excluded from actionable comparisons.

### 4. Player — identity and current roster context (System)

```text
nba_player_id               bigint PK
name                        varchar(128)
current_team                varchar(5) ?
roster_season               varchar(7) ?
prior_mpg                   double precision ?
prior_games                 integer >= 0
eligible                    boolean
roster_observed_at          timestamp ?
roster_import_id            FK DataImport ?
sportsgameodds_player_id     varchar(128) UNIQUE ?
sportsgameodds_name          varchar(128) ?
mapping_method              enum(normalized_name_team, manual_verified) ?
mapping_status              enum(unmapped, verified, conflict)
mapping_validated_team      varchar(5) ?
mapping_validated_at         timestamp ?
```

A single supported odds provider permits its identity mapping to be held here.
Name/team conflict checks still apply on every reuse. Trades update current roster
fields and require mapping revalidation; they do not alter old predictions. The
existing cached NBA historical logs continue to supply historical team membership
for training. Raw roster observations can be archived through DataImport artifacts.
Supporting multiple providers or multiple simultaneous aliases would justify a
separate mapping table later; it is not needed in the proposed initial scope.

### 5. PredictionRun — one model execution (System)

```text
id                          UUID PK
season                      varchar(7)
generated_at                timestamp
feature_cutoff              date
model_version               varchar(100)
model_sha256                char(64)
feature_schema_version      varchar(50)
metadata                    JSON object
output_path                 text ?
output_sha256               char(64) ?
row_count                   integer >= 0
status                      enum(success, partial, failed)
created_at                  timestamp
```

This stores execution metadata, not another set of prediction rows. Run detail can
link to its saved CSV/JSON artifacts. The main predictions table stores the FK and
exact per-player features. Runs may remain after all of their predictions have been
replaced; that is an execution log, not duplicate operational predictions.

### 6. DataImport — ingestion/audit log (System)

```text
id                          UUID PK
source                      enum(nba, sportsgameodds)
kind                        enum(schedule, roster, odds, results)
request_fingerprint         char(64)
observed_at                 timestamp
imported_at                 timestamp
status                      enum(success, partial, failed)
request_scope               JSON object
row_count                   integer >= 0
diagnostics                 JSON object
payload_path                text ?
payload_sha256              char(64) ?
UNIQUE(source, kind, request_fingerprint, observed_at)
```

Request scope records event IDs, dates, bookmaker coverage, and completeness, never
credentials. Distinguish source observation time from import time: rereading yesterday's
cache must not make yesterday's lines look freshly checked. Same cached observation
is idempotent; a new actual API observation may get a new log entry without duplicating
any analysis rows. Durable source artifacts need a separate retained archive; the
current disposable .cache directory must not be treated as historical storage.

## Update, locking, and evaluation policy

1. **First prediction:** insert the player/game row.
2. **Newer pregame run, same value:** keep the same row ID and row count; refresh
   provenance/features if the new run supplies them. Advance last_checked_at.
   Comparing values uses full stored precision, not the rounded UI value.
3. **Newer pregame run, changed value:** replace prediction, features, team context,
   minutes, and run FK together on the same row. Update value_changed_at.
4. **Old CSV or simultaneous execution:** an atomic upsert plus the unique constraint
   prevents duplicates. Only a newer generation timestamp can replace a newer stored
   prediction. Reimporting the same run does nothing; equal-time conflicting runs
   are reported rather than arbitrarily chosen. Results cannot be cleared by import.
5. **Game begins:** refuse further prediction/line changes once official started
   status is known or scheduled tipoff has passed. Retain the final stored pregame
   values for evaluation; a command guard must enforce this even if no scheduler runs.
   Historical prediction imports require a separate validated migration/backfill path.
6. **Results:** update only actual/result fields after verified final status. Official
   corrections can update actuals without altering the pregame prediction or line.
7. **Disappearing eligibility:** only a complete, successfully processed scope can mark
   an earlier forecast withdrawn. A transient API/model failure must not silently
   withdraw valid predictions. Cancellation/reschedule changes are flagged for review;
   a game that actually started is never unlocked merely because its timestamp changes.
8. **Odds refresh:** use equivalent source-time guarded upserts. Incomplete/missing/
   unavailable quotes remain identifiable and excluded; don't resurrect older lines.
   A cached retry does not advance the source observation time.

This implements replacement in place, rather than delete-and-insert, preserving
stable URLs and relationships while removing the previous value from the main row.

**Tradeoff:** PostgreSQL would retain the latest pregame forecast and market state,
not every daily forecast or line movement. Earlier saved run/source artifacts may be
kept outside the analysis tables, but this proposal does not add hidden relational
prediction-revision rows. Reconstructing every previous daily recommendation would
require those artifacts or a separately approved audit-history feature.

Evaluation becomes one player-game observation per model prediction. Run filters
refer only to the retained predictions from that run, not its entire original output.
Paired line evaluation uses the frozen pregame market state and timestamp/freshness
checks at the game cutoff. Suspended/missing/stale lines are excluded. A last observed
pregame line is not necessarily the bookmaker's actual closing line. Reports must not
claim to reproduce every recommendation previously printed by the comparison command.

## Frontend and command proposal

Use Django admin, with the same login and access controls, organized into two areas:

- Analysis: Games, Model Predictions, Sportsbook Lines.
- System: Players, Prediction Runs, Data Imports.

Proposed stable entry routes (authenticated aliases to the appropriate admin views):

```text
/admin/games/
/admin/predictions/
/admin/lines/
/admin/system/players/
/admin/system/prediction-runs/
/admin/system/data-imports/
```

A game detail links to its predictions and lines. Prediction details expand Model
details; line details expand Provider details. Do not embed the entire run/game/
quote string in every cell. Old admin list routes should redirect to the replacement
views. Where several legacy detail IDs collapse to one current row, preserve an
old-to-new redirect mapping in the migration report and handle those detail routes.
These authenticated routes are implemented.

`python src/predict.py` should generate AND upsert predictions by default, then print
and export them. Keep `--store` as a compatible alias; offer `--no-store` explicitly.
The existing CSV-only default repeatedly caused comparisons to use an older DB run.
Database failures should report clearly that the CSV was saved but not imported.
`fetch_odds` upserts SportsbookLine; `compare_predictions_to_props` refreshes and
joins these rows without inserting comparison rows. Backfill updates ModelPrediction
actuals; evaluate reads those same predictions and matched sportsbook rows.

## Migration plan after approval

- Back up PostgreSQL and original run/source artifacts. Never reset the database or
  edit applied migration files.
- Audit V1 and V2 records. Consolidate into the new ModelPrediction and SportsbookLine
  schemas; resolve canonical IDs before imposing NOT NULL and uniqueness.
- For future player-games, keep the newest valid prediction. For games already begun,
  keep only the newest provably pregame prediction. If generation/cutoff provenance
  cannot be established, quarantine it from evaluation instead of guessing.
- Consolidate each bookmaker market using the most recent complete observation,
  including missing/unavailable states. Do not select the newest *available* quote
  if a newer observation suspended or removed it.
- Move ProviderPlayer/current roster fields into Player. Move shared run-level details
  to PredictionRun and observation diagnostics to DataImport. Merge verified result
  fields into their prediction rows. Preserve any orphan/unresolved records in a
  migration archive/report for review instead of silently dropping them.
- Retire PlayerForecast, SportsbookQuote, ForecastComparison, ProviderPlayer,
  RosterSnapshot, OddsSnapshot, and PlayerGameResult as active application tables only
  after their required information is transferred or archived and checks pass.
- Update all imports, CLI commands, evaluation queries, admin registration, routes,
  and related links. Prevent the legacy generator from writing into retired tables.
- Verify stable IDs, newest-wins upserts, repeated-run row counts, no post-tipoff
  mutation, missing/suspended quote handling, historical team correctness, preserved
  actuals, and correct page/CLI rendering before removing obsolete tables.

Approval target: six tables; three primary analysis pages; latest pregame values
updated in place; hidden model/provider details; no per-run duplicate prediction rows.
