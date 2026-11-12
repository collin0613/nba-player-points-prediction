# Six-table database implementation — 2026-10-20

The approved design in `proposed_database_structure.md` is implemented and live.

- Applied migrations 0008–0011 on the configured PostgreSQL database.
- Six core tables: Game, ModelPrediction, SportsbookLine, Player, PredictionRun, DataImport.
- Admin Analysis and System groups, six stable entry aliases, and redirects for retired list/detail routes.
- Atomic newest-pregame replacements, database uniqueness constraints, source-time checks, and tipoff locks.
- Prediction generation, legacy prediction storage, roster/schedule imports, sportsbook imports, comparisons, final results and evaluation all use the canonical tables.
- Normal prediction generation now stores by default. `--no-store` exports only.

## Preserved opening-night data

All 24 BOS @ DET predictions match the pre-migration backup exactly, including
unrounded points, forecast-time team and original run generation time. Migration
rehearsal also checked feature JSON and run UUIDs against the full source dump.
Game primary key 2, NBA ID `0022600001`, scheduled time `2026-10-20T19:00:00Z`,
and SportsGameOdds event `YcuJMzH3KxfKzxVKzSQs` are preserved. Predictions are locked.
NBA's result endpoint still reported not-final during this run; no actuals were invented.

## Executed refresh and verification

1. Fetched fresh NBA roster, schedule and current-season statistics with `src/predict.py --store`.
2. Corrected schedule ingestion to handle six NBA placeholders with unassigned teams; imported the completed run using `sync_population`.
3. Successfully reran the complete `src/predict.py --offline --store` command using those fresh caches. Final run: `85abb389-f43a-4b22-aa25-d55ce3ad7f1e`.
4. Refreshed SGO events and odds: 57 lines updated for PHI @ NYK and OKC @ SAS.
5. Comparison produced 48 available bookmaker comparisons for 27 players, sorted by absolute difference. Availability counts change over time.
6. Ran official-results verification and evaluation: BOS @ DET not final; no verified results to score yet.
7. Reimported the final run: 348 unchanged, zero new prediction rows.

Final verified totals: 1,200 assigned Games, 374 unique player/game ModelPredictions,
85 unique player/game/book SportsbookLines. The fresh run covers 348 players across
30 teams. Six unassigned matchups remain in the archived schedule with diagnostics.

28 Django tests and 14 ML pipeline tests pass. Both a synthetic legacy migration and
a rehearsal against the actual backup passed. Live PostgreSQL checks verified six
admin lists, six aliases, game details, a historical forecast redirect, uniqueness,
full-precision BOS/DET preservation and idempotent run replay. Django system checks,
migration-state checks and Python compilation passed.

## Backup and archives

- `.cache/backups/before-six-tables-20261020T203928.json`: full Django data export before migration, including private authentication records; keep private.
- `.cache/backups/bos-det-before.json`: explicit opening-game preservation baseline.
- `.cache/backups/models-before-six-tables.py` and `admin-before-six-tables.py`: prior model/admin source snapshots.
- `data/archive/pre_consolidation_*.json`: retired-table records, Games and PredictionRuns.
- DataImport with scope operation `six_table_consolidation`: counts, redirect mappings and quarantine report (empty).
- `data/archive/nba/` and `data/archive/sportsgameodds/`: retained source payloads.

These artifacts are local and ignored by Git. Nothing was committed or pushed.
