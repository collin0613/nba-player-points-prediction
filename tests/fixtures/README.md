# Provider fixtures

- `sportsgameodds/event.json`: trimmed from a successful authenticated v2/events
  NBA request on 2026-10-19. Retains three real PlayerOverview objects, paired
  full-game points, unavailable DraftKings quotes, available FanDuel quotes, and
  excluded markets. Deeplinks and unrelated markets were removed. No credentials.
  Shape checked against https://sportsgameodds.com/openapi.json on that date.
- `nba_final_summary.json` / `nba_final_boxscore.json`: trimmed successful official
  NBA Stats BoxScoreSummaryV3 / BoxScoreTraditionalV3 responses for 0022500002,
  retrieved 2026-10-19. Retain actual team scores, all player points/minutes and
  DNP comments. These are fixtures only, never production game identifiers.
- Tests copy these fixtures and explicitly mutate times, identities, availability,
  thresholds and final status for adverse cases. No test makes live API requests.

- `nba_team_gamelogs.csv`: 64 official 2024–25 team-game rows, retaining
  both teams for the first 16 BOS and LAL games. This small subset tests paired
  game validation, pregame rolling windows, and training/inference agreement
  without depending on downloaded files under `data/`.
