# Projected minutes: historical research

Historical research only. The current integration uses SportsGameOdds for props
and NBA Stats for observed results; no projected-minutes provider is connected.
The provider and credential observations below describe the original investigation,
not the current configuration.

Reviewed October 19, 2026. No provider-supplied minutes projection is connected in
this implementation. The existing NBA Stats endpoints supply observed minutes,
not injury-adjusted pregame forecasts. The Odds-API.io integration retrieves prices
and events. A Sportradar key is present, but this review did not establish its
entitlement to a projections product or a usable projected-minutes response.
No paid subscription was purchased and no credentials were printed.

## Provider options

- **SportsDataIO:** Its NBA PlayerGameProjection schema explicitly includes projected
  Minutes, Seconds, Started, injury fields, and Updated. Its documented date-based
  endpoint is `PlayerGameProjectionStatsByDate`. This is the clearest documented
  option for a programmatic integration, subject to subscription and historical access.
  https://sportsdata.io/developers/api-documentation/nba
  https://sportsdata.io/developers/data-dictionary/nba
- **RotoWire:** Publishes a dedicated NBA projected-minutes page. Useful for manual
  review; check licensed feed/export availability before assuming automated access.
  https://www.rotowire.com/basketball/projected-minutes.php
- **Sportradar:** Documents forward-looking player projections under the separate
  Synergy Advanced Basketball API. Product access, season IDs, exact response fields,
  player-ID mapping, and historical snapshots need verification. Having an NBA key
  alone does not establish this product's availability.
  https://developer.sportradar.com/basketball/reference/get_advanced-basketball-league-playerprojections

## What was implemented instead

The watchlist uses current roster membership plus previous-season average minutes
>=15. A historical minutes estimate blends the previous/current season mean with a
recent minutes EWMA. It is explicitly labeled as historical, not provider-supplied,
and receives exactly the same point-in-time calculation in training and prediction.

To integrate a provider later, store provider player ID, NBA ID, game ID, projected
minutes, projection publication time, and retrieval time. Backtests must use a
projection published before that game's forecast cutoff, not a revised historical
feed. When historical projections are unavailable, collect snapshots and evaluate
them prospectively before treating the feature as an established improvement.
