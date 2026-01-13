import os
from typing import Optional, List, Dict, Any, Iterable
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from django.core.management.base import BaseCommand, CommandError
from django.utils.dateparse import parse_datetime
from core.models import Game, SportsbookLine
from nba_api.stats.static import teams

EVENTS_URL = "https://api2.odds-api.io/v3/events"
ODDS_URL = "https://api2.odds-api.io/v3/odds"

LOCAL_TZ = ZoneInfo("America/New_York")

# Build once at import time
_TEAM_NAME_TO_ABBREV = {
    t["full_name"].lower(): t["abbreviation"]
    for t in teams.get_teams()
}

def handle_team_name(team_name: str) -> str:
    """
    Convert Odds API team name (e.g. 'Golden State Warriors', 'Utah Jazz')
    into official NBA abbreviation (e.g. 'GSW', 'UTA') from nba_api.stats.static.teams.

    Falls back to original string if not found.
    """
    if not team_name:
        return None

    key = team_name.strip().lower()
    return _TEAM_NAME_TO_ABBREV.get(key, team_name[:50])

def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def is_today_local(utc_z: str) -> bool:
    """
    utc_z: '2026-01-06T00:00:00Z'
    True if the event occurs on today's local date (America/New_York).
    """
    if not utc_z:
        return False
    try:
        dt_utc = datetime.fromisoformat(utc_z.replace("Z", "+00:00"))
        dt_local = dt_utc.astimezone(LOCAL_TZ)
    except Exception:
        return False

    today_local = datetime.now(LOCAL_TZ).date()
    return dt_local.date() == today_local


def fetch_today_nba_event_ids(api_key: str, limit: int = 50) -> List[str]:
    """
    One call: fetch basketball events, filter to NBA, filter to today's local date.
    Return up to `limit` event ids.
    """
    r = requests.get(
        EVENTS_URL,
        params={"apiKey": api_key, "sport": "basketball"},
        timeout=30,
    )
    if r.status_code != 200:
        raise CommandError(f"Events API error {r.status_code}: {r.text[:300]}")

    events = r.json()
    if not isinstance(events, list):
        raise CommandError("Unexpected events response (expected list).")

    nba_events = [e for e in events if (e.get("league") or {}).get("slug") == "usa-nba"]

    today_events: List[str] = []
    for e in nba_events:
        start_utc = e.get("date")
        if start_utc and is_today_local(start_utc):
            eid = e.get("id")
            if eid is not None:
                today_events.append(str(eid))

    # de-dupe preserving order
    today_events = list(dict.fromkeys(today_events))
    return today_events[:limit]


def _iter_market_dicts(market_obj: Any) -> Iterable[Dict[str, Any]]:
    """Yield dict markets from a list safely."""
    if isinstance(market_obj, list):
        for m in market_obj:
            if isinstance(m, dict):
                yield m


def _looks_like_points_market(name: str) -> bool:
    s = (name or "").strip().lower()
    return "points" in s or "pts" in s


def _odds_entry_hint(o: Dict[str, Any]) -> str:
    # Build a hint string from common fields to detect "points" entries in flat odds lists
    parts = [
        str(o.get("market", "")),
        str(o.get("name", "")),
        str(o.get("label", "")),
        str(o.get("type", "")),
        str(o.get("stat", "")),
        str(o.get("category", "")),
        str(o.get("group", "")),
    ]
    return " ".join(parts).lower()

def normalize_player_key_from_label(label: str, hint: str = "") -> Optional[str]:
    """
    Accept:
      - 'Jalen Brunson (Points)'  (preferred)
      - 'Jalen Brunson' if hint indicates this entry is points
    Return snake_case: 'jalen_brunson'
    """
    if not label:
        return None

    s = label.strip()

    # Case 1: explicit suffix
    if s.endswith("(Points)"):
        s = s.replace("(Points)", "").strip()

    # Case 2: no suffix -> only accept if hint indicates it's points
    else:
        h = (hint or "").lower()
        if "points" not in h and "pts" not in h:
            return None

    s = s.lower().replace(" ", "_")
    s = "_".join(s.split())
    return s or None

def extract_player_points_props_from_markets(
    markets: List[Dict[str, Any]],
    debug_player_props: bool = False,
    debug_prefix: str = "",
) -> List[Dict[str, Any]]:
    """
    Returns a list of normalized prop rows:
      {"market_name": "...", "player": "...", "line": float, "over": float|None, "under": float|None}

    Handles both:
      A) Market named like "Player Props - Points"
      B) Market named "Player Props" with nested submarkets or flat odds
    """
    out: List[Dict[str, Any]] = []

    for market in _iter_market_dicts(markets):
        market_name_raw = str(market.get("name", "")).strip()
        market_name_lc = market_name_raw.lower()

        if "player props" not in market_name_lc:
            continue

        # Examples: "Player Props - Points", "Player Props: Points", etc.
        if _looks_like_points_market(market_name_raw):
            odds_list = market.get("odds") or []
            for o in _iter_market_dicts(odds_list):
                player_label = str(o.get("label", "")).strip()
                player_key = normalize_player_key_from_label(player_label, hint=market_name_raw)

                if not player_key:
                    continue  # skip non-(Points) like (Pts+Rebs), etc.

                line = safe_float(o.get("hdp"))
                if line is None:
                    continue

                out.append({
                    "market_name": market_name_raw,
                    "player_label": player_label,     # keep original display label
                    "player_key": player_key,         # snake_case key
                    "line": line,
                    "over": safe_float(o.get("over")),
                    "under": safe_float(o.get("under")),
                })
            continue

        # Look for nested submarkets first
        nested = (
            market.get("markets")
            or market.get("submarkets")
            or market.get("props")
            or []
        )

        if debug_player_props:
            keys = list(market.keys())
            odds_len = len(market.get("odds") or []) if isinstance(market.get("odds"), list) else 0
            mk_len = len(market.get("markets") or []) if isinstance(market.get("markets"), list) else 0
            sm_len = len(market.get("submarkets") or []) if isinstance(market.get("submarkets"), list) else 0
            pr_len = len(market.get("props") or []) if isinstance(market.get("props"), list) else 0
            print(f"{debug_prefix}Player Props market keys={keys} | odds={odds_len} markets={mk_len} submarkets={sm_len} props={pr_len}")

        # Nested lists of dicts with their own name+odds
        if isinstance(nested, list) and nested:
            for sub in _iter_market_dicts(nested):
                sub_name_raw = str(sub.get("name", "")).strip()
                if not _looks_like_points_market(sub_name_raw):
                    continue
                odds_list = sub.get("odds") or []
                for o in _iter_market_dicts(odds_list):
                    player_label = str(o.get("label", "")).strip()
                    player_key = normalize_player_key_from_label(player_label, hint=market_name_raw)

                    if not player_key:
                        continue  # skip non-(Points) like (Pts+Rebs), etc.

                    line = safe_float(o.get("hdp"))
                    if line is None:
                        continue

                    out.append({
                        "market_name": market_name_raw,
                        "player_label": player_label,     # keep original display label
                        "player_key": player_key,         # snake_case key
                        "line": line,
                        "over": safe_float(o.get("over")),
                        "under": safe_float(o.get("under")),
                    })
            continue

        # Fallback: odds might be flat inside "Player Props" but include hints
        odds_list = market.get("odds") or []
        if isinstance(odds_list, list) and odds_list:
            for o in _iter_market_dicts(odds_list):
                hint = _odds_entry_hint(o)
                if ("points" not in hint) and ("pts" not in hint):
                    continue
                player_label = str(o.get("label", "")).strip()
                player_key = normalize_player_key_from_label(player_label, hint=market_name_raw)

                if not player_key:
                    continue  # skip non-(Points) like (Pts+Rebs), etc.

                line = safe_float(o.get("hdp"))
                if line is None:
                    continue

                out.append({
                    "market_name": market_name_raw,
                    "player_label": player_label,     # keep original display label
                    "player_key": player_key,         # snake_case key
                    "line": line,
                    "over": safe_float(o.get("over")),
                    "under": safe_float(o.get("under")),
                })

    return out


class Command(BaseCommand):
    help = "Fetch player points props from Odds API for event(s) and optionally store in Postgres."

    def add_arguments(self, parser):
        parser.add_argument("--event-id", help="Odds API eventId (single event)")
        parser.add_argument("--today", action="store_true", help="Fetch ALL NBA events occurring today (local time)")
        parser.add_argument("--events-limit", type=int, default=25, help="Max number of today's events to process")

        parser.add_argument(
            "--bookmakers",
            default="DraftKings,FanDuel",
            help="Comma-separated list, e.g. DraftKings,FanDuel",
        )

        parser.add_argument(
            "--debug-markets",
            action="store_true",
            help="Print available market names per bookmaker (for the first processed event) and exit.",
        )

        parser.add_argument(
            "--debug-player-props",
            action="store_true",
            help="Print internal structure info for 'Player Props' market if no points props are found.",
        )

        parser.add_argument(
            "--store",
            action="store_true",
            help="Actually store results into Postgres (otherwise just prints a sample).",
        )

        parser.add_argument(
            "--require-game",
            action="store_true",
            default=True,
            help=(
                "When used with --store, require that a matching Game (by odds_event_id) already exists. "
                "This is the normal pipeline mode (predict.py creates games)."
            ),
        )

        parser.add_argument(
            "--allow-create-game",
            action="store_true",
            default=False,
            help=(
                "Debug mode: if --store and Game doesn't exist yet, create it from Odds API response. "
                "Use this only for one-off testing."
            ),
        )

        parser.add_argument(
            "--print-limit",
            type=int,
            default=25,
            help="Max number of prop lines to print (for sanity).",
        )

    def handle(self, *args, **opts):
        api_key = os.getenv("ODDS_API_KEY")
        if not api_key:
            raise CommandError("ODDS_API_KEY is not set. Put it in your .env.")

        bookmakers = str(opts["bookmakers"])
        store = bool(opts["store"])
        print_limit = int(opts["print_limit"])
        debug_markets = bool(opts["debug_markets"])
        debug_player_props = bool(opts["debug_player_props"])

        # Determine which event IDs to process
        event_ids: List[str] = []
        if opts["today"]:
            event_ids = fetch_today_nba_event_ids(api_key, limit=int(opts["events_limit"]))
            if not event_ids:
                self.stdout.write("No NBA events found for today's local date.")
                return
        elif opts.get("event_id"):
            event_ids = [str(opts["event_id"])]
        else:
            raise CommandError("Provide either --event-id <id> OR --today")

        created = 0
        updated = 0
        printed = 0
        games_processed = 0
        total_found_props = 0

        for event_id in event_ids:
            resp = requests.get(
                ODDS_URL,
                params={"apiKey": api_key, "eventId": event_id, "bookmakers": bookmakers},
                timeout=30,
            )
            if resp.status_code != 200:
                self.stdout.write(self.style.WARNING(
                    f"Event {event_id}: Odds API error {resp.status_code}: {resp.text[:200]}"
                ))
                continue

            data = resp.json()

            # Debug: show market names and exit
            if debug_markets:
                self.stdout.write(self.style.WARNING(
                    f"\nDEBUG markets for event {event_id}: {data.get('away')} @ {data.get('home')} | {data.get('date')}"
                ))
                bookmakers_obj = data.get("bookmakers") or {}
                for bookmaker_name, markets in bookmakers_obj.items():
                    if not isinstance(markets, list):
                        continue
                    market_names = []
                    for m in markets:
                        if isinstance(m, dict) and m.get("name"):
                            market_names.append(str(m["name"]))
                    if market_names:
                        self.stdout.write(f"- {bookmaker_name}:")
                        for mn in sorted(set(market_names)):
                            self.stdout.write(f"    • {mn}")
                    else:
                        self.stdout.write(f"- {bookmaker_name}: (no markets returned)")
                return

            home = data.get("home")
            away = data.get("away")
            game_start = parse_datetime(data.get("date")) if data.get("date") else None

            require_game = bool(opts.get("require_game", True))
            allow_create_game = bool(opts.get("allow_create_game", False))

            # Resolve Game FK (SportsbookLine needs it)
            game = None
            if store:
                # The Game should already exist in the DB (created by predict.py).
                game = Game.objects.filter(odds_event_id=int(event_id)).first()

                if game is None:
                    if require_game and not allow_create_game:
                        self.stdout.write(
                            self.style.WARNING(
                                f"Event {event_id}: Game not found in DB (predict.py should have created it). "
                                f"Skipping store for this event. (Use --allow-create-game to create in debug mode.)"
                            )
                        )
                        # We can still print props, but we cannot store SportsbookLine without Game
                        # So we continue to next event if store is requested.
                        continue

                    if allow_create_game:
                        # Debug mode: create Game from Odds API response
                        if not (event_id and game_start and home and away):
                            self.stdout.write(
                                self.style.WARNING(
                                    f"Event {event_id}: missing required fields (home/away/date). "
                                    f"Cannot create Game; skipping store."
                                )
                            )
                            continue

                        game, _ = Game.objects.get_or_create(
                            odds_event_id=int(event_id),
                            defaults={
                                "home_team": handle_team_name(str(home))[:50],
                                "away_team": handle_team_name(str(away))[:50],
                                "start_time_utc": data.get("date"),
                            },
                        )

            bookmakers_obj = data.get("bookmakers") or {}
            if not isinstance(bookmakers_obj, dict):
                self.stdout.write(self.style.WARNING(f"Event {event_id}: unexpected 'bookmakers' type"))
                continue

            games_processed += 1

            for bookmaker_name, markets in bookmakers_obj.items():
                if not isinstance(markets, list):
                    continue

                props = extract_player_points_props_from_markets(
                    markets,
                    debug_player_props=debug_player_props and printed == 0,
                    debug_prefix=f"{bookmaker_name} | ",
                )

                if not props:
                    continue

                total_found_props += len(props)

                for p in props:
                    player_label = p["player_label"]

                    line = p["line"]
                    over_price = p.get("over")
                    under_price = p.get("under")
                    market_name = p.get("market_name") or "Player Props - Points"

                    if printed < print_limit:
                        self.stdout.write(
                            f"{event_id} | {away} @ {home} | {data.get('date')} | "
                            f"{bookmaker_name} | {player_label} | line={line} | over={over_price} under={under_price} | market={market_name}"
                        )
                        printed += 1
                    
                    if store and game:
                        player_name_clean = player_label.replace("(Points)", "").strip()

                        obj, was_created = SportsbookLine.objects.update_or_create(
                            game=game,
                            bookmaker=str(bookmaker_name),
                            player_name=player_name_clean,
                            stat="points",
                            defaults={
                                "line": float(line),
                                "over_price": over_price,
                                "under_price": under_price,
                            },
                        )

                        if was_created:
                            created += 1
                        else:
                            updated += 1

        if store:
            self.stdout.write(
                self.style.SUCCESS(
                    f"Processed {games_processed} games. "
                    f"Created {created} lines, updated {updated} lines."
                )
            )
        else:
            self.stdout.write(self.style.SUCCESS(
                f"Processed {games_processed} games. Printed {printed} lines. (total extracted props={total_found_props})"
            ))
            self.stdout.write("Run again with --store to save into Postgres.")
