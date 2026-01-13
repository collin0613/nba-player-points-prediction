# core/management/commands/compare_predictions_to_props.py
# Compare model predictions (from the CSV) to Odds API player points prop lines and print diffs. Call this after running src/predict.py.

import os
import csv
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple
import requests
from django.core.management.base import BaseCommand, CommandError
from core.management.commands.fetch_odds import extract_player_points_props_from_markets
from datetime import datetime
from zoneinfo import ZoneInfo
from django.utils.dateparse import parse_datetime

ODDS_URL = "https://api2.odds-api.io/v3/odds"

LOCAL_TZ = ZoneInfo("America/New_York")

def format_start_est(start_utc: str) -> str:
    """
    '2026-01-12T00:00:00Z' -> '2026-01-11 07:00 PM ET'
    Falls back to original string if parsing fails.
    """
    if not start_utc:
        return ""

    dt = parse_datetime(start_utc)
    if dt is None:
        # fallback for strings like ...Z if parse_datetime ever returns None
        try:
            dt = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
        except Exception:
            return start_utc

    # Ensure timezone-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))

    dt_local = dt.astimezone(LOCAL_TZ)
    return dt_local.strftime("%Y-%m-%d %I:%M %p ET")


def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pred_key_from_snake(s: str) -> str:
    # already in snake_case in CSV
    return (s or "").strip().lower()


def odds_player_key_from_label(label: str) -> Optional[str]:
    """
    'Jalen Brunson (Points)' -> 'jalen_brunson'
    Only accept literal '(Points)' entries (strict points-only props).
    """
    if not label:
        return None
    t = label.strip()
    if not t.endswith("(Points)"):
        return None
    t = t.replace("(Points)", "").strip()
    t = t.lower().replace(" ", "_")
    t = "_".join(t.split())
    return t or None


def fetch_points_props_for_event(api_key: str, event_id: str, bookmakers: str) -> Dict[str, Dict[str, Any]]:
    r = requests.get(
        ODDS_URL,
        params={"apiKey": api_key, "eventId": event_id, "bookmakers": bookmakers},
        timeout=30,
    )
    if r.status_code != 200:
        raise CommandError(f"Odds API error {r.status_code} for event {event_id}: {r.text[:300]}")

    data = r.json()
    bookmakers_obj = data.get("bookmakers") or {}
    if not isinstance(bookmakers_obj, dict):
        return {}

    out: Dict[str, Dict[str, Any]] = {}

    for bookmaker_name, markets in bookmakers_obj.items():
        if not isinstance(markets, list):
            continue

        props = extract_player_points_props_from_markets(markets)

        for p in props:
            player_key = p.get("player_key")
            if not player_key:
                continue

            # keep first bookmaker we find for each player
            if player_key in out:
                continue

            player_label = str(p.get("player_label") or "").strip()
            # strip "(Points)" for pretty printing
            if player_label.endswith("(Points)"):
                player_label = player_label.replace("(Points)", "").strip()

            out[player_key] = {
                "player_label": player_label or player_key,
                "line": float(p["line"]),
                "over": p.get("over"),
                "under": p.get("under"),
                "bookmaker": str(bookmaker_name),
                "market": str(p.get("market_name") or "Player Props"),
            }

    return out



class Command(BaseCommand):
    help = "Compare model predictions (CSV) to Odds API player points prop lines and print diffs."

    def add_arguments(self, parser):
        parser.add_argument("--csv", required=True, help="Path to predictions CSV (from predict.py)")
        parser.add_argument(
            "--bookmakers",
            default="DraftKings,FanDuel",
            help="Comma-separated list, e.g. DraftKings,FanDuel",
        )
        parser.add_argument(
            "--max-events",
            type=int,
            default=50,
            help="Safety limit on how many distinct events to query",
        )

    def handle(self, *args, **opts):
        api_key = os.getenv("ODDS_API_KEY")
        if not api_key:
            raise CommandError("ODDS_API_KEY is not set. Put it in your .env.")

        csv_path = str(opts["csv"])
        bookmakers = str(opts["bookmakers"])
        max_events = int(opts["max_events"])

        # Build: event_id -> list of predictions for that game
        # Each prediction: {"player_key", "player_name_raw", "predicted_pts", "team", "opponent", "start_utc"}
        by_event: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        event_meta: Dict[str, Dict[str, str]] = {}

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"player_name", "predicted_pts", "odds_event_id", "next_game_start_utc", "team", "opponent"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise CommandError(f"CSV missing required columns: {sorted(missing)}")

            for row in reader:
                event_id = (row.get("odds_event_id") or "").strip()
                if not event_id:
                    continue

                player_name_raw = (row.get("player_name") or "").strip()
                player_key = pred_key_from_snake(player_name_raw)
                pred_pts = safe_float(row.get("predicted_pts"))
                if not player_key or pred_pts is None:
                    continue

                start_utc = (row.get("next_game_start_utc") or "").strip()
                team = (row.get("team") or "").strip()
                opp = (row.get("opponent") or "").strip()

                by_event[event_id].append({
                    "player_key": player_key,
                    "player_name_raw": player_name_raw,
                    "predicted_pts": float(pred_pts),
                    "team": team,
                    "opponent": opp,
                })

                if event_id not in event_meta:
                    event_meta[event_id] = {
                        "start_utc": start_utc,
                        "matchup_hint": f"{team} vs {opp}",
                    }

        if not by_event:
            self.stdout.write("No predictions with odds_event_id found in CSV.")
            return

        # Safety limit
        event_ids = list(by_event.keys())[:max_events]

        for event_id in event_ids:
            preds = by_event[event_id]
            meta = event_meta.get(event_id, {})
            start_utc = meta.get("start_utc", "")
            # Header
            start_est = format_start_est(start_utc)
            self.stdout.write(f"\n{event_id} | {meta.get('matchup_hint','')} | {start_est}")

            props = fetch_points_props_for_event(api_key, event_id, bookmakers)

            if not props:
                self.stdout.write("  (No player points props returned for this event.)")
                continue

            # Print only predicted players
            any_printed = False
            for p in preds:
                pk = p["player_key"]
                if pk not in props:
                    continue
                any_printed = True
                line = props[pk]["line"]
                book = props[pk]["bookmaker"]
                player_display = props[pk]["player_label"]
                model_pred = p["predicted_pts"]
                diff = model_pred - line

                self.stdout.write(
                    f"- {player_display}: model={model_pred:.1f}, {book}_line={line:.1f}, diff={diff:+.1f}"
                )

            if not any_printed:
                self.stdout.write("  (Props returned, but none matched your predicted player set.)")
