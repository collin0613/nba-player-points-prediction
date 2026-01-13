import os
import requests
from datetime import datetime, timedelta, timezone
from django.core.management.base import BaseCommand, CommandError

EVENTS_URL = "https://api2.odds-api.io/v3/events"

def parse_iso_z(dt_str: str):
    """Parse ISO strings like 2026-01-04T19:00:00Z into aware UTC datetime."""
    if not dt_str:
        return None
    # Odds API uses Z for UTC
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except ValueError:
        return None


class Command(BaseCommand):
    help = "Fetch NBA events from Odds-API.io with a deterministic time window."

    def add_arguments(self, parser):
        parser.add_argument("--sport", default="basketball", help="Sport slug (use 'basketball')")
        parser.add_argument("--limit", type=int, default=20, help="Max events to print (after filtering)")

        # Time filtering controls
        parser.add_argument(
            "--days",
            type=int,
            default=2,
            help="Window size in days from 'now' (default: 2).",
        )
        parser.add_argument(
            "--from",
            dest="from_mode",
            choices=["now", "today_utc"],
            default="today_utc",
            help="Start window from 'now' or from today's 00:00 UTC.",
        )
        parser.add_argument(
            "--only",
            choices=["any", "upcoming", "past"],
            default="upcoming",
            help="Filter to upcoming only (recommended), past only, or any within window.",
        )

    def handle(self, *args, **opts):
        api_key = os.getenv("ODDS_API_KEY")
        if not api_key:
            raise CommandError("ODDS_API_KEY is not set (put it in your .env).")

        sport = opts["sport"]
        limit = opts["limit"]
        days = opts["days"]
        from_mode = opts["from_mode"]
        only = opts["only"]

        # Define the time window in UTC
        now_utc = datetime.now(timezone.utc)
        if from_mode == "today_utc":
            window_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            window_start = now_utc

        window_end = window_start + timedelta(days=days)

        # Raw fetch (docs-style)
        resp = requests.get(
            EVENTS_URL,
            params={"apiKey": api_key, "sport": sport},
            timeout=30,
        )
        if resp.status_code != 200:
            raise CommandError(f"Odds API error {resp.status_code}: {resp.text[:300]}")

        events = resp.json()
        if not isinstance(events, list):
            raise CommandError(f"Unexpected response shape: expected list, got {type(events)}")

        # Filter to NBA only
        nba = [e for e in events if (e.get("league") or {}).get("slug") == "usa-nba"]

        # Filter by window + our own "past/upcoming" logic
        selected = []
        for e in nba:
            dt = parse_iso_z(e.get("date"))
            if not dt:
                continue

            if not (window_start <= dt < window_end):
                continue

            if only == "upcoming" and dt < now_utc:
                continue
            if only == "past" and dt >= now_utc:
                continue

            selected.append((dt, e))

        # Sort by datetime so output is predictable
        selected.sort(key=lambda x: x[0])

        self.stdout.write(
            self.style.SUCCESS(
                f"Fetched {len(events)} '{sport}' events; NBA-only: {len(nba)}.\n"
                f"Window: {window_start.isoformat()} → {window_end.isoformat()} | now={now_utc.isoformat()}\n"
                f"Selected ({only}): {len(selected)}\n"
            )
        )

        for dt, e in selected[:limit]:
            league = e.get("league") or {}
            # Our own derived status
            derived = "upcoming" if dt >= now_utc else "past"
            self.stdout.write(
                f"- id={e.get('id')} | {e.get('away')} @ {e.get('home')} | {e.get('date')} "
                f"| api_status={e.get('status')} | derived={derived} | league={league.get('slug')}"
            )
