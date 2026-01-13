# core/management/commands/update_actual_points.py
# Backfill ModelPrediction.actual_points for finished games using NBA boxscore data.

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Dict
import pandas as pd
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone
from nba_api.stats.endpoints import scoreboardv2, boxscoretraditionalv2
from nba_api.stats.static import teams
from core.models import ModelPrediction

LOCAL_TZ = ZoneInfo("America/New_York")
TEAM_ID_TO_ABBREV = {t["id"]: t["abbreviation"] for t in teams.get_teams()}
TEAM_ABBREV_TO_ID = {t["abbreviation"]: t["id"] for t in teams.get_teams()}
SLEEP_BETWEEN_CALLS_SEC = 0.6

def _sleep():
    time.sleep(SLEEP_BETWEEN_CALLS_SEC)


def ny_local_date_from_utc(dt_utc) -> Optional[datetime.date]:
    if not dt_utc:
        return None
    return dt_utc.astimezone(LOCAL_TZ).date()


def mmddyyyy(d: datetime.date) -> str:
    return d.strftime("%m/%d/%Y")


def find_nba_game_id_for_matchup(local_date: datetime.date, home_abbrev: str, away_abbrev: str) -> Optional[str]:
    """
    Uses NBA ScoreboardV2 for the local_date (NY) to find GAME_ID for the matchup.
    """
    home_id = TEAM_ABBREV_TO_ID.get(home_abbrev)
    away_id = TEAM_ABBREV_TO_ID.get(away_abbrev)
    if not home_id or not away_id:
        return None

    _sleep()
    sb = scoreboardv2.ScoreboardV2(game_date=mmddyyyy(local_date))
    dfs = sb.get_data_frames()
    if not dfs or dfs[0].empty:
        return None

    gh = dfs[0]  # GameHeader
    # Expected columns include GAME_ID, HOME_TEAM_ID, VISITOR_TEAM_ID
    for _, row in gh.iterrows():
        try:
            hid = int(row.get("HOME_TEAM_ID"))
            vid = int(row.get("VISITOR_TEAM_ID"))
        except Exception:
            continue

        if hid == home_id and vid == away_id:
            return str(row.get("GAME_ID"))

    return None


def get_player_pts_from_boxscore(game_id: str, player_id: int) -> Optional[float]:
    """
    Reads boxscore and returns PTS for player_id, or None if not found / not final yet.
    """
    _sleep()
    bs = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
    dfs = bs.get_data_frames()
    if not dfs or dfs[0].empty:
        return None

    ps = dfs[0]  # PlayerStats
    # PLAYER_ID, PTS
    row = ps[ps["PLAYER_ID"] == int(player_id)]
    if row.empty:
        return None
    pts = row.iloc[0].get("PTS")
    try:
        return float(pts)
    except Exception:
        return None


class Command(BaseCommand):
    help = "Backfill ModelPrediction.actual_points for finished games using NBA boxscore."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=10, help="Look back this many days (by game start time).")
        parser.add_argument("--dry-run", action="store_true", help="Print what would update, but do not write to DB.")
        parser.add_argument("--verbose", action="store_true", help="Print per-row skip reasons.")
        parser.add_argument("--buffer-hours", type=int, default=3, help="Treat games as finished only after this buffer.")

    def handle(self, *args, **opts):
        days = int(opts["days"])
        dry_run = bool(opts["dry_run"])
        verbose = bool(opts["verbose"])
        buffer_hours = int(opts["buffer_hours"])

        now = timezone.now()
        window_start = now - timedelta(days=days)
        finished_cutoff = now - timedelta(hours=buffer_hours)

        qs = (
            ModelPrediction.objects
            .select_related("game")
            .filter(actual_points__isnull=True)
            .filter(game__start_time_utc__gte=window_start)
            .filter(game__start_time_utc__lte=finished_cutoff)
            .order_by("game__start_time_utc")
        )

        scanned = 0
        updated = 0
        skipped = 0

        for pred in qs:
            scanned += 1
            g = pred.game

            local_date = ny_local_date_from_utc(g.start_time_utc)
            if not local_date:
                skipped += 1
                if verbose:
                    self.stdout.write(f"SKIP {pred.player_name}: no local_date for game {g.id}")
                continue

            if not g.home_team or not g.away_team:
                skipped += 1
                if verbose:
                    self.stdout.write(f"SKIP {pred.player_name}: missing home/away abbrev on Game {g.id}")
                continue

            nba_game_id = find_nba_game_id_for_matchup(local_date, g.home_team, g.away_team)
            if not nba_game_id:
                skipped += 1
                if verbose:
                    self.stdout.write(
                        f"SKIP {pred.player_name}: could not find GAME_ID on {local_date} for {g.away_team}@{g.home_team}"
                    )
                continue

            pts = get_player_pts_from_boxscore(nba_game_id, pred.player_id)
            if pts is None:
                skipped += 1
                if verbose:
                    self.stdout.write(
                        f"SKIP {pred.player_name}: boxscore missing player_id={pred.player_id} for GAME_ID={nba_game_id}"
                    )
                continue

            if verbose or dry_run:
                self.stdout.write(
                    f"{'DRY' if dry_run else 'UPD'} {pred.player_name}: "
                    f"{g.away_team}@{g.home_team} {g.start_time_utc} -> actual_points={pts}"
                )

            if not dry_run:
                with transaction.atomic():
                    pred.actual_points = pts
                    pred.save(update_fields=["actual_points"])
                updated += 1
            else:
                updated += 1

        self.stdout.write(self.style.SUCCESS(f"Done. updated={updated}, skipped={skipped}, scanned={scanned}"))
