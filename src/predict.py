# src/predict.py
# Main file: run this to execute full workflow. Description below.
"""
predict.py

- Fetch 2025-26 (current season) game logs for a set of players
- Backfill any missing actual_points for finished games in Postgres
- Find each player's NEXT scheduled matchup (using NBA scoreboard)
- Build the exact feature columns the trained pipeline expects
- Load the saved .joblib model and predict next-game points for each player
- Print and save results to CSV and to Postgres
"""
from __future__ import annotations
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

import django
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Iterable, Optional, Tuple, List
import requests
from zoneinfo import ZoneInfo
import joblib
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import playergamelog, leaguegamelog, scoreboardv2
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.static import teams
from dotenv import load_dotenv
load_dotenv()  # loads .env into os.environ for this script
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tracker.settings")
django.setup()

from django.db import transaction
from core.models import Game, ModelPrediction, SportsbookLine
from core.management.commands.fetch_odds import extract_player_points_props_from_markets
from dataclasses import dataclass
from django.utils import timezone

# nba_api
from nba_api.stats.endpoints import playergamelog
from core.models import Game, ModelPrediction

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "models/ridge_points_model_2024-25.joblib"
OUTPUT_CSV = "predictions_next_games_2025-26.csv"
SEASON_PRED = "2025-26"  # season we are predicting within; update in future for current NBA season
SEASON_TYPE = "Regular Season"
MIN_THRESHOLD = 15
ROLL_5 = 5
ROLL_10 = 10
ODDS_EVENTS_URL = "https://api2.odds-api.io/v3/events"
LOCAL_TZ = ZoneInfo("America/New_York")
ODDS_LOOKAHEAD_DAYS = 14  # align with schedule lookahead
SCHEDULE_LOOKAHEAD_DAYS = 14 # Look ahead window for finding next scheduled game
TEAM_ABBREV_TO_FULL = {t["abbreviation"]: t["full_name"] for t in teams.get_teams()}
TEAM_FULL_TO_ABBREV = {t["full_name"]: t["abbreviation"] for t in teams.get_teams()}
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_URL = "https://api2.odds-api.io/v3/odds"
ODDS_BOOKMAKERS = "DraftKings" # "DraftKings,FanDuel"
SLEEP_BETWEEN_CALLS_SEC = 0.6 # Throttling / retries for nba_api rate limits
MAX_RETRIES = 3
RETRY_SLEEP_SEC = 1.5

# current player set - add all players that we get data from
PLAYERS = [
    {"player_name": "anthony_edwards", "player_id": 1630162},
    {"player_name": "cade_cunningham", "player_id": 1630595},
    {"player_name": "giannis_antetokounmpo", "player_id": 203507},
    {"player_name": "jalen_brunson", "player_id": 1628973},
    {"player_name": "lebron_james", "player_id": 2544},
    {"player_name": "shai_gilgeous-alexander", "player_id": 1628983},
    {"player_name": "stephen_curry", "player_id": 201939},
]

# current feature set (keep aligned with train_model.py)
NUM_FEATURES = [
    "is_home",
    "rest_days",
    "pts_avg_5",
    "pts_avg_10",
    "min_avg_5",
    "pts_allowed_avg_10",
    "pace_proxy_avg_10",
    "star_pts_allowed_avg_10",
]
CAT_FEATURES = ["player_name"]
TEAM_ID_TO_ABBREV = {t["id"]: t["abbreviation"] for t in teams.get_teams()}
TEAM_ABBREV_TO_ID = {t["abbreviation"]: t["id"] for t in teams.get_teams()}

# ==================================
# Helpers to backfill actual_points
# ==================================

@dataclass
class BackfillResult:
    checked_predictions: int
    updated_predictions: int
    skipped_no_gamelog: int
    skipped_no_match: int
    skipped_already_filled: int


def _normalize_matchup(s: str) -> str:
    return (s or "").upper().replace("VS.", "VS").replace("  ", " ").strip()

def _admin_like_dt_str(dt) -> str:
    month = dt.strftime("%b.")
    day = dt.day
    year = dt.year
    hour = dt.hour % 12 or 12
    ampm = "a.m." if dt.hour < 12 else "p.m."
    minute = dt.minute
    if minute == 0:
        return f"{month} {day}, {year}, {hour} {ampm}"
    return f"{month} {day}, {year}, {hour}:{minute:02d} {ampm}"


def _fetch_player_gamelog_df(player_id: int, season: str, season_type: str) -> pd.DataFrame:
    gl = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star=season_type,
    )
    df = gl.get_data_frames()[0].copy()
    if df.empty:
        return df
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    # Normalize matchup column for matching
    df["MATCHUP_NORM"] = df["MATCHUP"].astype(str).map(_normalize_matchup)
    return df


def _find_pts_for_game_from_gamelog(
    gamelog_df: pd.DataFrame,
    team_abbrev: str,
    opponent_abbrev: str,
    was_home: bool,
    utc_start_time,
    utc_window_hours: int = 12,
) -> Optional[float]:
    """
    We match a specific game by:
      1) opponent + home/away encoded in MATCHUP string
      2) GAME_DATE within a wide window around the game UTC start time (handles local-date vs UTC drift)
    """
    if gamelog_df.empty:
        return None

    # build expected matchup pattern
    if was_home:
        expected = _normalize_matchup(f"{team_abbrev} VS {opponent_abbrev}")
    else:
        expected = _normalize_matchup(f"{team_abbrev} @ {opponent_abbrev}")

    # filter by matchup first (strongest signal)
    df = gamelog_df[gamelog_df["MATCHUP_NORM"] == expected].copy()
    if df.empty:
        return None

    # then filter by date window around UTC start time
    start = pd.Timestamp(utc_start_time).tz_convert("UTC") - pd.Timedelta(hours=utc_window_hours)
    end = pd.Timestamp(utc_start_time).tz_convert("UTC") + pd.Timedelta(hours=utc_window_hours)

    # treat GAME_DATE as naive date marker; compare on date range to be resilient
    # We accept rows whose GAME_DATE.date is in [start.date-1, end.date+1]
    start_date = start.date() - timedelta(days=1)
    end_date = end.date() + timedelta(days=1)

    df = df[(df["GAME_DATE"].dt.date >= start_date) & (df["GAME_DATE"].dt.date <= end_date)]
    if df.empty:
        return None

    # If multiple (rare), take the one closest in date to utc_start_time (by absolute day diff)
    target_day = pd.Timestamp(utc_start_time).date()
    df["DAY_DIFF"] = (df["GAME_DATE"].dt.date.apply(lambda d: abs((d - target_day).days))).astype(int)
    df = df.sort_values(["DAY_DIFF", "GAME_DATE"]).reset_index(drop=True)

    pts = df.loc[0, "PTS"]
    try:
        return float(pts)
    except Exception:
        return None


def backfill_actual_points(
    *,
    season: str,
    season_type: str,
    model_version: Optional[str] = None,
    only_games_started_before_minutes: int = 30,
    max_games: int = 50,
    utc_window_hours: int = 12,
    verbose: bool = True,
) -> BackfillResult:
    """
    Fills ModelPrediction.actual_points for already-started games whose actual_points is NULL.

      - Find DB games with missing actuals (via predictions) and start_time_utc < now - buffer
      - For each game, load its predictions
      - For each player, pull their gamelog for the season and locate the row matching that game
        using MATCHUP (home/away + opponent) + a date window around the game's UTC tip time.
      - Update actual_points (locked in transaction)

    Note:
      - This does NOT rely on NBA GAME_DATE == UTC date.
      - Uses player_id so it won't break on name differences.
    """
    now_utc = timezone.now()
    cutoff = now_utc - timedelta(minutes=only_games_started_before_minutes)

    # Find games with at least one prediction missing actual_points
    games_qs = (
        Game.objects.filter(start_time_utc__lt=cutoff, predictions__actual_points__isnull=True)
        .distinct()
        .order_by("-start_time_utc")[:max_games]
    )

    # Cache gamelogs by player_id so we don't hammer nba_api repeatedly
    gamelog_cache: Dict[int, pd.DataFrame] = {}

    checked = 0
    updated = 0
    skipped_no_gamelog = 0
    skipped_no_match = 0
    skipped_already = 0

    for game in games_qs:
        # Derive opponent + home/away from the game's teams
        home = (game.home_team or "").upper().strip()
        away = (game.away_team or "").upper().strip()

        preds_qs = game.predictions.all()
        if model_version:
            preds_qs = preds_qs.filter(model_version=model_version)

        preds = list(preds_qs)
        if not preds:
            continue

        if verbose:
            print(
                f"\n[BACKFILL] Game {away} @ {home} | odds_event_id={game.odds_event_id} | "
                f"start_time_utc={game.start_time_utc.isoformat()} ({_admin_like_dt_str(game.start_time_utc)})"
            )

        for pred in preds:
            checked += 1

            if pred.actual_points is not None:
                skipped_already += 1
                continue

            player_id = int(pred.player_id)

            # load/cached gamelog
            if player_id not in gamelog_cache:
                try:
                    gamelog_cache[player_id] = _fetch_player_gamelog_df(player_id, season, season_type)
                except Exception:
                    gamelog_cache[player_id] = pd.DataFrame()

            gl_df = gamelog_cache[player_id]
            if gl_df.empty:
                skipped_no_gamelog += 1
                if verbose:
                    print(f"  - {pred.player_name} ({player_id}): no gamelog -> skip")
                continue

            # Determine player's team abbrev for matchup expectation
            try:
                last_matchup = str(gl_df.sort_values("GAME_DATE").iloc[-1]["MATCHUP_NORM"])
                team_abbrev = last_matchup.split(" ", 1)[0].strip()
            except Exception:
                team_abbrev = None

            if not team_abbrev:
                skipped_no_match += 1
                if verbose:
                    print(f"  - {pred.player_name} ({player_id}): couldn't infer team abbrev -> skip")
                continue

            # For this DB game, expected opponent + home flag depends on whether team_abbrev is home or away
            if team_abbrev == home:
                opponent = away
                was_home = True
            elif team_abbrev == away:
                opponent = home
                was_home = False
            else:
                # Player isn't on either team anymore (trade?) or inference wrong.
                skipped_no_match += 1
                if verbose:
                    print(
                        f"  - {pred.player_name} ({player_id}): inferred team={team_abbrev} not in {away}@{home} -> skip"
                    )
                continue

            pts = _find_pts_for_game_from_gamelog(
                gamelog_df=gl_df,
                team_abbrev=team_abbrev,
                opponent_abbrev=opponent,
                was_home=was_home,
                utc_start_time=game.start_time_utc,
                utc_window_hours=utc_window_hours,
            )

            if pts is None:
                skipped_no_match += 1
                if verbose:
                    print(f"  - {pred.player_name} ({player_id}): no matching gamelog row -> skip")
                continue

            # Update with locking to avoid races if ran multiple times
            try:
                with transaction.atomic():
                    locked = (
                        ModelPrediction.objects.select_for_update()
                        .get(game=game, player_id=player_id)
                    )
                    if locked.actual_points is None:
                        locked.actual_points = float(pts)
                        locked.save(update_fields=["actual_points"])
                        updated += 1
                        if verbose:
                            print(f"  ~ {pred.player_name} ({player_id}): actual_points={pts}")
            except Exception as e:
                if verbose:
                    print(f"  - {pred.player_name} ({player_id}): update failed: {e}")

    return BackfillResult(
        checked_predictions=checked,
        updated_predictions=updated,
        skipped_no_gamelog=skipped_no_gamelog,
        skipped_no_match=skipped_no_match,
        skipped_already_filled=skipped_already,
    )

# -----------------------------
# Helpers: safe nba_api calls
# -----------------------------
def _sleep():
    time.sleep(SLEEP_BETWEEN_CALLS_SEC)

def call_with_retries(fn, *args, **kwargs):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _sleep()
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SEC * attempt)
            else:
                raise last_err

# -----------------------------
# Team context (defense + pace)
# -----------------------------
def build_team_context_rolling(season: str) -> pd.DataFrame:
    """
    Builds team context from played games so far:
      - pts_allowed_avg_10: rolling avg points allowed (last 10)
      - pace_proxy_avg_10: rolling avg of game total points (last 10)

    Returns dataframe with columns:
      TEAM_ABBREVIATION, GAME_DATE, pts_allowed_avg_10, pace_proxy_avg_10
    """
    lg = call_with_retries(
        leaguegamelog.LeagueGameLog,
        season=season,
        season_type_all_star=SEASON_TYPE,
    )
    df = lg.get_data_frames()[0].copy()

    # Expected columns: GAME_ID, GAME_DATE, TEAM_ABBREVIATION, PTS
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    # Total points in each game (sum of both teams)
    game_totals = df.groupby("GAME_ID")["PTS"].sum().reset_index(name="game_total_pts")
    df = df.merge(game_totals, on="GAME_ID", how="left")

    # Points allowed: total_pts - team_pts
    total_pts = df.groupby("GAME_ID")["PTS"].transform("sum")
    df["PTS_ALLOWED"] = total_pts - df["PTS"]

    # Rolling by team (leakage-safe: shift(1))
    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
    df["pts_allowed_avg_10"] = (
        df.groupby("TEAM_ABBREVIATION")["PTS_ALLOWED"].shift(1).rolling(10).mean()
    )
    df["pace_proxy_avg_10"] = (
        df.groupby("TEAM_ABBREVIATION")["game_total_pts"].shift(1).rolling(10).mean()
    )
    ctx = df[["TEAM_ABBREVIATION", "GAME_DATE", "pts_allowed_avg_10", "pace_proxy_avg_10"]].copy()
    return ctx


def latest_team_context(ctx: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    For each team, grab the latest non-null values for context features.
    Returns dict: { "BOS": {"pts_allowed_avg_10": x, "pace_proxy_avg_10": y}, ... }

    If a team doesn't have enough history, values may be NaN; caller should fill defaults.
    """
    out: Dict[str, Dict[str, float]] = {}
    ctx = ctx.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    for team, g in ctx.groupby("TEAM_ABBREVIATION"):
        row = g.dropna(subset=["pts_allowed_avg_10", "pace_proxy_avg_10"], how="any").tail(1)
        if len(row) == 0:
            out[team] = {"pts_allowed_avg_10": np.nan, "pace_proxy_avg_10": np.nan}
        else:
            r = row.iloc[0]
            out[team] = {
                "pts_allowed_avg_10": float(r["pts_allowed_avg_10"]),
                "pace_proxy_avg_10": float(r["pace_proxy_avg_10"]),
            }
    return out

# -----------------------------
# Player features from 2025-26 logs
# -----------------------------
def parse_team_from_matchup(matchup: str) -> str:
    # MATCHUP looks like "NYK vs. CLE" or "NYK @ BOS"
    return matchup.split(" ")[0]


def rolling_mean_last_n(series: pd.Series, n: int) -> Optional[float]:
    vals = series.dropna().to_list()
    if len(vals) < n:
        return None
    return float(np.mean(vals[-n:]))


def compute_player_recent_features(player_id: int) -> Tuple[pd.DataFrame, Optional[Dict[str, float]]]:
    """
    Fetch player 2025-26 gamelog and compute the features needed for their NEXT GAME to make a prediction.
    Returns:
      - full cleaned gamelog df
      - dict of recent rolling features (pts_avg_5, pts_avg_10, min_avg_5) or None if not enough data
    """
    gl = call_with_retries(
        playergamelog.PlayerGameLog,
        player_id=player_id,
        season=SEASON_PRED,
        season_type_all_star=SEASON_TYPE,
    )
    df = gl.get_data_frames()[0].copy()
    if df.empty:
        return df, None

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    df = df.dropna(subset=["PTS", "MIN", "MATCHUP"])
    df = df[df["MIN"] >= MIN_THRESHOLD].copy()

    if df.empty:
        return df, None

    pts_avg_5 = rolling_mean_last_n(df["PTS"], ROLL_5)
    pts_avg_10 = rolling_mean_last_n(df["PTS"], ROLL_10)
    min_avg_5 = rolling_mean_last_n(df["MIN"], ROLL_5)

    # Need enough history for the features the model expects
    if pts_avg_5 is None or pts_avg_10 is None or min_avg_5 is None:
        return df, None

    recent = {
        "pts_avg_5": pts_avg_5,
        "pts_avg_10": pts_avg_10,
        "min_avg_5": min_avg_5,
    }
    return df, recent


# -----------------------------
# Find next scheduled game for a team
# -----------------------------
def normalize_scoreboard_date(d: datetime) -> str:
    return d.strftime("%m/%d/%Y")

def find_next_game_for_team(team_abbrev: str) -> Optional[Tuple[datetime, str, int]]:
    """
    Returns (game_date, opponent_abbrev, is_home) for the next scheduled game
    within SCHEDULE_LOOKAHEAD_DAYS, using ScoreboardV2.
      - Scoreboard reliably includes HOME_TEAM_ID and VISITOR_TEAM_ID
      - Abbreviation columns may not exist, so we map IDs -> abbreviations ourselves.
    """
    if team_abbrev not in TEAM_ABBREV_TO_ID:
        raise ValueError(f"Unknown team abbrev: {team_abbrev}")

    target_team_id = TEAM_ABBREV_TO_ID[team_abbrev]

    start = datetime.now()
    for i in range(SCHEDULE_LOOKAHEAD_DAYS + 1):
        day = start + timedelta(days=i)
        date_str = normalize_scoreboard_date(day)
        sb = call_with_retries(scoreboardv2.ScoreboardV2, game_date=date_str)

        dfs = sb.get_data_frames()
        if not dfs:
            continue

        games = dfs[0]
        if games is None or games.empty:
            continue

        # Expected columns in GameHeader include HOME_TEAM_ID and VISITOR_TEAM_ID (or VISITOR_TEAM_ID)
        for _, row in games.iterrows():
            home_id = row.get("HOME_TEAM_ID")
            away_id = row.get("VISITOR_TEAM_ID", row.get("AWAY_TEAM_ID"))

            if pd.isna(home_id) or pd.isna(away_id):
                continue

            home_id = int(home_id)
            away_id = int(away_id)

            if home_id == target_team_id:
                opp_abbrev = TEAM_ID_TO_ABBREV.get(away_id)
                if opp_abbrev:
                    return (day, opp_abbrev, 1)

            if away_id == target_team_id:
                opp_abbrev = TEAM_ID_TO_ABBREV.get(home_id)
                if opp_abbrev:
                    return (day, opp_abbrev, 0)

    return None


def compute_rest_days(last_game_date: pd.Timestamp, next_game_date: datetime) -> float:
    days = (pd.Timestamp(next_game_date.date()) - last_game_date.normalize()).days
    if days < 0:
        days = 0
    # match your training behavior: cap at 5, fallback usually 3 for unknown
    return float(min(days, 5))


# Compute the UTC date/time key for Odds API mapping
def normalize_team_name(name: str) -> str:
    """
    Normalize team names to improve matching between Odds API and nba_api.
    Odds sometimes uses 'LA Clippers' / 'LA Lakers' while nba_api uses 'Los Angeles ...'
    """
    s = (name or "").strip().lower()

    # unify LA naming (only case where two teams share a city name)
    s = s.replace("la clippers", "los angeles clippers")
    s = s.replace("la lakers", "los angeles lakers")
    s = s.replace("-", " ")
    s = " ".join(s.split())
    return s


def local_date_from_utc_z(utc_z: str) -> Optional[str]:
    """
    Convert '2026-01-06T00:00:00Z' -> 'YYYY-MM-DD' in America/New_York.
    """
    if not utc_z:
        return None
    try:
        dt_utc = datetime.fromisoformat(utc_z.replace("Z", "+00:00"))
        dt_local = dt_utc.astimezone(LOCAL_TZ)
        return str(dt_local.date())
    except Exception:
        return None


def build_odds_nba_event_map(days: int = ODDS_LOOKAHEAD_DAYS) -> Dict[tuple, Dict[str, str]]:
    """
    One API call. Returns mapping:
      key = (local_game_date, away_abbrev, home_abbrev)
      val = {"event_id": "...", "start_utc": "2026-01-06T00:00:00Z", "status": "..."}
    """
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        # predict.py can still run with an empty map if no key
        return {}

    resp = requests.get(
        ODDS_EVENTS_URL,
        params={"apiKey": api_key, "sport": "basketball"},
        timeout=30,
    )
    resp.raise_for_status()
    events = resp.json()
    if not isinstance(events, list):
        return {}

    # Filter strictly to NBA
    nba_events = [e for e in events if (e.get("league") or {}).get("slug") == "usa-nba"]

    # Time window in LOCAL time (today -> today+days)
    now_local = datetime.now(LOCAL_TZ)
    window_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    window_end = window_start + timedelta(days=days)

    # Build mapping
    out: Dict[tuple, Dict[str, str]] = {}

    for e in nba_events:
        start_utc = e.get("date")  # ISO Z string
        event_id = str(e.get("id"))
        home_name = e.get("home")
        away_name = e.get("away")

        if not (start_utc and home_name and away_name and event_id):
            continue

        # Convert start time to local datetime for windowing + local date keying
        try:
            dt_utc = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
            dt_local = dt_utc.astimezone(LOCAL_TZ)
        except Exception:
            continue

        if not (window_start <= dt_local < window_end):
            continue

        local_date = str(dt_local.date())

        # Convert Odds team names -> abbreviations using normalized full names
        home_norm = normalize_team_name(home_name)
        away_norm = normalize_team_name(away_name)

        # Try direct full-name mapping after normalization
        home_abbrev = None
        away_abbrev = None

        # Build a normalized full_name -> abbrev lookup once per call
        norm_full_to_abbrev = {normalize_team_name(k): v for k, v in TEAM_FULL_TO_ABBREV.items()}
        home_abbrev = norm_full_to_abbrev.get(home_norm)
        away_abbrev = norm_full_to_abbrev.get(away_norm)
        if not home_abbrev or not away_abbrev:
            continue

        key = (local_date, away_abbrev, home_abbrev)
        out[key] = {
            "event_id": event_id,
            "start_utc": start_utc,
            "status": str(e.get("status", "")),
        }

    return out


def odds_key_for_matchup(local_date: str, team_abbrev: str, opp_abbrev: str, is_home: int) -> tuple:
    """
    Create the key that matches build_odds_nba_event_map().
    key = (local_date, away_abbrev, home_abbrev)
    """
    if is_home == 1:
        # team is home, opponent is away
        return (local_date, opp_abbrev, team_abbrev)
    else:
        # team is away, opponent is home
        return (local_date, team_abbrev, opp_abbrev)


# -----------------------------
# Helpers fetch the player points prop lines from Odds API
# -----------------------------
def _tracked_player_keys() -> set[str]:
    return {p["player_name"] for p in PLAYERS}

def game_already_has_any_points_lines(game: Game, bookmakers: list[str]) -> bool:
    """
    Skip Odds API calls for games we've already populated (any lines at all).
    """
    return SportsbookLine.objects.filter(
        game=game,
        stat="points",
        bookmaker__in=bookmakers,
    ).exists()

def fetch_and_store_points_props_for_game(
    *,
    game: Game,
    bookmakers_csv: str,
    only_player_keys: set[str],
    print_sample_limit: int = 0,
) -> tuple[int, int]:
    """
    Fetch Odds API odds for a single event_id and store ONLY tracked players' Points lines.
    """
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY is not set in .env")

    resp = requests.get(
        ODDS_URL,
        params={"apiKey": ODDS_API_KEY, "eventId": str(game.odds_event_id), "bookmakers": bookmakers_csv},
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"[WARN] Odds API {resp.status_code} for event {game.odds_event_id}: {resp.text[:200]}")
        return (0, 0)

    data = resp.json()
    bookmakers_obj = data.get("bookmakers") or {}
    if not isinstance(bookmakers_obj, dict):
        print(f"[WARN] Unexpected bookmakers shape for event {game.odds_event_id}")
        return (0, 0)

    created = 0
    updated = 0
    printed = 0

    for bookmaker_name, markets in bookmakers_obj.items():
        if not isinstance(markets, list):
            continue

        props = extract_player_points_props_from_markets(markets)
        if not props:
            continue

        for p in props:
            player_key = p.get("player_key")  # snake_case, e.g. 'jalen_brunson'
            if not player_key or player_key not in only_player_keys:
                continue  # only store props for the set of tracked players

            player_label = str(p.get("player_label") or "").strip()  # e.g. "Jalen Brunson (Points)"
            player_name_clean = player_label.replace("(Points)", "").strip()

            line = p.get("line")
            if line is None:
                continue

            over_price = p.get("over")
            under_price = p.get("under")

            if printed < print_sample_limit:
                print(
                    f"{game.odds_event_id} | {game.away_team} @ {game.home_team} | "
                    f"{game.start_time_utc} | {bookmaker_name} | {player_label} | "
                    f"line={line} | over={over_price} under={under_price}"
                )
                printed += 1

            obj, was_created = SportsbookLine.objects.update_or_create(
                game=game,
                bookmaker=str(bookmaker_name),
                player_name=player_name_clean,
                stat="points",
                defaults={
                    "line": float(line),
                    "over_price": float(over_price) if over_price is not None else None,
                    "under_price": float(under_price) if under_price is not None else None,
                },
            )
            if was_created:
                created += 1
            else:
                updated += 1

    return (created, updated)

# -----------------------------
# Main prediction flow
# -----------------------------
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    pipeline = joblib.load(MODEL_PATH)

    # Before we generate/store new predictions (or right after loading the model), we backfill the actual_points for any already-played games missing them.
    # When calling predict.py every day, we keep filling in actuals for past games predicted and actively predicting the next game in one executed file.
    res = backfill_actual_points(
        season=SEASON_PRED,          # e.g. "2025-26"
        season_type=SEASON_TYPE,     # e.g. "Regular Season"
        model_version=None,          
        verbose=True,
    )
    print("\n[BACKFILL SUMMARY]", res)

    # Proceed to next-game predictions
    print(f"Loaded model pipeline from: {MODEL_PATH}")
    print(f"Predicting next game points for season: {SEASON_PRED}")
    print("Building team context (defense + pace) from league games played so far...")
    ctx = build_team_context_rolling(SEASON_PRED)
    team_ctx_latest = latest_team_context(ctx)
    print("Fetching Odds API NBA events to map UTC start times...")
    odds_map = build_odds_nba_event_map(days=ODDS_LOOKAHEAD_DAYS)
    print(f"Odds map entries (NBA games in window): {len(odds_map)}")

    # League-average fallbacks (in case early season has NaNs)
    league_def_fallback = float(ctx["pts_allowed_avg_10"].dropna().mean()) if ctx["pts_allowed_avg_10"].notna().any() else 112.0
    league_pace_fallback = float(ctx["pace_proxy_avg_10"].dropna().mean()) if ctx["pace_proxy_avg_10"].notna().any() else 225.0
    results: List[Dict[str, object]] = []

    # Track which games this run stored/updated (so we can fetch odds for them once)
    touched_event_ids: set[int] = set()
    only_player_keys = _tracked_player_keys()

    for p in PLAYERS:
        player_name = p["player_name"]
        player_id = p["player_id"]
        print(f"\n--- {player_name} ({player_id}) ---")

        gamelog_df, recent = compute_player_recent_features(player_id)
        if gamelog_df.empty:
            print("No 2025-26 games found (or all filtered out). Skipping.")
            continue
        if recent is None:
            print(f"Not enough games to compute rolling features yet (need {ROLL_10}). Skipping.")
            continue

        # Determine current team from most recent game matchup
        last_row = gamelog_df.tail(1).iloc[0]
        last_game_date = pd.Timestamp(last_row["GAME_DATE"])
        last_matchup = str(last_row["MATCHUP"])
        team_abbrev = parse_team_from_matchup(last_matchup)

        next_game = find_next_game_for_team(team_abbrev)
        if next_game is None:
            print(f"No scheduled game found within {SCHEDULE_LOOKAHEAD_DAYS} days for {team_abbrev}. Skipping.")
            continue

        next_game_date, opponent_abbrev, is_home = next_game
        rest_days = compute_rest_days(last_game_date, next_game_date)
        opp_ctx = team_ctx_latest.get(opponent_abbrev, {"pts_allowed_avg_10": np.nan, "pace_proxy_avg_10": np.nan})
        pts_allowed_avg_10 = opp_ctx.get("pts_allowed_avg_10", np.nan)
        pace_proxy_avg_10 = opp_ctx.get("pace_proxy_avg_10", np.nan)

        # Fill missing context with league averages
        if not np.isfinite(pts_allowed_avg_10):
            pts_allowed_avg_10 = league_def_fallback
        if not np.isfinite(pace_proxy_avg_10):
            pace_proxy_avg_10 = league_pace_fallback

        # star_pts_allowed_avg_10 is not computed from team gamelog; provide a reasonable proxy.
        # Simple fallback: use pts_allowed_avg_10
        star_pts_allowed_avg_10 = float(pts_allowed_avg_10)

        # Build a single-row feature frame
        X = pd.DataFrame([{
            "is_home": int(is_home),
            "rest_days": float(rest_days),
            "pts_avg_5": float(recent["pts_avg_5"]),
            "pts_avg_10": float(recent["pts_avg_10"]),
            "min_avg_5": float(recent["min_avg_5"]),
            "pts_allowed_avg_10": float(pts_allowed_avg_10),
            "pace_proxy_avg_10": float(pace_proxy_avg_10),
            "star_pts_allowed_avg_10": float(star_pts_allowed_avg_10),
            "player_name": player_name,
        }])

        # Predict
        pred_pts = float(pipeline.predict(X)[0])

        # Print next game info and the prediction
        print(f"Team: {team_abbrev} | Next: {'HOME' if is_home else 'AWAY'} vs {opponent_abbrev} on {next_game_date.date()}")
        print(f"Features: pts_avg_5={recent['pts_avg_5']:.2f}, pts_avg_10={recent['pts_avg_10']:.2f}, min_avg_5={recent['min_avg_5']:.2f}, rest_days={rest_days}")
        print(f"Opponent context: pts_allowed_avg_10={pts_allowed_avg_10:.2f}, pace_proxy_avg_10={pace_proxy_avg_10:.2f}, star_pts_allowed_avg_10={star_pts_allowed_avg_10:.2f}")
        print(f"Prediction: {pred_pts:.2f} points")

        next_game_date_local = str(next_game_date.date())
        odds_key = odds_key_for_matchup(next_game_date_local, team_abbrev, opponent_abbrev, int(is_home))
        odds_info = odds_map.get(odds_key)
        next_game_start_utc = odds_info["start_utc"] if odds_info else None
        odds_event_id = odds_info["event_id"] if odds_info else None
        
        # Persist Game to Postgres
        game = None
        if odds_event_id and next_game_start_utc:
            game, _ = Game.objects.get_or_create(
                odds_event_id=int(odds_event_id),
                defaults={
                    "home_team": team_abbrev if is_home else opponent_abbrev,
                    "away_team": opponent_abbrev if is_home else team_abbrev,
                    "start_time_utc": next_game_start_utc,
                },
            )
            touched_event_ids.add(int(odds_event_id))
               
        # Persist ModelPrediction to Postgres
        if game:
            ModelPrediction.objects.update_or_create(
                game=game,
                player_name=player_name,
                defaults={
                    "player_id": player_id,
                    "predicted_pts": float(round(pred_pts, 1)),
                    "model_version": "ridge_points_model_2024-25" # Change if other updated models are created
                },
            )

        results.append({
            "player_name": player_name,
            "player_id": player_id,
            "team": team_abbrev,
            "next_game_date_local": next_game_date_local,
            "next_game_start_utc": next_game_start_utc,
            "odds_event_id": odds_event_id,   
            "opponent": opponent_abbrev,
            "is_home": int(is_home),
            "rest_days": float(rest_days),
            "pts_avg_5": float(recent["pts_avg_5"]),
            "pts_avg_10": float(recent["pts_avg_10"]),
            "min_avg_5": float(recent["min_avg_5"]),
            "pts_allowed_avg_10": float(pts_allowed_avg_10),
            "pace_proxy_avg_10": float(pace_proxy_avg_10),
            "star_pts_allowed_avg_10": float(star_pts_allowed_avg_10),
            "predicted_pts": float(round(pred_pts, 1)),
        })

    if not results:
        print("\nNo predictions generated (likely not enough 2025-26 games yet, or schedule not found).")
        return

    out_df = (
        pd.DataFrame(results)
        .sort_values(["next_game_date_local", "player_name"])
        .reset_index(drop=True)
    )

    # ----------------------------------------------------
    # After predictions: fetch/store sportsbook lines once per touched game
    # ----------------------------------------------------
    if touched_event_ids:
        bookmakers_list = [b.strip() for b in ODDS_BOOKMAKERS.split(",") if b.strip()]
        print(f"\nFetching Odds API points props for {len(touched_event_ids)} game(s) from this run...")

        # Pull the Game rows for the touched event ids
        games = list(Game.objects.filter(odds_event_id__in=list(touched_event_ids)))
        total_created = 0
        total_updated = 0
        total_skipped = 0
        total_unavailable = 0

        for g in games:
            if game_already_has_any_points_lines(g, bookmakers_list):
                total_skipped += 1
                continue

            result = fetch_and_store_points_props_for_game(
                game=g,
                bookmakers_csv=ODDS_BOOKMAKERS,
                only_player_keys=only_player_keys,
                print_sample_limit=0,
            )

            if isinstance(result, tuple):
                if len(result) == 2:
                    c, u = result
                    unavail = 0
                elif len(result) == 3:
                    c, u, unavail = result
                else:
                    raise ValueError(f"Unexpected return from fetch_and_store_points_props_for_game: {result!r}")
            else:
                raise ValueError(f"fetch_and_store_points_props_for_game returned non-tuple: {type(result)}")

            total_created += c
            total_updated += u
            total_unavailable += unavail

        print(
            f"Odds store complete. created={total_created}, updated={total_updated}, "
            f"unavailable_lines={total_unavailable}, skipped_games={total_skipped} (already had lines)."
        )

    # Outputs predictions to CSV file at the root
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved predictions -> {OUTPUT_CSV}")
    print(
    out_df[
        ["player_name", "team", "next_game_date_local", "opponent", "is_home", "predicted_pts"]
    ].to_string(index=False)
)

if __name__ == "__main__":
    main()
