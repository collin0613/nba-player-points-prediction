"""Manual cloud connectivity probe. No DB, model, payload storage, or retries.

Uses the same endpoints as src/data_io.py and src/odds_api.py but deliberately
avoids their project dependencies for the initial GitHub-hosted runner test.
One NBA request and, when a key is configured, one SportsGameOdds page per run.
"""
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests


def get_json(url, *, params, headers):
    response = requests.get(url, params=params, headers=headers,
                            timeout=(10, 30), allow_redirects=False)
    if response.status_code != 200:
        raise ValueError(f"HTTP {response.status_code}")
    return response.json()


def check_nba():
    payload = get_json(
        "https://stats.nba.com/stats/leaguegamelog",
        params={"Counter": 0, "Direction": "DESC", "LeagueID": "00",
                "PlayerOrTeam": "T", "Season": "2026-27",
                "SeasonType": "Regular Season", "Sorter": "DATE",
                "DateFrom": "", "DateTo": ""},
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
                 "Accept": "application/json, text/plain, */*",
                 "Referer": "https://www.nba.com/"},
    )
    result = next(item for item in payload["resultSets"]
                  if item["name"] == "LeagueGameLog")
    headers, rows = result["headers"], result["rowSet"]
    required = {"GAME_ID", "TEAM_ID", "PTS", "GAME_DATE"}
    if not required.issubset(headers) or not rows:
        raise ValueError("Empty or incomplete NBA result")
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("Malformed NBA row")
    records = [dict(zip(headers, row)) for row in rows]
    if any(r[k] is None for r in records for k in required):
        raise ValueError("Missing required NBA value")
    keys = {(r["GAME_ID"], r["TEAM_ID"]) for r in records}
    if len(keys) != len(records):
        raise ValueError("Duplicate NBA team/game rows")
    counts = Counter(r["GAME_ID"] for r in records)
    if any(count != 2 for count in counts.values()):
        raise ValueError("Expected two team rows per game")
    return f"{len(records)} team rows; {len(counts)} games"


def check_sportsbook():
    key = os.environ.get("SPORTSGAMEODDS_API_KEY", "").strip()
    if not key:
        raise ValueError("Missing SPORTSGAMEODDS_API_KEY repository secret")
    now = datetime.now(timezone.utc)
    payload = get_json(
        "https://api.sportsgameodds.com/v2/events",
        params={"leagueID": "NBA", "bookmakerID": "draftkings,fanduel",
                "includeAltLines": "false", "limit": 5,
                "startsAfter": now.isoformat(),
                "startsBefore": (now + timedelta(days=7)).isoformat()},
        headers={"x-api-key": key},
    )
    if (not isinstance(payload, dict) or payload.get("success") is False
            or not isinstance(payload.get("data"), list)
            or any(not isinstance(event, dict) for event in payload["data"])):
        raise ValueError("Invalid SportsGameOdds response envelope")
    events = payload["data"]
    if any(event.get("leagueID") != "NBA" for event in events):
        raise ValueError("Unexpected league in SportsGameOdds response")
    points = sum(
        1 for event in events for odd in (event.get("odds") or {}).values()
        if isinstance(odd, dict) and odd.get("statID") == "points"
        and odd.get("periodID") == "game" and odd.get("betTypeID") == "ou"
        and odd.get("playerID")
    )
    # Missing markets do not mean authentication/connectivity failed.
    return (f"{len(events)} events on first page; {points} player-points O/U entries. "
            "Connectivity only: no pagination or quote/identity validation."
            + (" No events available in this window." if not events else ""))


def main():
    messages = []
    failed = False
    for label, check in [("NBA Stats", check_nba), ("SportsGameOdds", check_sportsbook)]:
        started = time.monotonic()
        try:
            detail = check()
            status = "PASS"
        except requests.RequestException as exc:
            # Never print request objects, headers, response bodies, or credentials.
            status, detail = "FAIL", type(exc).__name__
            failed = True
        except Exception as exc:
            status = "FAIL"
            detail = str(exc) if type(exc) is ValueError else type(exc).__name__
            failed = True
        message = f"{label}: {status} ({time.monotonic()-started:.1f}s) — {detail}"
        print(message, flush=True)
        messages.append(message)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a") as handle:
            handle.write("## API connectivity results\n\n")
            handle.write("\n".join(f"- {message}" for message in messages) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
