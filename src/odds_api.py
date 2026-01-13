# src/odds_api.py
# Fetch odds data from Odds API, focusing on player points props.

from __future__ import annotations

import os
import time
import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


ODDS_API_BASE_URL = "https://api2.odds-api.io/v3/odds"


@dataclass(frozen=True)
class PlayerPointsLine:
    event_id: str
    bookmaker: str
    player_name: str
    line: float
    over_price: Optional[float]
    under_price: Optional[float]
    market_name: str  # e.g. "Player Props - Points"


class OddsApiClient:
    def __init__(self, api_key: Optional[str] = None, timeout_s: int = 20):
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing ODDS_API_KEY env var.")
        self.timeout_s = timeout_s

    def get_odds_for_event(
        self,
        event_id: str,
        bookmakers: Optional[List[str]] = None,
        retries: int = 2,
        backoff_s: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Fetch raw odds response for a single event.
        """
        params = {
            "apiKey": self.api_key,
            "eventId": event_id,
        }
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                resp = requests.get(ODDS_API_BASE_URL, params=params, timeout=self.timeout_s)
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, dict):
                    raise ValueError("Unexpected response format (expected dict).")
                return data
            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(backoff_s * (attempt + 1))
                else:
                    raise RuntimeError(f"Odds API request failed after retries: {e}") from e

        # should never reach
        raise RuntimeError(f"Odds API request failed: {last_err}")

    @staticmethod
    def extract_player_points_lines(raw: Dict[str, Any], event_id: str) -> List[PlayerPointsLine]:
        """
        Parse out all Player Props - Points lines across bookmakers.
        Returns one record per (bookmaker, player) line.
        """
        results: List[PlayerPointsLine] = []

        bookmakers_obj = raw.get("bookmakers", {})
        if not isinstance(bookmakers_obj, dict):
            return results

        for bookmaker_name, markets in bookmakers_obj.items():
            if not isinstance(markets, list):
                continue

            for market in markets:
                if not isinstance(market, dict):
                    continue

                market_name = str(market.get("name", "")).strip()

                # Keep it strict: only points props
                # (You can expand later if you want rebounds/assists etc.)
                is_points_props = "player props" in market_name.lower() and "points" in market_name.lower()
                if not is_points_props:
                    continue

                odds_list = market.get("odds", [])
                if not isinstance(odds_list, list):
                    continue

                for o in odds_list:
                    if not isinstance(o, dict):
                        continue

                    player_name = str(o.get("label", "")).strip()
                    if not player_name:
                        continue

                    # hdp is the line, usually a float like 27.5
                    hdp = o.get("hdp", None)
                    try:
                        line = float(hdp)
                    except (TypeError, ValueError):
                        continue

                    over_price = OddsApiClient._safe_float(o.get("over"))
                    under_price = OddsApiClient._safe_float(o.get("under"))

                    results.append(
                        PlayerPointsLine(
                            event_id=str(event_id),
                            bookmaker=str(bookmaker_name),
                            player_name=player_name,
                            line=line,
                            over_price=over_price,
                            under_price=under_price,
                            market_name=market_name,
                        )
                    )

        return results

    @staticmethod
    def _safe_float(x: Any) -> Optional[float]:
        if x is None:
            return None
        try:
            return float(x)
        except (TypeError, ValueError):
            return None


def normalize_player_name(name: str) -> str:
    """
    Minimal normalization so matching is less brittle.
    You can add more rules as you discover mismatches.
    """
    n = name.strip().lower()
    n = n.replace(".", "")
    n = n.replace("’", "'")
    n = " ".join(n.split())
    return n
