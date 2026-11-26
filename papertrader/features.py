"""Feature engineering shared by training and live decisions.

The model is trained on market-average odds only.  The live path receives a
single consensus snapshot (or a Bet365 fallback) and must produce exactly the
same 16 columns, in the same order, as the historical training frame.

Historical rows use football-data's local kickoff date/time.  Live rows convert
``kickoff_utc`` to Europe/Madrid before deriving the calendar features, so a
late kickoff does not accidentally move to the next day.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .domain import Fixture, OddsSnapshot, ensure_utc

LEAGUE_CODES = {
    "La Liga": 1,
    "Premier League": 2,
    "Bundesliga": 3,
    "Serie A": 4,
    "Ligue 1": 5,
}

# Order matters: the persisted model stores this list and the live path builds
# the one-row frame with the same order.
FEATURES: list[str] = [
    "logit_home_away",
    "logit_draw_away",
    "market_overround",
    "month",
    "day",
    "time_mins",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "time_sin",
    "time_cos",
    "is_weekend",
    "is_midweek",
    "matchday",
    "league_enc",
]

MARKET_AVERAGE_COLUMNS = [
    "market_avg_home_win",
    "market_avg_draw",
    "market_avg_away_win",
]


class FeatureError(ValueError):
    """Raised when a live fixture cannot be turned into a model row."""


def _as_arrays(home, draw, away):
    return (
        np.asarray(home, dtype=float),
        np.asarray(draw, dtype=float),
        np.asarray(away, dtype=float),
    )


def implied_probabilities(home, draw, away):
    """Normalize 1X2 decimal odds and return probabilities + overround.

    Raises ``ValueError`` for anything that is not a complete, finite set of
    decimal odds greater than 1.  This is the strict path used live.
    """
    h, d, a = _as_arrays(home, draw, away)
    if np.any(~np.isfinite(np.stack([h, d, a]))) or np.any(
        np.stack([h, d, a]) <= 1.0
    ):
        raise ValueError("odds must be finite decimal odds greater than 1")
    inverse = 1.0 / np.stack([h, d, a])
    overround = inverse.sum(axis=0)
    probabilities = inverse / overround
    return probabilities[0], probabilities[1], probabilities[2], overround


def _implied_probabilities_lenient(home, draw, away):
    """Vectorized probabilities for historical data with possible gaps."""
    h, d, a = _as_arrays(home, draw, away)
    stack = np.stack([1.0 / h, 1.0 / d, 1.0 / a])
    with np.errstate(divide="ignore", invalid="ignore"):
        overround = stack.sum(axis=0)
        probabilities = stack / overround
    invalid = (
        ~np.isfinite(overround)
        | (overround <= 0)
        | np.any(~np.isfinite(stack), axis=0)
    )
    for array in probabilities:
        array[invalid] = np.nan
    overround = np.where(invalid, np.nan, overround)
    return probabilities[0], probabilities[1], probabilities[2], overround


def market_favorite(home, draw, away):
    """Return the shortest-priced outcome using the project's tie rules."""
    h, d, a = _as_arrays(home, draw, away)
    conditions = [
        (h <= d) & (h <= a),
        (a <= d) & (h >= a),
        (a >= d) & (h >= d),
    ]
    result = np.select(conditions, ["home", "away", "draw"], default=None)
    if np.ndim(result) == 0:
        return str(result)
    return result


def favorite_confidence(home, draw, away) -> float:
    """Best (highest) normalized implied probability."""
    p_home, p_draw, p_away, _ = implied_probabilities(home, draw, away)
    return float(max(p_home, p_draw, p_away))


def _calendar_features(
    local_dt: datetime,
    season_start: date,
    league_enc: int,
) -> dict[str, float]:
    month = local_dt.month
    day = local_dt.day
    time_mins = local_dt.hour * 60 + local_dt.minute
    dow = local_dt.weekday()
    time_fraction = time_mins / 1440.0
    matchday = max((local_dt.date() - season_start).days // 7 + 1, 1)
    return {
        "month": float(month),
        "day": float(day),
        "time_mins": float(time_mins),
        "month_sin": float(np.sin(2 * np.pi * (month - 1) / 12)),
        "month_cos": float(np.cos(2 * np.pi * (month - 1) / 12)),
        "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
        "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
        "time_sin": float(np.sin(2 * np.pi * time_fraction)),
        "time_cos": float(np.cos(2 * np.pi * time_fraction)),
        "is_weekend": float(dow >= 5),
        "is_midweek": float(dow in (1, 2, 3)),
        "matchday": float(matchday),
        "league_enc": float(league_enc),
    }


def build_live_features(
    fixture: Fixture,
    snapshot: OddsSnapshot,
    season_start: date | None = None,
    league_enc: int | None = None,
    timezone: str = "Europe/Madrid",
) -> dict[str, float]:
    """Build the exact model row for one live fixture.

    ``season_start`` falls back to ``fixture.season_start``; if neither is
    available the feature is undefined and the decision must be skipped.
    """
    start = season_start or fixture.season_start
    if start is None:
        raise FeatureError(
            f"season_start missing for fixture {fixture.event_id}"
        )
    code = LEAGUE_CODES.get(fixture.league) if league_enc is None else league_enc
    if code is None:
        raise FeatureError(f"unknown league: {fixture.league!r}")
    if not snapshot.is_complete():
        raise FeatureError("snapshot does not contain a complete 1X2 price")

    p_home, p_draw, p_away, overround = implied_probabilities(
        snapshot.home, snapshot.draw, snapshot.away
    )
    local_dt = ensure_utc(fixture.kickoff_utc).astimezone(ZoneInfo(timezone))
    features = {
        "logit_home_away": float(np.log(p_home / p_away)),
        "logit_draw_away": float(np.log(p_draw / p_away)),
        "market_overround": float(overround),
    }
    features.update(_calendar_features(local_dt, start, int(code)))

    ordered = {name: features[name] for name in FEATURES}
    missing = [name for name, value in ordered.items() if not np.isfinite(value)]
    if missing:
        raise FeatureError(f"non-finite features: {sorted(missing)}")
    return ordered


def historical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add the model columns, market pick, and target to historical rows.

    Expected inputs come from :func:`simulator.load_matches` and include
    ``date``, ``time``, ``league``, ``season``, ``winner`` and the
    ``market_avg_*`` odds.  Rows with missing/zero input keep NaN features and
    are dropped by the trainer rather than silently imputed.
    """
    required = {
        "date", "time", "league", "season", "winner",
        *MARKET_AVERAGE_COLUMNS,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"historical data missing columns: {missing}")

    out = data.copy()
    dates = pd.to_datetime(out["date"], errors="coerce")
    times = pd.to_datetime(out["time"], format="%H:%M", errors="coerce")
    dow = dates.dt.day_of_week
    month = dates.dt.month
    time_mins = times.dt.hour * 60 + times.dt.minute
    time_fraction = time_mins / 1440.0

    out["month"] = month
    out["day"] = dates.dt.day
    out["time_mins"] = time_mins
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["time_sin"] = np.sin(2 * np.pi * time_fraction)
    out["time_cos"] = np.cos(2 * np.pi * time_fraction)
    out["is_weekend"] = (dow >= 5).astype(int)
    out["is_midweek"] = dow.isin([1, 2, 3]).astype(int)

    season_start = out.groupby(["league", "season"])["date"].transform("min")
    out["matchday"] = (
        (dates - pd.to_datetime(season_start)).dt.days // 7 + 1
    ).clip(lower=1)
    out["league_enc"] = out["league"].map(LEAGUE_CODES)

    home = out["market_avg_home_win"]
    draw = out["market_avg_draw"]
    away = out["market_avg_away_win"]
    p_home, p_draw, p_away, overround = _implied_probabilities_lenient(
        home, draw, away
    )
    out["market_prob_home"] = p_home
    out["market_prob_draw"] = p_draw
    out["market_prob_away"] = p_away
    out["market_overround"] = overround
    with np.errstate(divide="ignore", invalid="ignore"):
        out["logit_home_away"] = np.log(p_home / p_away)
        out["logit_draw_away"] = np.log(p_draw / p_away)

    out["selection"] = market_favorite(home, draw, away)
    out["target"] = (out["selection"] == out["winner"]).astype(int)
    out.loc[out["selection"].isna(), "target"] = np.nan
    return out


def feature_frame(features: dict[str, float], columns=None) -> pd.DataFrame:
    """One-row DataFrame with the persisted feature order."""
    order = list(columns or FEATURES)
    return pd.DataFrame([{name: features[name] for name in order}])
