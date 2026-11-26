from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from papertrader.domain import Fixture, OddsSnapshot
from papertrader.features import (
    FEATURES,
    FeatureError,
    build_live_features,
    favorite_confidence,
    feature_frame,
    historical_features,
    implied_probabilities,
    market_favorite,
)
from simulator import load_matches

MADRID = ZoneInfo("Europe/Madrid")


def test_live_features_match_historical_row():
    data = load_matches()
    frame = historical_features(data)
    row = frame.iloc[100]
    league = row["league"]
    season = row["season"]
    season_start = pd.to_datetime(
        frame[(frame["league"] == league) & (frame["season"] == season)]["date"]
    ).min().date()

    local_dt = datetime.combine(
        pd.to_datetime(row["date"]).date(),
        datetime.strptime(row["time"], "%H:%M").time(),
        tzinfo=MADRID,
    )
    fixture = Fixture(
        event_id="x",
        league=league,
        home_team="H",
        away_team="A",
        kickoff_utc=local_dt.astimezone(timezone.utc),
        season_start=season_start,
    )
    snapshot = OddsSnapshot(
        source="consensus",
        home=float(row["market_avg_home_win"]),
        draw=float(row["market_avg_draw"]),
        away=float(row["market_avg_away_win"]),
        observed_at=datetime.now(timezone.utc),
    )
    live = build_live_features(fixture, snapshot)

    assert list(live) == FEATURES
    for name in FEATURES:
        assert live[name] == pytest.approx(float(row[name]), rel=1e-12, abs=1e-12)


def test_live_features_use_local_wall_time_not_utc_date():
    # 00:30 Madrid on 29 Nov is 23:30 UTC on 28 Nov.
    local = datetime(2026, 11, 29, 0, 30, tzinfo=MADRID)
    fixture = Fixture(
        event_id="late",
        league="La Liga",
        home_team="H",
        away_team="A",
        kickoff_utc=local.astimezone(timezone.utc),
        season_start=date(2026, 8, 15),
    )
    snapshot = OddsSnapshot(
        source="consensus", home=1.6, draw=4.2, away=5.5,
        observed_at=datetime.now(timezone.utc),
    )
    features = build_live_features(fixture, snapshot)
    assert features["month"] == 11
    assert features["day"] == 29
    assert features["time_mins"] == 30
    assert features["dow_sin"] == pytest.approx(
        np.sin(2 * np.pi * 6 / 7)  # Sunday
    )


def test_market_probabilities_and_favorite():
    probs = implied_probabilities(1.5, 4.0, 7.0)
    assert sum(probs[:3]) == pytest.approx(1.0)
    assert probs[3] > 1.0
    assert favorite_confidence(1.5, 4.0, 7.0) == pytest.approx(max(probs[:3]))
    assert market_favorite(1.5, 4.0, 7.0) == "home"
    assert market_favorite(7.0, 4.0, 1.5) == "away"
    assert market_favorite(2.8, 2.8, 2.8) == "home"  # deterministic tie rule


def test_invalid_live_odds_raise():
    with pytest.raises(ValueError):
        implied_probabilities(1.0, 4.0, 7.0)
    with pytest.raises(ValueError):
        implied_probabilities(float("nan"), 4.0, 7.0)
    with pytest.raises(ValueError):
        implied_probabilities(1.5, None, 7.0)


def test_live_features_require_season_start_and_known_league():
    fixture = Fixture(
        event_id="x",
        league="La Liga",
        home_team="H",
        away_team="A",
        kickoff_utc=datetime(2026, 11, 28, 20, tzinfo=timezone.utc),
        season_start=None,
    )
    snapshot = OddsSnapshot(
        source="consensus", home=1.5, draw=4.0, away=7.0
    )
    with pytest.raises(FeatureError):
        build_live_features(fixture, snapshot)

    fixture = Fixture(
        event_id="x",
        league="Unknown",
        home_team="H",
        away_team="A",
        kickoff_utc=datetime(2026, 11, 28, 20, tzinfo=timezone.utc),
        season_start=date(2026, 8, 15),
    )
    with pytest.raises(FeatureError):
        build_live_features(fixture, snapshot)


def test_feature_frame_order_and_dtype():
    values = {name: float(i) for i, name in enumerate(FEATURES)}
    frame = feature_frame(values)
    assert list(frame.columns) == FEATURES
    assert frame.iloc[0].tolist() == [float(i) for i in range(len(FEATURES))]
