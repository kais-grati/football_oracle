from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import joblib
import pandas as pd
import pytest

from papertrader.features import FEATURES
from papertrader.model import (
    MarketModel,
    load_market_model,
    train_market_model,
)
from papertrader.sizing import WALLET_PROFILES, stake_for


def synthetic_history(rows: int = 180) -> pd.DataFrame:
    records = []
    start = date(2023, 8, 1)
    for index in range(rows):
        day = start + timedelta(days=index // 2)
        winner = ("home", "draw", "away")[index % 3]
        records.append({
            "match_id": str(index),
            "date": day.isoformat(),
            "time": "20:00",
            "league": "La Liga",
            "season": "2023-24",
            "winner": winner,
            "market_avg_home_win": 1.55,
            "market_avg_draw": 4.10,
            "market_avg_away_win": 6.20,
        })
    return pd.DataFrame(records)


def test_train_and_load_market_model(tmp_path):
    path = tmp_path / "model.joblib"
    metadata = train_market_model(path, synthetic_history())
    assert metadata["feature_set"] == "market_only_v1"
    assert metadata["rows"] == 180
    assert path.exists()

    model = load_market_model(path)
    assert model.features == FEATURES
    prediction = model.predict({name: 0.0 for name in FEATURES})
    assert 0.0 <= prediction.logreg_prob <= 1.0
    assert 0.0 <= prediction.tree_prob <= 1.0
    assert model.version == metadata["trained_at"]


def test_load_rejects_mismatched_feature_set(tmp_path):
    path = tmp_path / "bad.joblib"
    joblib.dump({
        "logreg": object(),
        "tree": object(),
        "features": ["wrong"],
        "metadata": {},
    }, path)
    with pytest.raises(ValueError):
        load_market_model(path)


def test_train_rejects_empty_or_single_class():
    with pytest.raises(ValueError):
        train_market_model("/tmp/never.joblib", synthetic_history(0))


def test_wallet_profiles_are_ordered_and_capped():
    confidence = 0.80
    available = 100.0
    stakes = {
        name: stake_for(name, confidence, available)
        for name in WALLET_PROFILES
    }
    assert stakes["Conservative"] <= 2.0
    assert stakes["Moderate"] <= 5.0
    assert stakes["Medium"] <= 10.0
    assert stakes["Medium-High"] <= 20.0
    assert stakes["High"] <= 30.0
    assert stakes["Conservative"] < stakes["Moderate"] < stakes["Medium"]
    assert stakes["Medium"] < stakes["Medium-High"] < stakes["High"]


def test_fallback_multiplier_halves_stake():
    normal = stake_for("Medium", 0.75, 100.0, 1.0)
    fallback = stake_for("Medium", 0.75, 100.0, 0.5)
    assert fallback == pytest.approx(normal * 0.5)


def test_stake_never_exceeds_available_and_bad_inputs_are_zero():
    assert stake_for("Aggressive", 0.95, 100.0, 1.0) <= 100.0
    assert stake_for("Medium", 0.75, 100.0, 2.0) <= 100.0
    assert stake_for("Medium", float("nan"), 100.0) == 0.0
    assert stake_for("Medium", 0.75, 0.0) == 0.0
    assert stake_for("Medium", 0.75, -1.0) == 0.0
    with pytest.raises(KeyError):
        stake_for("Unknown", 0.75, 100.0)
