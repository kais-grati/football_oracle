"""Tune the pseudo-linear sizer on validation, then test out of sample.

The data is split chronologically into train / validation / test (60/20/20).
Models are fitted on train only. For each discriminator, the pseudo-linear
sizer's `(x_trans, y_trans, slope_sec_curve)` are grid-searched on validation,
maximizing `OBJECTIVE`; the winning parameters are then evaluated once on the
held-out test window.

Run:
    python tune_pseudo_linear.py
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from simulator import (
    AllOfSelector,
    ModelSelector,
    PseudoLinearSizer,
    Simulator,
    ThresholdSelector,
    load_matches,
)

FEATURES = [
    "time_mins",
    "prob_home",
    "prob_draw",
    "prob_away",
    "market_avg_home_win",
    "market_avg_away_win",
    "market_avg_draw",
    "month",
    "day",
    "league_enc",
    "odds_home_win",
    "odds_away_win",
    "odds_draw",
]

LEAGUE = "La Liga"
TRAIN_RATIO = 0.6
VALID_RATIO = 0.2  # test uses the remaining 0.2
THRESHOLD = 0.65

X_TRANS_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
# y_trans=0 zeroes every stake below x_trans, turning the sizer into a hard
# threshold and collapsing the discriminator differences; keep a positive floor.
Y_TRANS_GRID = [0.01, 0.02, 0.05, 0.10, 0.15]
SLOPE_GRID = [0.00, 0.25, 0.50, 1.00, 2.00, 4.00]

OBJECTIVE = "final_bankroll"
MAX_DRAWDOWN_PCT = -100.0  # discard parameter sets below this, e.g. -50.0


def split_chronologically(data: pd.DataFrame):
    """Return train, validation and test slices in date order."""
    train_end = int(TRAIN_RATIO * len(data))
    valid_end = int((TRAIN_RATIO + VALID_RATIO) * len(data))
    return (
        data.iloc[:train_end].copy(),
        data.iloc[train_end:valid_end].copy(),
        data.iloc[valid_end:].copy(),
    )


def build_discriminators(train_raw: pd.DataFrame) -> dict:
    """Fit the models on train and return the discriminator selectors."""
    train_raw = train_raw.dropna(subset=FEATURES + ["guessed_correct"]).copy()

    train = train_raw.copy()
    scaler = StandardScaler()
    train[FEATURES] = scaler.fit_transform(train[FEATURES])

    logreg = LogisticRegression(random_state=42, max_iter=1000)
    logreg.fit(train[FEATURES], train_raw["guessed_correct"])

    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=42)
    tree.fit(train_raw[FEATURES], train_raw["guessed_correct"])

    logreg_selector = ModelSelector(logreg, FEATURES, scaler=scaler)
    tree_selector = ModelSelector(tree, FEATURES)
    threshold_selector = ThresholdSelector(THRESHOLD)

    return {
        "No filter": None,
        f"Threshold {THRESHOLD:.2f}": threshold_selector,
        "Logistic regression": logreg_selector,
        "Decision tree": tree_selector,
        "Logistic + tree agree": AllOfSelector(logreg_selector, tree_selector),
        "Logistic + tree + threshold agree": AllOfSelector(
            logreg_selector, tree_selector, threshold_selector
        ),
    }


def tune_selector(selector, validation: pd.DataFrame, sim: Simulator) -> pd.DataFrame:
    """Grid-search the pseudo-linear sizer on one validation slice."""
    rows = []
    for x_trans, y_trans, slope in product(X_TRANS_GRID, Y_TRANS_GRID, SLOPE_GRID):
        sizer = PseudoLinearSizer(x_trans, y_trans, slope)
        metrics = sim.run(validation, selector=selector, sizer=sizer).metrics

        score = metrics[OBJECTIVE]
        if (
            not metrics["bets"]
            or not np.isfinite(score)
            or metrics["max_drawdown_pct"] < MAX_DRAWDOWN_PCT
        ):
            score = -np.inf

        rows.append({
            "x_trans": x_trans,
            "y_trans": y_trans,
            "slope": slope,
            "bets": metrics["bets"],
            "hit_rate_pct": metrics["accuracy"] * 100,
            "final_bankroll": metrics["final_bankroll"],
            "roi_pct": metrics["roi_pct"],
            "yield_pct": metrics["yield_pct"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "score": score,
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["score", "max_drawdown_pct"], ascending=[False, False])
        .reset_index(drop=True)
    )


def run_optimization(data: pd.DataFrame | None = None, verbose: bool = True):
    """Tune on validation and evaluate the chosen parameters once on test.

    Returns the summary table and a dict of the full validation search per
    discriminator.
    """
    data = load_matches() if data is None else data
    train_raw, validation, test = split_chronologically(data)
    discriminators = build_discriminators(train_raw)
    sim = Simulator(league=LEAGUE)

    summary_rows = []
    searches = {}

    for name, selector in discriminators.items():
        search = tune_selector(selector, validation, sim)
        searches[name] = search
        best = search.iloc[0]

        if not np.isfinite(best["score"]):
            summary_rows.append({
                "Discriminator": name,
                "x_trans": np.nan, "y_trans": np.nan, "slope": np.nan,
                "Val Bets": 0, "Val ROI %": np.nan, "Val DD %": np.nan,
                "Test Bets": 0, "Test Hit %": np.nan, "Test Final": np.nan,
                "Test ROI %": np.nan, "Test DD %": np.nan,
            })
            continue

        sizer = PseudoLinearSizer(
            best["x_trans"], best["y_trans"], best["slope"]
        )
        test_metrics = sim.run(test, selector=selector, sizer=sizer).metrics

        summary_rows.append({
            "Discriminator": name,
            "x_trans": best["x_trans"],
            "y_trans": best["y_trans"],
            "slope": best["slope"],
            "Val Bets": int(best["bets"]),
            "Val ROI %": best["roi_pct"],
            "Val DD %": best["max_drawdown_pct"],
            "Test Bets": test_metrics["bets"],
            "Test Hit %": test_metrics["accuracy"] * 100,
            "Test Final": test_metrics["final_bankroll"],
            "Test ROI %": test_metrics["roi_pct"],
            "Test DD %": test_metrics["max_drawdown_pct"],
        })

    summary = pd.DataFrame(summary_rows).set_index("Discriminator")

    if verbose:
        test_ratio = 1.0 - TRAIN_RATIO - VALID_RATIO
        print(
            f"League: {LEAGUE} | split: {TRAIN_RATIO:.0%} train / "
            f"{VALID_RATIO:.0%} validation / {test_ratio:.0%} test"
        )
        print(
            f"Objective: {OBJECTIVE} | drawdown floor: {MAX_DRAWDOWN_PCT:.0f}% | "
            f"grid: {len(X_TRANS_GRID) * len(Y_TRANS_GRID) * len(SLOPE_GRID)} sets"
        )
        print(summary.round(2).to_string())

    return summary, searches


def main() -> pd.DataFrame:
    summary, _ = run_optimization()
    return summary


if __name__ == "__main__":
    main()
