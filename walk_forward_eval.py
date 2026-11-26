"""Season-by-season walk-forward evaluation.

For every season after `TRAIN_SEASONS`, the models are retrained on all prior
seasons and used to predict that season. All strategies are evaluated on the
same out-of-sample seasons, on La Liga only, with the Conservative, Moderate,
Medium, Medium-High, High and Aggressive pseudo-linear sizers.

Results are reported per season and pooled, with a bet-level bootstrap
confidence interval for the pooled yield on turnover.

Run:
    python walk_forward_eval.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from simulator import PseudoLinearSizer, Simulator, load_matches

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
THRESHOLD = 0.65
TRAIN_SEASONS = 1  # seasons used before the first out-of-sample season
BOOTSTRAP_SAMPLES = 10_000
RANDOM_STATE = 42

# Same profiles as run_discriminators.py, ordered by risk.
SIZERS = {
    "Conservative": PseudoLinearSizer(0.65, 0.01, 0.10, max_fraction=0.02),
    "Moderate": PseudoLinearSizer(0.65, 0.02, 0.25, max_fraction=0.05),
    "Medium": PseudoLinearSizer(0.65, 0.04, 0.50, max_fraction=0.10),
    "Medium-High": PseudoLinearSizer(0.65, 0.06, 0.60, max_fraction=0.20),
    "High": PseudoLinearSizer(0.65, 0.08, 0.90, max_fraction=0.30),
    "Aggressive": PseudoLinearSizer(0.65, 0.10, 2.0),
}


def season_order(data: pd.DataFrame) -> list[str]:
    """Seasons in chronological order."""
    return data.groupby("season")["date"].min().sort_values().index.tolist()


def fit_models(train: pd.DataFrame):
    """Fit logistic regression and decision tree on one training window."""
    train = train.dropna(subset=FEATURES + ["guessed_correct"])
    X = train[FEATURES]
    y = train["guessed_correct"]

    logreg = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
    ]).fit(X, y)

    tree = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=50, random_state=RANDOM_STATE
    ).fit(X, y)

    return logreg, tree


def walk_forward_predictions(data: pd.DataFrame, seasons: list[str]):
    """Out-of-sample predictions for each model, plus the fold sizes."""
    logreg_pred = pd.Series(False, index=data.index)
    tree_pred = pd.Series(False, index=data.index)
    folds = []

    for i, season in enumerate(seasons):
        if i < TRAIN_SEASONS:
            continue

        train = data[data["season"].isin(seasons[:i])]
        test = data[data["season"] == season]
        logreg, tree = fit_models(train)

        complete = test[FEATURES].notna().all(axis=1)
        if complete.any():
            X = test.loc[complete, FEATURES]
            idx = test.index[complete]
            logreg_pred.loc[idx] = logreg.predict(X).astype(bool)
            tree_pred.loc[idx] = tree.predict(X).astype(bool)

        folds.append({
            "Season": season,
            "Train matches": len(train),
            "Test matches": len(test),
            "La Liga test": int((test["league"] == LEAGUE).sum()),
        })

    return logreg_pred, tree_pred, pd.DataFrame(folds)


def build_strategies(data: pd.DataFrame, seasons: list[str],
                     logreg_pred: pd.Series,
                     tree_pred: pd.Series) -> dict[str, pd.Series]:
    """Boolean masks for every strategy on the out-of-sample seasons."""
    test_seasons = seasons[TRAIN_SEASONS:]
    oos = data["season"].isin(test_seasons)
    confidence = data[["prob_home", "prob_draw", "prob_away"]].max(axis=1)

    return {
        "No filter": oos,
        "Threshold 0.65": oos & (confidence >= THRESHOLD),
        "Logistic regression": oos & logreg_pred,
        "Decision tree": oos & tree_pred,
        "Logistic + tree agree": oos & logreg_pred & tree_pred,
    }


def bootstrap_yield(history: pd.DataFrame,
                    samples: int = BOOTSTRAP_SAMPLES) -> tuple[float, float]:
    """Bet-level 95% bootstrap interval for yield on turnover."""
    if history.empty:
        return np.nan, np.nan

    profits = history["profit"].to_numpy()
    stakes = history["stake"].to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.integers(0, len(history), size=(samples, len(history)))
    draws = profits[idx].sum(axis=1) / stakes[idx].sum(axis=1)

    low, high = np.percentile(draws, [2.5, 97.5]) * 100
    return float(low), float(high)


def evaluate_per_season(data: pd.DataFrame, strategies: dict[str, pd.Series],
                        seasons: list[str], sim: Simulator) -> pd.DataFrame:
    """Fresh-bankroll simulation for every sizing/strategy/season."""
    rows = []
    for sizing, sizer in SIZERS.items():
        for season in seasons[TRAIN_SEASONS:]:
            in_season = data["season"] == season
            for name, mask in strategies.items():
                metrics = sim.run(
                    data, selector=mask & in_season, sizer=sizer
                ).metrics
                rows.append({
                    "Sizing": sizing,
                    "Season": season,
                    "Strategy": name,
                    "Bets": metrics["bets"],
                    "Hit %": metrics["accuracy"] * 100,
                    "Yield %": metrics["yield_pct"],
                    "ROI %": metrics["roi_pct"],
                    "Max DD %": metrics["max_drawdown_pct"],
                })
    return pd.DataFrame(rows)


def evaluate_pooled(data: pd.DataFrame, strategies: dict[str, pd.Series],
                    sim: Simulator) -> pd.DataFrame:
    """One continuous bankroll per sizing/strategy across OOS seasons."""
    rows = []
    for sizing, sizer in SIZERS.items():
        for name, mask in strategies.items():
            result = sim.run(data, selector=mask, sizer=sizer)
            metrics = result.metrics
            low, high = bootstrap_yield(result.history)
            rows.append({
                "Sizing": sizing,
                "Strategy": name,
                "Bets": metrics["bets"],
                "Hit %": metrics["accuracy"] * 100,
                "Avg odds": metrics["avg_odds"],
                "Yield %": metrics["yield_pct"],
                "Yield 95% CI": f"{low:+.1f}% .. {high:+.1f}%",
                "ROI %": metrics["roi_pct"],
                "Final": metrics["final_bankroll"],
                "Max DD %": metrics["max_drawdown_pct"],
            })
    return pd.DataFrame(rows).set_index(["Sizing", "Strategy"])


def main():
    data = load_matches()
    seasons = season_order(data)
    if len(seasons) <= TRAIN_SEASONS:
        raise ValueError("Not enough seasons for the requested burn-in.")

    logreg_pred, tree_pred, folds = walk_forward_predictions(data, seasons)
    strategies = build_strategies(data, seasons, logreg_pred, tree_pred)
    sim = Simulator(league=LEAGUE)

    per_season = evaluate_per_season(data, strategies, seasons, sim)
    pooled = evaluate_pooled(data, strategies, sim)

    print("Walk-forward folds")
    print(folds.to_string(index=False))

    profiles = ", ".join(
        f"{name} (max {sizer.max_fraction:.0%})"
        if sizer.max_fraction is not None
        else f"{name} (uncapped)"
        for name, sizer in SIZERS.items()
    )
    print(f"\nLeague: {LEAGUE}")
    print(f"Sizing profiles: {profiles}")

    agreement = per_season[
        per_season["Strategy"] == "Logistic + tree agree"
    ].copy()
    agreement["Sizing"] = pd.Categorical(
        agreement["Sizing"], categories=list(SIZERS), ordered=True
    )

    for metric in ["Yield %", "ROI %", "Max DD %"]:
        print(f"\nPer-season {metric} — Logistic + tree agree")
        table = agreement.pivot(
            index="Season", columns="Sizing", values=metric
        )
        print(table.round(2).to_string())

    print("\nPooled out-of-sample — Logistic + tree agree")
    print(
        pooled.xs("Logistic + tree agree", level="Strategy")
        .round(2)
        .to_string()
    )

    print("\nPooled yield (%) by strategy")
    yield_by_strategy = (
        pooled["Yield %"].unstack("Strategy").reindex(list(SIZERS))
    )
    print(yield_by_strategy.round(2).to_string())

    return per_season, pooled


if __name__ == "__main__":
    main()
