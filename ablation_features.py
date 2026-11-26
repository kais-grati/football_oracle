"""Feature ablation for the Logistic + tree agree strategy.

Compares four feature sets under the same season-based walk-forward, fixed
Conservative 2% sizer, and La Liga evaluation:

- Book only
- Book + raw market averages
- Book + engineered market features
- Book + raw averages + engineered features

The engineered features are book/market overrounds, log price ratios, implied
probability differences, and whether the book and market agree on the favorite.

Run:
    python ablation_features.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from simulator import PseudoLinearSizer, Simulator, load_matches

LEAGUE = "La Liga"
TRAIN_SEASONS = 1
BOOTSTRAP_SAMPLES = 10_000
RANDOM_STATE = 42

SIZER = PseudoLinearSizer(0.65, 0.01, 0.10, max_fraction=0.02)

BOOK_FEATURES = [
    "time_mins",
    "prob_home",
    "prob_draw",
    "prob_away",
    "month",
    "day",
    "league_enc",
    "odds_home_win",
    "odds_away_win",
    "odds_draw",
]

MARKET_FEATURES = [
    "market_avg_home_win",
    "market_avg_draw",
    "market_avg_away_win",
]

ENGINEERED_FEATURES = [
    "book_overround",
    "market_overround",
    "price_ratio_home",
    "price_ratio_draw",
    "price_ratio_away",
    "prob_diff_home",
    "prob_diff_draw",
    "prob_diff_away",
    "favorite_agrees",
]

FEATURE_SETS = {
    "Book only": BOOK_FEATURES,
    "Book + avg": BOOK_FEATURES + MARKET_FEATURES,
    "Book + engineered": BOOK_FEATURES + ENGINEERED_FEATURES,
    "Book + avg + engineered": BOOK_FEATURES + MARKET_FEATURES + ENGINEERED_FEATURES,
}


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add book/market overrounds, price ratios, prob diffs, favorite agreement."""
    data = data.copy()
    odds = data[["odds_home_win", "odds_draw", "odds_away_win"]]
    average = data[MARKET_FEATURES]

    inverse_book = 1 / odds
    inverse_avg = 1 / average
    data["book_overround"] = inverse_book.sum(axis=1)
    data["market_overround"] = inverse_avg.sum(axis=1)

    market_prob = inverse_avg.div(inverse_avg.sum(axis=1), axis=0)
    prob_columns = {
        "home": "market_avg_home_win",
        "draw": "market_avg_draw",
        "away": "market_avg_away_win",
    }
    for side, avg_col in prob_columns.items():
        data[f"prob_diff_{side}"] = data[f"prob_{side}"] - market_prob[avg_col]

    price_columns = {
        "home": ("odds_home_win", "market_avg_home_win"),
        "draw": ("odds_draw", "market_avg_draw"),
        "away": ("odds_away_win", "market_avg_away_win"),
    }
    for side, (book_col, avg_col) in price_columns.items():
        data[f"price_ratio_{side}"] = np.log(data[book_col] / data[avg_col])

    book_fav = odds.to_numpy().argmin(axis=1)
    avg_values = average.to_numpy()
    valid = ~np.isnan(avg_values).any(axis=1)
    market_fav = np.full(len(data), np.nan)
    market_fav[valid] = avg_values[valid].argmin(axis=1)
    data["favorite_agrees"] = np.where(
        valid, (book_fav == market_fav).astype(float), np.nan
    )

    return data


def season_order(data: pd.DataFrame) -> list[str]:
    """Seasons in chronological order."""
    return data.groupby("season")["date"].min().sort_values().index.tolist()


def walk_forward_predictions(data: pd.DataFrame, seasons: list[str],
                             features: list[str]):
    """Out-of-sample logreg and tree predictions for one feature set."""
    logreg_pred = pd.Series(False, index=data.index)
    tree_pred = pd.Series(False, index=data.index)

    for i, season in enumerate(seasons):
        if i < TRAIN_SEASONS:
            continue

        train = data[data["season"].isin(seasons[:i])].dropna(
            subset=features + ["guessed_correct"]
        )
        test = data[data["season"] == season]

        logreg = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
        ]).fit(train[features], train["guessed_correct"])

        tree = DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=50, random_state=RANDOM_STATE
        ).fit(train[features], train["guessed_correct"])

        complete = test[features].notna().all(axis=1)
        if complete.any():
            X_test = test.loc[complete, features]
            idx = test.index[complete]
            logreg_pred.loc[idx] = logreg.predict(X_test).astype(bool)
            tree_pred.loc[idx] = tree.predict(X_test).astype(bool)

    return logreg_pred, tree_pred


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


def main():
    data = add_engineered_features(load_matches())
    seasons = season_order(data)
    test_seasons = seasons[TRAIN_SEASONS:]
    oos = data["season"].isin(test_seasons)
    sim = Simulator(league=LEAGUE)

    pooled_rows = []
    season_rows = []

    for name, features in FEATURE_SETS.items():
        logreg_pred, tree_pred = walk_forward_predictions(
            data, seasons, features
        )
        mask = oos & logreg_pred & tree_pred

        result = sim.run(data, selector=mask, sizer=SIZER)
        metrics = result.metrics
        low, high = bootstrap_yield(result.history)
        pooled_rows.append({
            "Feature set": name,
            "Features": len(features),
            "Bets": metrics["bets"],
            "Hit %": metrics["accuracy"] * 100,
            "Avg odds": metrics["avg_odds"],
            "Yield %": metrics["yield_pct"],
            "Yield 95% CI": f"{low:+.1f}% .. {high:+.1f}%",
            "ROI %": metrics["roi_pct"],
            "Max DD %": metrics["max_drawdown_pct"],
        })

        for season in test_seasons:
            metrics = sim.run(
                data,
                selector=mask & (data["season"] == season),
                sizer=SIZER,
            ).metrics
            season_rows.append({
                "Season": season,
                "Feature set": name,
                "Bets": metrics["bets"],
                "Yield %": metrics["yield_pct"],
                "ROI %": metrics["roi_pct"],
                "Max DD %": metrics["max_drawdown_pct"],
            })

    pooled = pd.DataFrame(pooled_rows).set_index("Feature set")
    per_season = pd.DataFrame(season_rows)

    print("Feature ablation — Logistic + tree agree | La Liga | Conservative 2%")
    print(f"Walk-forward: {TRAIN_SEASONS} training season(s); "
          f"test seasons {', '.join(test_seasons)}")
    print()
    print(pooled.round(2).to_string())

    print("\nPer-season yield (%)")
    yields = per_season.pivot(
        index="Season", columns="Feature set", values="Yield %"
    )
    print(yields.round(2).to_string())

    print("\nPer-season max drawdown (%)")
    drawdowns = per_season.pivot(
        index="Season", columns="Feature set", values="Max DD %"
    )
    print(drawdowns.round(2).to_string())

    return pooled, per_season


if __name__ == "__main__":
    main()
