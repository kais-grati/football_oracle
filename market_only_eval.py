"""Compare the current model with a market-average-only discriminator.

Variants:
- Current: Bet365 + engineered features, Bet365 execution.
- Market only, Bet365 execution: market-average features and selection, bets
  settled at the Bet365 price of the selected market favorite.
- Market only, consensus execution: same model/selection, settled at the
  market-average price (not executable as one book; isolates the signal).
- Market favorite, no model: all market favorites, no discriminator.

Walk-forward: train on prior seasons, test 2022-23..2025-26, La Liga,
Conservative 2% sizer, logreg + tree must agree.

Run: python market_only_eval.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ablation_features import (
    BOOK_FEATURES,
    ENGINEERED_FEATURES,
    add_engineered_features,
    bootstrap_yield,
    season_order,
    walk_forward_predictions,
)
from simulator import PseudoLinearSizer, Simulator, load_matches

LEAGUE = "La Liga"
SIZER = PseudoLinearSizer(0.65, 0.04, 0.50, max_fraction=0.10)

CURRENT_FEATURES = BOOK_FEATURES + ENGINEERED_FEATURES
MARKET_FEATURES = [
    "time_mins",
    "market_prob_home",
    "market_prob_draw",
    "market_prob_away",
    "market_avg_home_win",
    "market_avg_draw",
    "market_avg_away_win",
    "market_overround",
    "month",
    "day",
    "league_enc",
]


def market_view(data: pd.DataFrame) -> pd.DataFrame:
    """Add market-average probabilities, favorite, and target."""
    out = data.copy()
    avg = out[["market_avg_home_win", "market_avg_draw", "market_avg_away_win"]]
    inverse = 1 / avg
    total = inverse.sum(axis=1)

    out["market_prob_home"] = inverse["market_avg_home_win"] / total
    out["market_prob_draw"] = inverse["market_avg_draw"] / total
    out["market_prob_away"] = inverse["market_avg_away_win"] / total
    out["market_overround"] = total

    conditions = [
        (avg["market_avg_home_win"] <= avg["market_avg_draw"])
        & (avg["market_avg_home_win"] <= avg["market_avg_away_win"]),
        (avg["market_avg_away_win"] <= avg["market_avg_draw"])
        & (avg["market_avg_home_win"] >= avg["market_avg_away_win"]),
        (avg["market_avg_away_win"] >= avg["market_avg_draw"])
        & (avg["market_avg_home_win"] >= avg["market_avg_draw"]),
    ]
    out["pred_market"] = np.select(
        conditions, ["home", "away", "draw"], default=None
    )
    out["market_guessed"] = out["pred_market"] == out["winner"]
    return out


def run_variant(frame: pd.DataFrame, features: list[str], mask,
                label: str) -> tuple[dict, pd.DataFrame]:
    sim = Simulator(league=LEAGUE)
    result = sim.run(frame, selector=mask, sizer=SIZER)
    metrics = result.metrics
    low, high = bootstrap_yield(result.history)
    row = {
        "Variant": label,
        "Bets": metrics["bets"],
        "Hit %": metrics["accuracy"] * 100,
        "Avg odds": metrics["avg_odds"],
        "Yield %": metrics["yield_pct"],
        "Yield 95% CI": f"{low:+.1f}% .. {high:+.1f}%",
        "ROI %": metrics["roi_pct"],
        "Final": metrics["final_bankroll"],
        "Max DD %": metrics["max_drawdown_pct"],
    }

    seasons = season_order(frame)
    rows = []
    for season in seasons[1:]:
        season_metrics = sim.run(
            frame,
            selector=mask & (frame["season"] == season),
            sizer=SIZER,
        ).metrics
        rows.append({
            "Season": season,
            "Variant": label,
            "Bets": season_metrics["bets"],
            "Yield %": season_metrics["yield_pct"],
            "ROI %": season_metrics["roi_pct"],
            "Max DD %": season_metrics["max_drawdown_pct"],
        })
    return row, pd.DataFrame(rows)


def market_frame(data: pd.DataFrame, execution: str) -> pd.DataFrame:
    """Market-only selection/features, with either consensus or Bet365 prices."""
    frame = market_view(data)
    if execution == "consensus":
        frame["odds_home_win"] = frame["market_avg_home_win"]
        frame["odds_draw"] = frame["market_avg_draw"]
        frame["odds_away_win"] = frame["market_avg_away_win"]
    # The sizer and model use market probabilities and the market favorite.
    frame["prob_home"] = frame["market_prob_home"]
    frame["prob_draw"] = frame["market_prob_draw"]
    frame["prob_away"] = frame["market_prob_away"]
    frame["pred"] = frame["pred_market"]
    frame["guessed_correct"] = frame["market_guessed"]
    return frame


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_matches()

    # Current model: Bet365 + engineered features, Bet365 execution.
    current = add_engineered_features(data)
    seasons = season_order(current)
    oos = current["season"].isin(seasons[1:])
    logreg, tree = walk_forward_predictions(
        current, seasons, CURRENT_FEATURES
    )
    current_mask = oos & logreg & tree
    current_row, current_seasons = run_variant(
        current, CURRENT_FEATURES, current_mask,
        "Current: Bet365 + engineered",
    )

    rows = [current_row]
    season_rows = [current_seasons]

    # Market-average-only model, two execution prices.
    for execution, label in [
        ("bet365", "Market avg only, Bet365 execution"),
        ("consensus", "Market avg only, consensus execution"),
    ]:
        frame = market_frame(data, execution)
        seasons = season_order(frame)
        oos = frame["season"].isin(seasons[1:])
        logreg, tree = walk_forward_predictions(
            frame, seasons, MARKET_FEATURES
        )
        mask = oos & logreg & tree
        row, by_season = run_variant(frame, MARKET_FEATURES, mask, label)
        rows.append(row)
        season_rows.append(by_season)

    # No-model baseline: every market favorite.
    for execution, label in [
        ("bet365", "Market favorite, no model (Bet365)"),
        ("consensus", "Market favorite, no model (consensus)"),
    ]:
        frame = market_frame(data, execution)
        seasons = season_order(frame)
        oos = frame["season"].isin(seasons[1:])
        mask = oos & frame["market_guessed"].notna()
        row, by_season = run_variant(frame, MARKET_FEATURES, mask, label)
        rows.append(row)
        season_rows.append(by_season)

    pooled = pd.DataFrame(rows).set_index("Variant")
    per_season = pd.concat(season_rows, ignore_index=True)

    print("Walk-forward comparison — La Liga, Conservative sizer")
    print(pooled.round(2).to_string())
    print("\nPer-season yield (%)")
    yields = per_season.pivot(
        index="Season", columns="Variant", values="Yield %"
    )
    print(yields.round(2).to_string())
    print("\nPer-season bets")
    bets = per_season.pivot(
        index="Season", columns="Variant", values="Bets"
    )
    print(bets.fillna(0).astype(int).to_string())

    return pooled, per_season


if __name__ == "__main__":
    main()
