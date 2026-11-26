"""Preprocessing ablation for the market-average-only, Bet365-execution model.

Baseline features:
    market_avg_* odds, market_prob_* probabilities, market_overround,
    month, day, time_mins, league_enc.

Tested changes, one at a time:
- log_odds: replace raw odds + probabilities with three log odds.
- two_logits: two probability logits (home/away, draw/away) + overround.
- calendar_flags: add is_weekend, is_midweek, matchday.
- cyclical_time: replace raw month/day/time_mins with sin/cos for month,
  day-of-week, and time-of-day, plus weekend/midweek/matchday flags.
- onehot_league: replace ordinal league_enc with league one-hot columns.
- pairings and the combined two_logits + cyclical + one-hot set.

Primary evaluation trains on all leagues and tests each league separately.
A secondary run trains only on La Liga and tests La Liga.

The sizer is the one currently configured in market_only_eval.py.

Run: python market_feature_ablation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ablation_features import (
    bootstrap_yield,
    season_order,
    walk_forward_predictions,
)
from market_only_eval import SIZER, market_frame
from simulator import Simulator, load_matches

LEAGUES = ["La Liga", "Premier League", "Bundesliga", "Serie A", "Ligue 1"]
REPORT_DIR = Path("reports")

REP_RAW = [
    "market_avg_home_win",
    "market_avg_draw",
    "market_avg_away_win",
    "market_prob_home",
    "market_prob_draw",
    "market_prob_away",
    "market_overround",
]
REP_LOG_ODDS = [
    "log_market_avg_home",
    "log_market_avg_draw",
    "log_market_avg_away",
    "market_overround",
]
REP_LOGITS = ["logit_home_away", "logit_draw_away", "market_overround"]

RAW_TIME = ["month", "day", "time_mins"]
CYCLICAL = [
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "time_sin",
    "time_cos",
    "is_weekend",
    "is_midweek",
    "matchday",
]
FLAGS = ["is_weekend", "is_midweek", "matchday"]

LEAGUE_COLS = [
    "league_La_Liga",
    "league_Premier_League",
    "league_Bundesliga",
    "league_Serie_A",
    "league_Ligue_1",
]

VARIANTS = {
    "baseline": REP_RAW + RAW_TIME + ["league_enc"],
    "log_odds": REP_LOG_ODDS + RAW_TIME + ["league_enc"],
    "two_logits": REP_LOGITS + RAW_TIME + ["league_enc"],
    "two_logits_calendar": REP_LOGITS + RAW_TIME + ["league_enc"] + FLAGS,
    "two_logits_cyclical": REP_LOGITS + CYCLICAL + ["league_enc"],
    "two_logits_calendar_cyclical": (
        REP_LOGITS + RAW_TIME + CYCLICAL + ["league_enc"]
    ),
    "calendar_flags": REP_RAW + RAW_TIME + ["league_enc"] + FLAGS,
    "cyclical_time": REP_RAW + CYCLICAL + ["league_enc"],
    "onehot_league": REP_RAW + RAW_TIME + LEAGUE_COLS,
    "two_logits_onehot": REP_LOGITS + RAW_TIME + LEAGUE_COLS,
    "cyclical_onehot": REP_RAW + CYCLICAL + LEAGUE_COLS,
    "two_logits_cyclical_onehot": REP_LOGITS + CYCLICAL + LEAGUE_COLS,
    "log_odds_cyclical_onehot": REP_LOG_ODDS + CYCLICAL + LEAGUE_COLS,
}


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add market representations, calendar features, and league one-hot."""
    out = data.copy()
    dates = pd.to_datetime(out["date"])
    dow = dates.dt.day_of_week
    month = dates.dt.month

    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    time_fraction = out["time_mins"] / 1440.0
    out["time_sin"] = np.sin(2 * np.pi * time_fraction)
    out["time_cos"] = np.cos(2 * np.pi * time_fraction)

    out["is_weekend"] = (dow >= 5).astype(int)
    out["is_midweek"] = dow.isin([1, 2, 3]).astype(int)
    season_start = out.groupby(["league", "season"])["date"].transform("min")
    out["matchday"] = (
        (dates - pd.to_datetime(season_start)).dt.days // 7 + 1
    ).clip(lower=1)

    probabilities = out[
        ["market_prob_home", "market_prob_draw", "market_prob_away"]
    ].clip(1e-9, 1 - 1e-9)
    out["logit_home_away"] = np.log(
        probabilities["market_prob_home"] / probabilities["market_prob_away"]
    )
    out["logit_draw_away"] = np.log(
        probabilities["market_prob_draw"] / probabilities["market_prob_away"]
    )

    average_columns = {
        "home": "market_avg_home_win",
        "draw": "market_avg_draw",
        "away": "market_avg_away_win",
    }
    for side, column in average_columns.items():
        out[f"log_market_avg_{side}"] = np.log(out[column])

    short = {
        "La Liga": "La_Liga",
        "Premier League": "Premier_League",
        "Bundesliga": "Bundesliga",
        "Serie A": "Serie_A",
        "Ligue 1": "Ligue_1",
    }
    out["league_short"] = out["league"].map(short)
    dummies = pd.get_dummies(
        out["league_short"], prefix="league", dtype=int
    )
    out = pd.concat([out, dummies], axis=1)
    return out


def metrics_row(result, variant: str, training: str, league: str) -> dict:
    metrics = result.metrics
    low, high = bootstrap_yield(result.history)
    return {
        "Variant": variant,
        "Training": training,
        "League": league,
        "Bets": metrics["bets"],
        "Hit %": metrics["accuracy"] * 100,
        "Avg odds": metrics["avg_odds"],
        "Yield %": metrics["yield_pct"],
        "Yield 95% CI": f"{low:+.1f}% .. {high:+.1f}%",
        "ROI %": metrics["roi_pct"],
        "Final": metrics["final_bankroll"],
        "Max DD %": metrics["max_drawdown_pct"],
    }


def evaluate_variant(frame: pd.DataFrame, features: list[str],
                     variant: str) -> tuple[list[dict], pd.DataFrame]:
    """Train on all leagues, evaluate each league and the union."""
    seasons = season_order(frame)
    oos = frame["season"].isin(seasons[1:])
    logreg, tree = walk_forward_predictions(frame, seasons, features)
    mask = oos & logreg & tree

    rows = []
    for league in LEAGUES + ["All leagues"]:
        if league == "All leagues":
            sim = Simulator(league=None)
            selector = mask
        else:
            sim = Simulator(league=league)
            selector = mask & (frame["league"] == league)
        result = sim.run(frame, selector=selector, sizer=SIZER)
        rows.append(metrics_row(result, variant, "All leagues", league))

    season_rows = []
    sim = Simulator(league="La Liga")
    for season in seasons[1:]:
        result = sim.run(
            frame,
            selector=mask
            & (frame["league"] == "La Liga")
            & (frame["season"] == season),
            sizer=SIZER,
        )
        metrics = result.metrics
        season_rows.append({
            "Variant": variant,
            "Season": season,
            "Bets": metrics["bets"],
            "Yield %": metrics["yield_pct"],
            "ROI %": metrics["roi_pct"],
            "Max DD %": metrics["max_drawdown_pct"],
        })
    return rows, pd.DataFrame(season_rows)


def evaluate_laliga_only(frame: pd.DataFrame, features: list[str],
                         variant: str) -> dict:
    """Train only on La Liga and evaluate La Liga."""
    league_frame = frame[frame["league"] == "La Liga"].copy()
    seasons = season_order(league_frame)
    oos = league_frame["season"].isin(seasons[1:])
    logreg, tree = walk_forward_predictions(
        league_frame, seasons, features
    )
    mask = oos & logreg & tree
    result = Simulator(league="La Liga").run(
        league_frame, selector=mask, sizer=SIZER
    )
    return metrics_row(result, variant, "La Liga only", "La Liga")


def main():
    data = market_frame(load_matches(), "bet365")
    frame = add_features(data)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    pooled_rows = []
    season_rows = []
    laliga_rows = []

    for variant, features in VARIANTS.items():
        print(f"Running {variant}...")
        rows, by_season = evaluate_variant(frame, features, variant)
        pooled_rows.extend(rows)
        season_rows.append(by_season)
        laliga_rows.append(evaluate_laliga_only(frame, features, variant))

    pooled = pd.DataFrame(pooled_rows)
    per_season = pd.concat(season_rows, ignore_index=True)
    laliga_only = pd.DataFrame(laliga_rows)

    main = pooled[pooled["Training"] == "All leagues"]
    yields = main.pivot(
        index="Variant", columns="League", values="Yield %"
    ).reindex(list(VARIANTS))
    yields = yields[LEAGUES + ["All leagues"]]
    baseline = yields.loc["baseline"]
    delta = yields.subtract(baseline)

    laliga_scope = laliga_only.merge(
        main[main["League"] == "La Liga"][
            ["Variant", "Yield %", "ROI %", "Max DD %", "Bets"]
        ],
        on="Variant",
        suffixes=(" (La Liga only)", " (all leagues)"),
    ).set_index("Variant")

    season_yields = per_season.pivot(
        index="Variant", columns="Season", values="Yield %"
    ).reindex(list(VARIANTS))

    print("\n=== All-league training — yield (%) by test league ===")
    print(yields.round(2).to_string())
    print("\n=== Yield delta vs baseline (percentage points) ===")
    print(delta.round(2).to_string())
    print("\n=== La Liga: training scope ===")
    print(laliga_scope.round(2).to_string())
    print("\n=== La Liga per-season yield (%) ===")
    print(season_yields.round(2).to_string())

    pooled.to_csv(REPORT_DIR / "market_feature_ablation_pooled.csv", index=False)
    yields.round(4).to_csv(REPORT_DIR / "market_feature_ablation_yields.csv")
    delta.round(4).to_csv(REPORT_DIR / "market_feature_ablation_delta.csv")
    laliga_scope.round(4).to_csv(
        REPORT_DIR / "market_feature_ablation_laliga_scope.csv"
    )
    season_yields.round(4).to_csv(
        REPORT_DIR / "market_feature_ablation_laliga_seasons.csv"
    )
    print(f"\nFull metrics: {REPORT_DIR / 'market_feature_ablation_pooled.csv'}")
    return pooled, per_season


if __name__ == "__main__":
    main()
