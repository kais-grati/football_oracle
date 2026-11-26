"""Run the notebook's discriminators through the class-based simulator.

Runs the threshold rule, logistic regression, decision tree, and their
agreement combinations on the test window.

Run:
    python run_discriminators.py
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from simulator import (
    AllOfSelector,
    KellySizer,
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
TRAIN_RATIO = 0.8
THRESHOLD = 0.65

PSEUDO_X_TRANS = 0.65
PSEUDO_Y_TRANS = 0.10
PSEUDO_SLOPE = 2.0
KELLY_FRACTION = 1/4
KELLY_CAP = 1/4

SIZINGS = {
    # Capped at 2% of bankroll.
    "Conservative": PseudoLinearSizer(0.65, 0.01, 0.10, max_fraction=0.02),
    # Capped at 5% of bankroll.
    "Moderate": PseudoLinearSizer(0.65, 0.02, 0.25, max_fraction=0.05),
    # Capped at 10% of bankroll.
    "Medium": PseudoLinearSizer(0.65, 0.04, 0.50, max_fraction=0.10),
    # Capped at 20% of bankroll.
    "Medium-High": PseudoLinearSizer(0.65, 0.06, 0.60, max_fraction=0.20),
    # Capped at 30% of bankroll.
    "High": PseudoLinearSizer(0.65, 0.08, 0.90, max_fraction=0.30),
    # Current settings, uncapped: stakes can exceed 50% of bankroll.
    "Aggressive": PseudoLinearSizer(PSEUDO_X_TRANS, PSEUDO_Y_TRANS, PSEUDO_SLOPE),
}
DEFAULT_SIZING = "Moderate"


def build_discriminators():
    """Fit the models and pair each discriminator with its sizer."""
    data = load_matches()
    split = int(TRAIN_RATIO * len(data))

    train_raw = data.iloc[:split].dropna(subset=FEATURES + ["guessed_correct"]).copy()
    test_raw = data.iloc[split:].copy()

    train = train_raw.copy()
    scaler = StandardScaler()
    train[FEATURES] = scaler.fit_transform(train[FEATURES])

    logreg = LogisticRegression(random_state=42, max_iter=1000)
    logreg.fit(train[FEATURES], train_raw["guessed_correct"])

    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=42)
    tree.fit(train_raw[FEATURES], train_raw["guessed_correct"])

    logreg_selector = ModelSelector(logreg, FEATURES, scaler=scaler)
    tree_selector = ModelSelector(tree, FEATURES)

    # Each model supplies its own win probability for the pick it backs.
    test_raw["logreg_prob"] = logreg_selector.probabilities(test_raw)
    test_raw["tree_prob"] = tree_selector.probabilities(test_raw)

    threshold_selector = ThresholdSelector(THRESHOLD)
    agree = AllOfSelector(logreg_selector, tree_selector)

    # (selector, sizing profile); each profile is resolved in main().
    experiments = {
        "No filter": (None, DEFAULT_SIZING),
        f"Threshold {THRESHOLD:.2f}": (threshold_selector, DEFAULT_SIZING),
        "Logistic regression": (logreg_selector, DEFAULT_SIZING),
        "Decision tree": (tree_selector, DEFAULT_SIZING),
        "Logistic + tree agree": (agree, DEFAULT_SIZING),
        "Logistic + tree + threshold agree": (
            AllOfSelector(logreg_selector, tree_selector, threshold_selector),
            DEFAULT_SIZING,
        ),
        "Logistic + tree agree [Conservative]": (agree, "Conservative"),
        "Logistic + tree agree [Medium]": (agree, "Medium"),
        "Logistic + tree agree [Medium-High]": (agree, "Medium-High"),
        "Logistic + tree agree [High]": (agree, "High"),
        "Logistic + tree agree [Aggressive]": (agree, "Aggressive"),
    }
    return experiments, test_raw


def main() -> pd.DataFrame:
    experiments, test_raw = build_discriminators()
    sim = Simulator(league=LEAGUE)

    rows = []
    for name, (selector, sizing) in experiments.items():
        sizer = SIZINGS[sizing]
        result = sim.run(test_raw, selector=selector, sizer=sizer)
        m = result.metrics
        rows.append({
            "Discriminator": name,
            "Sizing": sizing,
            "Bets": m["bets"],
            "Hit rate %": m["accuracy"] * 100,
            "Avg odds": m["avg_odds"],
            "Final bankroll": m["final_bankroll"],
            "ROI %": m["roi_pct"],
            "Yield %": m["yield_pct"],
            "Max drawdown %": m["max_drawdown_pct"],
        })

    table = pd.DataFrame(rows).set_index("Discriminator")
    profiles = ", ".join(
        f"{name} (max {sizer.max_fraction:.0%})"
        if sizer.max_fraction is not None
        else f"{name} (uncapped)"
        for name, sizer in SIZINGS.items()
    )
    print(f"League: {LEAGUE} | test window | default sizing: {DEFAULT_SIZING}")
    print(f"Sizing profiles: {profiles}")
    print(table.round(2).to_string())
    return table


if __name__ == "__main__":
    main()
