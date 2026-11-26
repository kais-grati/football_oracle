"""Betting simulator for the football oracle project.

Three small pieces:

- a selector decides which matches to back,
- a sizer decides how much to stake on each one,
- Simulator runs the bankroll loop and returns metrics plus the bet history.

Usage::

    from simulator import Simulator, ThresholdSelector, FlatSizer, load_matches

    data = load_matches()
    result = Simulator(league="La Liga").run(
        data,
        selector=ThresholdSelector(0.65),
        sizer=FlatSizer(0.10),
    )
    print(result)
    result.history.tail()

See ``run_discriminators.py`` for a complete example.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ODDS_PATH = BASE_DIR / "football_matches_odds.csv"
FEATURES_PATH = BASE_DIR / "match_features.csv"


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------

class Selector:
    """Return a boolean mask over a prepared match frame."""

    def select(self, frame: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def __call__(self, frame: pd.DataFrame) -> pd.Series:
        return self.select(frame)


class ThresholdSelector(Selector):
    """Back matches whose top implied probability reaches `threshold`."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def select(self, frame: pd.DataFrame) -> pd.Series:
        confidence = frame[["prob_home", "prob_draw", "prob_away"]].max(axis=1)
        return confidence >= self.threshold


class ModelSelector(Selector):
    """Back matches a fitted classifier predicts correctly."""

    def __init__(self, model, features: Sequence[str], scaler=None):
        self.model = model
        self.features = list(features)
        self.scaler = scaler

    def _model_input(self, frame: pd.DataFrame):
        complete = frame[self.features].notna().all(axis=1)
        if not complete.any():
            return complete, None
        X = frame.loc[complete, self.features]
        if self.scaler is not None:
            X = pd.DataFrame(
                self.scaler.transform(X), index=X.index, columns=self.features
            )
        return complete, X

    def select(self, frame: pd.DataFrame) -> pd.Series:
        complete, X = self._model_input(frame)
        picked = pd.Series(False, index=frame.index)
        if complete.any():
            picked.loc[complete] = np.asarray(self.model.predict(X), dtype=bool)
        return picked

    def probabilities(self, frame: pd.DataFrame) -> pd.Series:
        """Probability that the row's backed pick wins, per the model."""
        complete, X = self._model_input(frame)
        probabilities = pd.Series(np.nan, index=frame.index)
        if complete.any():
            probabilities.loc[complete] = self.model.predict_proba(X)[:, 1]
        return probabilities


class WalkForwardSelector(Selector):
    """Back matches predicted out-of-sample by time-ordered folds."""

    def __init__(self, make_model: Callable[[], object],
                 features: Sequence[str], n_splits: int = 5):
        self.make_model = make_model
        self.features = list(features)
        self.n_splits = n_splits
        self.predictions: pd.Series | None = None

    def fit(self, data: pd.DataFrame) -> "WalkForwardSelector":
        from sklearn.model_selection import TimeSeriesSplit

        required = ["date", "match_id", *self.features, "guessed_correct"]
        source = data.dropna(subset=required)
        source = source.sort_values(["date", "match_id"]).reset_index(drop=True)

        picks = pd.Series(False, index=source.index)
        for train_idx, test_idx in TimeSeriesSplit(n_splits=self.n_splits).split(source):
            train, test = source.iloc[train_idx], source.iloc[test_idx]
            model = self.make_model()
            model.fit(train[self.features], train["guessed_correct"])
            picks.iloc[test_idx] = np.asarray(
                model.predict(test[self.features]), dtype=bool
            )

        self.predictions = pd.Series(
            picks.to_numpy(), index=source["match_id"].to_numpy()
        )
        return self

    def select(self, frame: pd.DataFrame) -> pd.Series:
        if self.predictions is None:
            raise RuntimeError("Call WalkForwardSelector.fit(data) first.")
        return frame["match_id"].map(self.predictions).fillna(False).astype(bool)


class AllOfSelector(Selector):
    """Back a match only when every wrapped selector backs it."""

    def __init__(self, *selectors):
        if not selectors:
            raise ValueError("AllOfSelector needs at least one selector.")
        self.selectors = selectors

    def select(self, frame: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=frame.index)
        for selector in self.selectors:
            picked = selector(frame)
            if isinstance(picked, pd.Series):
                picked = picked.reindex(frame.index, fill_value=False)
            else:
                picked = pd.Series(
                    np.asarray(picked, dtype=bool), index=frame.index
                )
            mask &= picked.fillna(False).astype(bool)
        return mask


# ---------------------------------------------------------------------------
# Sizers
# ---------------------------------------------------------------------------

class Sizer:
    """Return the amount to stake given the bankroll and the match row."""

    def stake(self, bankroll: float, match) -> float:
        raise NotImplementedError

    def __call__(self, bankroll: float, match) -> float:
        return self.stake(bankroll, match)


class FlatSizer(Sizer):
    """Stake a constant share of the current bankroll."""

    def __init__(self, fraction: float = 0.10):
        self.fraction = fraction

    def stake(self, bankroll: float, match) -> float:
        return bankroll * self.fraction


class KellySizer(Sizer):
    """Kelly stake from a model-probability column and the decimal odds.

    `probability` names a column in the simulation data. `fraction` scales
    Kelly (0.5 = half Kelly) and `cap` bounds the stake as a share of the
    bankroll. Picks with no positive edge are skipped.
    """

    def __init__(self, probability="model_prob", fraction=1.0, cap=0.25):
        self.probability = probability
        self.fraction = fraction
        self.cap = cap

    def stake(self, bankroll, match):
        p = getattr(match, self.probability, np.nan)
        odds = match.odds_taken
        net = odds - 1
        if not np.isfinite(p) or not np.isfinite(odds) or net <= 0:
            return 0.0
        kelly = (p * odds - 1) / net
        return bankroll * float(np.clip(self.fraction * kelly, 0.0, self.cap))


class ConfidenceSizer(Sizer):
    """Stake `fraction × confidence` of the current bankroll.

    Confidence is the top implied probability (0 to 1), so `fraction` is the
    largest share of the bankroll the sizer will stake.
    """

    def __init__(self, fraction: float = 0.05):
        self.fraction = fraction

    def stake(self, bankroll: float, match) -> float:
        return bankroll * self.fraction * match.confidence

class PseudoLinearSizer(Sizer):
    """Piecewise-linear stake from confidence.

    Below `x_trans` the stake fraction is `y_trans / x_trans × confidence`;
    above it, the line continues with `slope_sec_curve`. `max_fraction` caps
    the stake as a share of the current bankroll.
    """

    def __init__(self, x_trans, y_trans, slope_sec_curve, max_fraction=None):
        if x_trans <= 0:
            raise ValueError("x_trans must be positive")
        self.x_trans = x_trans
        self.y_trans = y_trans
        self.slope_sec_curve = slope_sec_curve
        self.max_fraction = max_fraction

    def stake(self, bankroll, match):
        conf = match.confidence

        if conf < self.x_trans:
            fraction = (self.y_trans / self.x_trans) * conf
        else:
            fraction = self.slope_sec_curve * conf - (
                self.slope_sec_curve * self.x_trans - self.y_trans
            )

        if self.max_fraction is not None:
            fraction = min(fraction, self.max_fraction)
        return fraction * bankroll


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    metrics: dict
    history: pd.DataFrame

    def __str__(self) -> str:
        m = self.metrics
        if not m["bets"]:
            return f"0 bets | final {m['final_bankroll']:.2f}"
        return (
            f"{m['bets']} bets | hit {m['accuracy']:.1%} | "
            f"avg odds {m['avg_odds']:.2f} | final {m['final_bankroll']:.2f} | "
            f"ROI {m['roi_pct']:+.1f}% | yield {m['yield_pct']:+.2f}% | "
            f"max DD {m['max_drawdown_pct']:.1f}%"
        )


class Simulator:
    """Run a bankroll simulation over prepared match data.

    `selector` may be a Selector/callable returning a boolean mask, a raw
    boolean mask aligned with `data`, or None for every match. `sizer` may be
    a Sizer/callable `(bankroll, match) -> stake`, or None for a flat
    `bet_percentage` of the bankroll.
    """

    def __init__(self, starting_capital: float = 100.0,
                 bet_percentage: float = 0.10, league: str | None = None):
        self.starting_capital = float(starting_capital)
        self.bet_percentage = float(bet_percentage)
        self.league = league

    def run(self, data: pd.DataFrame, selector=None, sizer=None) -> SimulationResult:
        sim = data.copy()

        raw_mask = isinstance(selector, (pd.Series, np.ndarray, list, tuple))
        if raw_mask:
            sim = sim[self._raw_mask(sim, selector)]
            selector = None

        if self.league is not None:
            sim = sim[sim["league"] == self.league]
        sim = _add_bet_columns(sim)
        sim = sim.sort_values(["date", "match_id"]).reset_index(drop=True)

        selected = self._selector_mask(sim, selector)
        sim = sim[
            selected & sim["odds_taken"].notna() & (sim["odds_taken"] > 1)
        ].reset_index(drop=True)

        return self._play(sim, sizer)

    @staticmethod
    def _raw_mask(frame: pd.DataFrame, mask) -> pd.Series:
        if isinstance(mask, pd.Series):
            return mask.reindex(frame.index, fill_value=False).astype(bool)
        values = np.asarray(mask, dtype=bool)
        if len(values) != len(frame):
            raise ValueError(
                f"Mask has {len(values)} entries but data has {len(frame)} rows"
            )
        return pd.Series(values, index=frame.index)

    @staticmethod
    def _selector_mask(frame: pd.DataFrame, selector) -> pd.Series:
        if selector is None:
            return pd.Series(True, index=frame.index)
        if callable(selector):
            mask = selector(frame)
            if isinstance(mask, pd.Series):
                return mask.reindex(frame.index, fill_value=False).astype(bool)
            return pd.Series(np.asarray(mask, dtype=bool), index=frame.index)
        raise TypeError(f"Unsupported selector: {type(selector)!r}")

    def _play(self, sim: pd.DataFrame, sizer) -> SimulationResult:
        if sizer is None:
            sizer = FlatSizer(self.bet_percentage)
        elif not callable(sizer):
            raise TypeError(f"Unsupported sizer: {type(sizer)!r}")

        bankroll = self.starting_capital
        history = []
        for match in sim.itertuples(index=False):
            stake = sizer(bankroll, match)
            if not np.isfinite(stake):
                stake = 0.0
            stake = float(np.clip(stake, 0.0, bankroll))
            if stake <= 0:
                continue

            profit = stake * (match.odds_taken - 1) if match.guessed_correct else -stake
            bankroll += profit
            history.append({
                "match_id": match.match_id,
                "date": match.date,
                "confidence": match.confidence,
                "odds": match.odds_taken,
                "won": match.guessed_correct,
                "stake": stake,
                "profit": profit,
                "bankroll": bankroll,
            })

        history = pd.DataFrame(history)
        return SimulationResult(self._metrics(history, bankroll), history)

    def _metrics(self, history: pd.DataFrame, bankroll: float) -> dict:
        if history.empty:
            return {
                "bets": 0, "accuracy": np.nan, "avg_confidence": np.nan,
                "avg_odds": np.nan, "avg_stake": 0.0,
                "final_bankroll": self.starting_capital, "net_profit": 0.0,
                "total_staked": 0.0, "roi_pct": np.nan, "yield_pct": np.nan,
                "max_drawdown_pct": 0.0,
            }

        peak = history["bankroll"].cummax()
        drawdown = (history["bankroll"] - peak) / peak
        net_profit = bankroll - self.starting_capital
        total_staked = history["stake"].sum()
        return {
            "bets": len(history),
            "accuracy": float(history["won"].mean()),
            "avg_confidence": float(history["confidence"].mean()),
            "avg_odds": float(history["odds"].mean()),
            "avg_stake": float(history["stake"].mean()),
            "final_bankroll": float(bankroll),
            "net_profit": float(net_profit),
            "total_staked": float(total_staked),
            "roi_pct": float(net_profit / self.starting_capital * 100),
            "yield_pct": float(net_profit / total_staked * 100),
            "max_drawdown_pct": float(drawdown.min() * 100),
        }


def _add_bet_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the odds, probability, edge and confidence behind the pick."""
    frame = frame.copy()
    frame["odds_taken"] = np.select(
        [frame["pred"] == "home", frame["pred"] == "draw", frame["pred"] == "away"],
        [frame["odds_home_win"], frame["odds_draw"], frame["odds_away_win"]],
        default=np.nan,
    )
    frame["prob_taken"] = np.select(
        [frame["pred"] == "home", frame["pred"] == "draw", frame["pred"] == "away"],
        [frame["prob_home"], frame["prob_draw"], frame["prob_away"]],
        default=np.nan,
    )
    frame["confidence"] = frame[["prob_home", "prob_draw", "prob_away"]].max(axis=1)
    frame["edge"] = frame["prob_taken"] * frame["odds_taken"] - 1
    return frame


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_matches(odds_path=ODDS_PATH, features_path=FEATURES_PATH) -> pd.DataFrame:
    """Load the odds and feature files into the frame the simulator expects."""
    matches = pd.read_csv(odds_path)
    features = pd.read_csv(features_path)

    skip = {"match_id", "date", "home_team", "away_team"}
    feature_cols = [c for c in features.columns if c not in skip]
    matches = matches.merge(
        features[["match_id", *feature_cols]], on="match_id", how="left"
    )

    conditions = [
        (matches["odds_home_win"] <= matches["odds_draw"])
        & (matches["odds_home_win"] <= matches["odds_away_win"]),
        (matches["odds_away_win"] <= matches["odds_draw"])
        & (matches["odds_home_win"] >= matches["odds_away_win"]),
        (matches["odds_away_win"] >= matches["odds_draw"])
        & (matches["odds_home_win"] >= matches["odds_draw"]),
    ]
    matches["pred"] = np.select(conditions, ["home", "away", "draw"], default=None)

    dates = pd.to_datetime(matches["date"])
    matches["month"] = dates.dt.month
    matches["day"] = dates.dt.day
    times = pd.to_datetime(matches["time"], format="%H:%M")
    matches["time_mins"] = times.dt.hour * 60 + times.dt.minute

    league_codes = {
        "La Liga": 1, "Premier League": 2, "Bundesliga": 3,
        "Serie A": 4, "Ligue 1": 5,
    }
    matches["league_enc"] = matches["league"].map(league_codes)

    inverse = 1 / matches[["odds_home_win", "odds_draw", "odds_away_win"]]
    total = inverse.sum(axis=1)
    matches["prob_home"] = inverse["odds_home_win"] / total
    matches["prob_draw"] = inverse["odds_draw"] / total
    matches["prob_away"] = inverse["odds_away_win"] / total

    matches["guessed_correct"] = matches["pred"] == matches["winner"]
    return matches.sort_values(["date", "match_id"]).reset_index(drop=True)
