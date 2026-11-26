"""Training and loading the market-only logreg + tree pair.

The persisted object is deliberately plain (a dict of sklearn estimators plus
metadata) so it can be inspected with joblib and so a future model format can
replace it without changing the scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .domain import ModelPrediction
from .features import FEATURES, historical_features

FEATURE_SET = "market_only_v1"
MODEL_VERSION_KEY = "trained_at"


@dataclass
class MarketModel:
    """A fitted pair of classifiers and the exact columns they consume."""

    logreg: Any
    tree: Any
    features: list[str]
    metadata: dict[str, Any]

    @property
    def version(self) -> str:
        return str(self.metadata.get(MODEL_VERSION_KEY, "unknown"))

    def predict(self, features: dict[str, float]) -> ModelPrediction:
        """Return both probabilities and both binary calls for one row."""
        missing = [name for name in self.features if name not in features]
        if missing:
            raise ValueError(f"missing model features: {missing}")
        frame = pd.DataFrame(
            [{name: float(features[name]) for name in self.features}],
            columns=self.features,
        )
        logreg_prob = float(self.logreg.predict_proba(frame)[0, 1])
        tree_prob = float(self.tree.predict_proba(frame)[0, 1])
        return ModelPrediction(
            logreg_prob=logreg_prob,
            tree_prob=tree_prob,
            logreg_pick=bool(self.logreg.predict(frame)[0]),
            tree_pick=bool(self.tree.predict(frame)[0]),
            version=self.version,
        )


def train_market_model(
    model_path: str | Path,
    data: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Fit both models on historical market-average features and persist them.

    ``data`` is injectable for tests; by default the historical CSVs are loaded
    through :func:`simulator.load_matches`.
    """
    if data is None:
        from simulator import load_matches

        data = load_matches()

    frame = historical_features(data)
    required = FEATURES + ["target"]
    train = frame.dropna(subset=required).copy()
    if train.empty:
        raise ValueError("no complete historical rows to train on")
    if train["target"].nunique() < 2:
        raise ValueError("training data must contain both target classes")

    logreg = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(random_state=42, max_iter=1000),
            ),
        ]
    ).fit(train[FEATURES], train["target"].astype(int))
    tree = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=50, random_state=42
    ).fit(train[FEATURES], train["target"].astype(int))

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = {
        MODEL_VERSION_KEY: trained_at,
        "feature_set": FEATURE_SET,
        "rows": int(len(train)),
        "date_min": str(train["date"].min()),
        "date_max": str(train["date"].max()),
    }
    bundle = {
        "logreg": logreg,
        "tree": tree,
        "features": list(FEATURES),
        "metadata": metadata,
    }
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return metadata


def load_market_model(model_path: str | Path) -> MarketModel:
    """Load and validate a model artifact saved by :func:`train_market_model`."""
    bundle = joblib.load(Path(model_path))
    if not isinstance(bundle, dict):
        raise ValueError("model artifact is not a dict bundle")
    missing = {"logreg", "tree", "features", "metadata"} - set(bundle)
    if missing:
        raise ValueError(f"model artifact missing keys: {sorted(missing)}")
    if list(bundle["features"]) != list(FEATURES):
        raise ValueError(
            "model artifact was trained on a different feature set"
        )
    return MarketModel(
        logreg=bundle["logreg"],
        tree=bundle["tree"],
        features=list(bundle["features"]),
        metadata=dict(bundle["metadata"]),
    )
