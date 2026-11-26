"""Wallet definitions and the stake policy.

All wallets share the same selection; only the pseudo-linear sizing profile
differs.  The profiles are the six levels used in the research notebooks and
in the existing paper trader.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from simulator import PseudoLinearSizer

WALLET_PROFILES = {
    "Conservative": PseudoLinearSizer(0.65, 0.01, 0.10, max_fraction=0.02),
    "Moderate": PseudoLinearSizer(0.65, 0.02, 0.25, max_fraction=0.05),
    "Medium": PseudoLinearSizer(0.65, 0.04, 0.50, max_fraction=0.10),
    "Medium-High": PseudoLinearSizer(0.65, 0.06, 0.60, max_fraction=0.20),
    "High": PseudoLinearSizer(0.65, 0.08, 0.90, max_fraction=0.30),
    "Aggressive": PseudoLinearSizer(0.65, 0.10, 2.00),
}

WALLET_NAMES = tuple(WALLET_PROFILES)


def stake_for(
    wallet: str,
    confidence: float,
    available: float,
    fallback_multiplier: float = 1.0,
) -> float:
    """Size one stake against the wallet's available bankroll.

    ``available`` is ``bankroll - pending exposure``.  Fallback decisions use
    the same sizer but scale the result, then the stake is clipped to what is
    actually available.  Invalid/non-positive results become zero (no bet).
    """
    if wallet not in WALLET_PROFILES:
        raise KeyError(f"unknown wallet: {wallet}")
    if not np.isfinite(available) or available <= 0:
        return 0.0
    if not np.isfinite(confidence):
        return 0.0
    sizer = WALLET_PROFILES[wallet]
    raw = sizer.stake(available, SimpleNamespace(confidence=float(confidence)))
    multiplier = float(fallback_multiplier)
    if not np.isfinite(raw) or not np.isfinite(multiplier) or multiplier <= 0:
        return 0.0
    stake = raw * multiplier
    return float(np.clip(stake, 0.0, available))
