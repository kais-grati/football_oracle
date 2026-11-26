"""Per-fixture paper-trading engine.

The package turns the research model into a small, testable production loop:

    fixtures -> T-30m decision -> consensus snapshot
             -> Bet365 fallback if consensus is unusable
             -> model -> wallet sizing -> ledger -> Discord

The public entry point is :mod:`papertrader.cli` (also exposed through the
``paper_trader.py`` wrapper).  The modules in this package are deliberately
small and dependency-injected so the scheduler can be replayed with a fake
clock and fake providers.
"""

__all__ = ["__version__"]
__version__ = "1.0.0"
