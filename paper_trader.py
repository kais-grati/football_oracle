"""Compatibility entry point for the paper-trading CLI.

The implementation lives in :mod:`papertrader`.  Run this wrapper exactly as
before from the project root::

    python paper_trader.py train
    python paper_trader.py tick
    python paper_trader.py run --interval 30
    python paper_trader.py stats
"""

from __future__ import annotations

from papertrader.cli import main

if __name__ == "__main__":
    main()
