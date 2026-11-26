"""Configuration for the paper-trading scheduler.

Settings are read from environment variables (or a ``.env`` file for local
runs).  Secrets are marked ``repr=False`` so they never appear in logs or in
an accidental ``print(settings)``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Mapping

DEFAULT_DATA_DIR = Path("data/paper")
DEFAULT_LEAGUE = "La Liga"
DEFAULT_LEAGUE_ID = 3
DEFAULT_TIMEZONE = "Europe/Madrid"


def load_env_file(path: str | Path = ".env") -> None:
    """Load ``KEY=VALUE`` lines into ``os.environ`` without overriding.

    The project uses keys containing hyphens (``BZZOIRO-API-KEY``), which are
    valid environment variable names even though they are not valid Python
    identifiers.  Existing environment variables always win, so a value from
    the shell or Docker ``env_file`` is never clobbered by a stale local file.
    """
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    """Everything the scheduler needs that is not a provider or a model."""

    db_path: Path = DEFAULT_DATA_DIR / "ledger.sqlite"
    model_path: Path = DEFAULT_DATA_DIR / "model.joblib"
    log_path: Path = DEFAULT_DATA_DIR / "scheduler.log"
    league: str = DEFAULT_LEAGUE
    league_id: int = DEFAULT_LEAGUE_ID
    timezone: str = DEFAULT_TIMEZONE
    starting_capital: float = 100.0
    decision_lead_minutes: int = 30
    safety_margin_minutes: int = 10
    max_staleness_minutes: int = 45
    fallback_stake_multiplier: float = 0.50
    tick_seconds: int = 30
    retry_interval_seconds: int = 120
    settlement_interval_seconds: int = 300
    fixtures_refresh_minutes: int = 360
    bzzoiro_base_url: str = "https://sports.bzzoiro.com"
    moviodds_base_url: str = "https://moviodds.com"
    moviodds_league: str = "Spain La Liga"
    webhook_url: str = field(default="", repr=False)
    bzzoiro_api_key: str = field(default="", repr=False)
    moviodds_api_key: str = field(default="", repr=False)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from an environment mapping.

        ``env`` is injectable for tests.  Defaults are intentionally safe for
        paper trading: a 30-minute decision lead, a 10-minute cutoff, and a
        50% stake haircut for Bet365 fallback decisions.
        """
        values = os.environ if env is None else env

        def text(key: str, default: str = "") -> str:
            return str(values.get(key, default) or default).strip()

        def number(key: str, default, cast):
            raw = values.get(key)
            if raw is None or str(raw).strip() == "":
                return default
            try:
                return cast(str(raw).strip())
            except (TypeError, ValueError):
                return default

        data_dir = Path(text("PAPER_DATA_DIR", str(DEFAULT_DATA_DIR)))
        return cls(
            db_path=Path(text("PAPER_DB", str(data_dir / "ledger.sqlite"))),
            model_path=Path(text("PAPER_MODEL", str(data_dir / "model.joblib"))),
            log_path=Path(text("PAPER_LOG", str(data_dir / "scheduler.log"))),
            league=text("PAPER_LEAGUE", DEFAULT_LEAGUE),
            league_id=number("PAPER_LEAGUE_ID", DEFAULT_LEAGUE_ID, int),
            timezone=text("TZ", DEFAULT_TIMEZONE),
            starting_capital=number(
                "PAPER_STARTING_CAPITAL", 100.0, float
            ),
            decision_lead_minutes=number(
                "PAPER_DECISION_LEAD_MINUTES", 30, int
            ),
            safety_margin_minutes=number(
                "PAPER_SAFETY_MARGIN_MINUTES", 10, int
            ),
            max_staleness_minutes=number(
                "PAPER_MAX_STALENESS_MINUTES", 45, int
            ),
            fallback_stake_multiplier=number(
                "PAPER_FALLBACK_STAKE_MULTIPLIER", 0.50, float
            ),
            tick_seconds=number("PAPER_TICK_SECONDS", 30, int),
            retry_interval_seconds=number(
                "PAPER_RETRY_INTERVAL_SECONDS", 120, int
            ),
            settlement_interval_seconds=number(
                "PAPER_SETTLEMENT_INTERVAL_SECONDS", 300, int
            ),
            fixtures_refresh_minutes=number(
                "PAPER_FIXTURES_REFRESH_MINUTES", 360, int
            ),
            bzzoiro_base_url=text(
                "BZZOIRO_BASE_URL", "https://sports.bzzoiro.com"
            ),
            moviodds_base_url=text(
                "MOVIODDS_BASE_URL", "https://moviodds.com"
            ),
            moviodds_league=text(
                "PAPER_MOVIODDS_LEAGUE", "Spain La Liga"
            ),
            webhook_url=text("DISCORD_WEBHOOK_URL"),
            bzzoiro_api_key=(
                text("BZZOIRO-API-KEY") or text("BZZOIRO_API_KEY")
            ),
            moviodds_api_key=(
                text("ODDS-API-KEY") or text("ODDS_API_KEY")
            ),
        )

    @property
    def decision_lead(self) -> timedelta:
        return timedelta(minutes=self.decision_lead_minutes)

    @property
    def safety_margin(self) -> timedelta:
        return timedelta(minutes=self.safety_margin_minutes)
