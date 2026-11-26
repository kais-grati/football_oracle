"""Long-running scheduler and one-shot tick.

Every tick settles finished bets, refreshes the fixture list when stale,
processes fixtures whose T-30m has arrived, and marks windows that closed
without a decision.  All external calls are isolated per component so one
provider failure never stops the rest of the matchday.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from . import db
from .config import Settings
from .domain import ensure_utc, iso_utc, parse_utc

logger = logging.getLogger(__name__)

SYNC_LAST_KEY = "fixtures_last_sync"
SYNC_DATE_KEY = "fixtures_last_sync_date"


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


@dataclass
class TickResult:
    """Summary of one scheduler tick, handy for logs and tests."""

    now: datetime
    settled: dict[str, Any] = field(default_factory=dict)
    settlement_deferred: bool = False
    synced_fixtures: int = 0
    due_fixtures: int = 0
    deferred: int = 0
    processed: int = 0
    skipped: int = 0
    retries: int = 0
    missed: int = 0
    errors: list[str] = field(default_factory=list)
    outcomes: list[Any] = field(default_factory=list)


class Scheduler:
    def __init__(
        self,
        settings: Settings,
        conn: sqlite3.Connection,
        engine,
        fixture_provider,
        bet365_provider,
        settlement,
        clock: SystemClock | None = None,
    ):
        self.settings = settings
        self.conn = conn
        self.engine = engine
        self.fixtures = fixture_provider
        self.bet365 = bet365_provider
        self.settlement = settlement
        self.clock = clock or SystemClock()
        if getattr(self.engine, "clock", None) is None:
            self.engine.clock = self.clock
        # A transient provider failure should not hammer the APIs every tick.
        # This is intentionally in-memory: after a restart an immediate retry
        # is safe because the ledger's unique keys prevent double decisions.
        self._last_attempt: dict[str, datetime] = {}
        self._last_settlement: datetime | None = None

    def tick(self, now: datetime | None = None) -> TickResult:
        explicit_now = now is not None
        current = ensure_utc(now or self.clock.now())
        result = TickResult(now=current)

        try:
            settlement_interval = timedelta(
                seconds=self.settings.settlement_interval_seconds
            )
            if (
                self._last_settlement is None
                or current - self._last_settlement >= settlement_interval
            ):
                self._last_settlement = current
                result.settled = self.settlement.settle_pending(current)
            else:
                result.settlement_deferred = True
        except Exception as exc:
            logger.exception("settlement failed")
            result.errors.append(f"settlement: {exc}")

        try:
            result.synced_fixtures = self.sync_fixtures(current)
        except Exception as exc:
            logger.exception("fixture sync failed")
            result.errors.append(f"fixture_sync: {exc}")

        due = db.due_fixtures(
            self.conn,
            current,
            self.settings.decision_lead_minutes,
            self.settings.safety_margin_minutes,
        )
        result.due_fixtures = len(due)
        candidates = []
        retry_interval = timedelta(
            seconds=self.settings.retry_interval_seconds
        )
        for fixture in due:
            last = self._last_attempt.get(fixture.event_id)
            if last is not None and current - last < retry_interval:
                result.deferred += 1
                continue
            candidates.append(fixture)

        if candidates:
            # Mark the attempt before any network call: a failed board fetch
            # also deserves the retry backoff.
            for fixture in candidates:
                self._last_attempt[fixture.event_id] = current
            board = {}
            try:
                fetched = self.bet365.fetch_board(candidates)
                board = fetched if isinstance(fetched, dict) else {}
                if not isinstance(fetched, dict):
                    result.errors.append("bet365_board: non-dict response")
            except Exception as exc:
                logger.exception("Bet365 board fetch failed")
                result.errors.append(f"bet365_board: {exc}")
            for fixture in candidates:
                try:
                    process_now = (
                        current if explicit_now
                        else max(current, ensure_utc(self.clock.now()))
                    )
                    outcome = self.engine.process(fixture, board, process_now)
                    result.outcomes.append(outcome)
                    if outcome.status == "placed":
                        result.processed += 1
                    elif outcome.status == "retry":
                        result.retries += 1
                    else:
                        result.skipped += 1
                except Exception as exc:
                    logger.exception(
                        "decision failed for %s", fixture.event_id
                    )
                    result.errors.append(
                        f"{fixture.event_id}: {exc}"
                    )

        try:
            result.missed = db.mark_missed(
                self.conn,
                current,
                self.settings.decision_lead_minutes,
                self.settings.safety_margin_minutes,
            )
        except Exception as exc:
            logger.exception("mark_missed failed")
            result.errors.append(f"mark_missed: {exc}")
        return result

    def sync_fixtures(self, now: datetime) -> int:
        """Refresh today's fixtures unless the schedule was just synced."""
        local_today = now.astimezone(
            ZoneInfo(self.settings.timezone)
        ).date()
        last = parse_utc(db.get_meta(self.conn, SYNC_LAST_KEY))
        last_date = db.get_meta(self.conn, SYNC_DATE_KEY)
        refresh = timedelta(minutes=self.settings.fixtures_refresh_minutes)
        if (
            last is not None
            and last_date == local_today.isoformat()
            and now - last < refresh
        ):
            return 0

        # Include the previous UTC date so a local after-midnight kickoff is
        # not missed when the provider filters by UTC event date.
        start = local_today - timedelta(days=1)
        fixtures = self.fixtures.fetch_fixtures(
            start, local_today, include_all=True
        )
        count = db.upsert_fixtures(self.conn, fixtures, updated_at=now)
        db.set_meta(self.conn, SYNC_LAST_KEY, iso_utc(now))
        db.set_meta(self.conn, SYNC_DATE_KEY, local_today.isoformat())
        logger.info(
            "synced %d fixtures for local date %s", count, local_today
        )
        return count

    def run(
        self,
        once: bool = False,
        interval_seconds: int | None = None,
    ) -> None:
        interval = (
            self.settings.tick_seconds
            if interval_seconds is None else int(interval_seconds)
        )
        interval = max(1, interval)
        logger.info(
            "scheduler started (tick=%ss, lead=%s, safety=%s)",
            interval,
            self.settings.decision_lead_minutes,
            self.settings.safety_margin_minutes,
        )
        while True:
            try:
                result = self.tick()
            except Exception:
                # A tick is isolated in normal operation; this catch keeps a
                # transient SQLite/network error from killing the container.
                logger.exception("scheduler tick crashed")
                if once:
                    raise
                time.sleep(interval)
                continue
            logger.info(
                "tick: due=%d deferred=%d placed=%d retry=%d skipped=%d "
                "missed=%d settled=%d errors=%d",
                result.due_fixtures,
                result.deferred,
                result.processed,
                result.retries,
                result.skipped,
                result.missed,
                int(result.settled.get("settled", 0)),
                len(result.errors),
            )
            if once:
                return
            time.sleep(interval)
