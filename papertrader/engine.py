"""The T-30m decision engine.

The engine owns one fixture at a time.  It is deliberately synchronous and
side-effect explicit: providers in, ledger out.  The scheduler supplies the
Bet365 board for the due fixtures so one HTTP request can serve several
matches with the same kickoff time.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime

from . import db
from .config import Settings
from .domain import (
    DecisionOutcome,
    Fixture,
    ModelPrediction,
    OddsSnapshot,
    SnapshotAttempt,
    ensure_utc,
    iso_utc,
)
from .features import (
    FeatureError,
    build_live_features,
    favorite_confidence,
    market_favorite,
)
from .notifier import DiscordNotifier
from .sizing import stake_for

logger = logging.getLogger(__name__)

CONSENSUS_SOURCE = "consensus"
FALLBACK_SOURCE = "bet365_fallback"
NO_SOURCE = "none"


class DecisionEngine:
    """Decide, size, record, and notify for one fixture."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        model,
        consensus_provider,
        settings: Settings,
        notifier: DiscordNotifier | None = None,
        clock=None,
    ):
        self.conn = conn
        self.model = model
        self.consensus = consensus_provider
        self.settings = settings
        self.notifier = notifier or DiscordNotifier(settings.webhook_url)
        self.clock = clock

    def _effective_now(self, now: datetime) -> datetime:
        """Use the wall clock if a network call made ``now`` stale."""
        if self.clock is None:
            return now
        return max(now, ensure_utc(self.clock.now()))

    # -- public API --------------------------------------------------------

    def process(
        self,
        fixture: Fixture,
        bet365_snapshots: dict[str, OddsSnapshot],
        now: datetime,
    ) -> DecisionOutcome:
        now = self._effective_now(ensure_utc(now))
        existing = db.get_decision(self.conn, fixture.event_id)
        if existing is not None:
            return DecisionOutcome(
                fixture_id=fixture.event_id,
                status="already_decided",
                decision_id=int(existing["id"]),
                odds_source=str(existing["odds_source"]),
                recorded=False,
            )

        if fixture.kickoff_utc <= now:
            return self._skip(
                fixture,
                now,
                "after_kickoff",
                reason="kickoff has passed",
            )
        if now >= fixture.kickoff_utc - self.settings.safety_margin:
            return self._skip(
                fixture,
                now,
                "outside_cutoff",
                reason="inside safety margin before kickoff",
            )

        execution = bet365_snapshots.get(fixture.event_id)
        consensus_attempt = self._fetch_consensus(fixture)
        consensus_snapshot, consensus_reason = self._validate_consensus(
            consensus_attempt, now
        )
        fallback_reason = None
        feature_snapshot: OddsSnapshot | None = None
        odds_source = NO_SOURCE

        if consensus_snapshot is not None:
            feature_snapshot = consensus_snapshot
            odds_source = CONSENSUS_SOURCE
        elif execution is not None and execution.is_complete():
            fallback_fresh, fallback_reason = self._freshness(execution, now)
            if fallback_fresh:
                feature_snapshot = execution
                odds_source = FALLBACK_SOURCE
                fallback_reason = consensus_reason or "consensus_unavailable"
            else:
                fallback_reason = fallback_reason or "fallback_stale"
        else:
            fallback_reason = consensus_reason or "consensus_unavailable"

        if feature_snapshot is None:
            reason = fallback_reason or "no_snapshot"
            return self._retry_or_skip(
                fixture,
                now,
                reason=reason,
                consensus_attempt=consensus_attempt,
                execution=execution,
                odds_source=NO_SOURCE,
            )

        if execution is None or not execution.is_complete():
            return self._retry_or_skip(
                fixture,
                now,
                reason="execution_unavailable",
                consensus_attempt=consensus_attempt,
                execution=execution,
                odds_source=odds_source,
                feature_snapshot=feature_snapshot,
                fallback_reason=fallback_reason,
            )
        execution_fresh, execution_reason = self._freshness(execution, now)
        if not execution_fresh:
            return self._retry_or_skip(
                fixture,
                now,
                reason=f"execution_{execution_reason}",
                consensus_attempt=consensus_attempt,
                execution=execution,
                odds_source=odds_source,
                feature_snapshot=feature_snapshot,
                fallback_reason=fallback_reason,
            )

        selection = market_favorite(
            feature_snapshot.home, feature_snapshot.draw, feature_snapshot.away
        )
        execution_odds = execution.price(selection)
        if execution_odds is None or execution_odds <= 1.0:
            return self._retry_or_skip(
                fixture,
                now,
                reason="execution_selection_unavailable",
                consensus_attempt=consensus_attempt,
                execution=execution,
                odds_source=odds_source,
                feature_snapshot=feature_snapshot,
                fallback_reason=fallback_reason,
            )

        try:
            features = build_live_features(
                fixture,
                feature_snapshot,
                season_start=fixture.season_start,
                timezone=self.settings.timezone,
            )
            confidence = favorite_confidence(
                feature_snapshot.home,
                feature_snapshot.draw,
                feature_snapshot.away,
            )
            prediction = self.model.predict(features)
        except (FeatureError, ValueError, KeyError) as exc:
            return self._retry_or_skip(
                fixture,
                now,
                reason=f"feature_or_model_error: {exc}",
                consensus_attempt=consensus_attempt,
                execution=execution,
                odds_source=odds_source,
                feature_snapshot=feature_snapshot,
                fallback_reason=fallback_reason,
            )

        if not prediction.agrees:
            return self._record(
                fixture,
                now,
                status="no_bet",
                odds_source=odds_source,
                selection=selection,
                execution_odds=float(execution_odds),
                confidence=confidence,
                prediction=prediction,
                feature_snapshot=feature_snapshot,
                consensus_attempt=consensus_attempt,
                execution=execution,
                fallback_reason=fallback_reason,
                reason="models_disagree",
                bets=[],
            )

        # A slow provider call must not let a decision slip past the cutoff.
        now = self._effective_now(now)
        if now >= fixture.kickoff_utc - self.settings.safety_margin:
            return self._skip(
                fixture,
                now,
                "outside_cutoff",
                reason="provider calls consumed the decision window",
            )

        bets, notifier_bets = self._size_wallets(
            fixture,
            selection,
            execution_odds,
            confidence,
            prediction,
            now,
            fallback=odds_source == FALLBACK_SOURCE,
        )
        status = "placed" if bets else "no_bet"
        reason = None if bets else "no_stake"
        outcome = self._record(
            fixture,
            now,
            status=status,
            odds_source=odds_source,
            selection=selection,
            execution_odds=float(execution_odds),
            confidence=confidence,
            prediction=prediction,
            feature_snapshot=feature_snapshot,
            consensus_attempt=consensus_attempt,
            execution=execution,
            fallback_reason=fallback_reason,
            reason=reason,
            bets=bets,
        )

        if outcome.recorded and bets:
            self._notify(
                fixture,
                now,
                selection,
                execution_odds,
                confidence,
                prediction,
                feature_snapshot,
                execution,
                odds_source,
                fallback_reason,
                notifier_bets,
                outcome.decision_id,
            )
        return outcome

    # -- snapshots ---------------------------------------------------------

    def _fetch_consensus(self, fixture: Fixture) -> SnapshotAttempt:
        try:
            attempt = self.consensus.fetch_consensus(fixture.event_id)
        except Exception as exc:  # provider isolation is the point
            logger.warning(
                "consensus fetch failed for %s: %s", fixture.event_id, exc
            )
            return SnapshotAttempt(None, f"provider_error: {exc}")
        if not isinstance(attempt, SnapshotAttempt):
            return SnapshotAttempt(None, "provider_returned_invalid_attempt")
        return attempt

    def _validate_consensus(
        self, attempt: SnapshotAttempt, now: datetime
    ) -> tuple[OddsSnapshot | None, str | None]:
        if attempt is None or attempt.snapshot is None:
            return None, attempt.reason if attempt else "no_snapshot"
        snapshot = attempt.snapshot
        if not snapshot.is_complete():
            return None, "incomplete_outcomes"
        fresh, reason = self._freshness(snapshot, now)
        if not fresh:
            return None, reason
        return snapshot, attempt.reason

    def _freshness(
        self, snapshot: OddsSnapshot, now: datetime
    ) -> tuple[bool, str | None]:
        if snapshot.observed_at is None:
            return False, "missing_snapshot_time"
        age = (now - snapshot.observed_at).total_seconds()
        if age < -300:
            return False, "snapshot_in_future"
        if age > self.settings.max_staleness_minutes * 60:
            return False, f"stale_snapshot:{int(age)}s"
        return True, None

    # -- sizing and ledger -------------------------------------------------

    def _size_wallets(
        self,
        fixture: Fixture,
        selection: str,
        execution_odds: float,
        confidence: float,
        prediction: ModelPrediction,
        now: datetime,
        fallback: bool,
    ) -> tuple[list[dict], list[dict]]:
        wallets = db.wallet_rows(self.conn)
        available = {w["name"]: float(w["available"]) for w in wallets}
        existing_wallets = {
            int(row["wallet_id"])
            for row in self.conn.execute(
                "SELECT wallet_id FROM bets WHERE match_id = ?",
                (fixture.event_id,),
            )
        }
        multiplier = (
            self.settings.fallback_stake_multiplier if fallback else 1.0
        )
        kickoff_label = fixture.local_kickoff(
            self.settings.timezone
        ).isoformat(timespec="minutes")
        bets: list[dict] = []
        notifier_bets: list[dict] = []

        for wallet in wallets:
            if int(wallet["id"]) in existing_wallets:
                # A restart or a partial previous write must never duplicate
                # a wallet's bet; only notify for genuinely new bets.
                continue
            name = wallet["name"]
            before = available[name]
            if before <= 0:
                continue
            stake = stake_for(name, confidence, before, multiplier)
            if stake <= 0:
                continue
            stake_pct = stake / before * 100
            bets.append({
                "wallet_id": int(wallet["id"]),
                "created_at": iso_utc(now),
                "match_id": fixture.event_id,
                "league": fixture.league,
                "kickoff": kickoff_label,
                "home_team": fixture.home_team,
                "away_team": fixture.away_team,
                "selection": selection,
                "odds_taken": float(execution_odds),
                "bookmaker": "Bet365",
                "confidence": float(confidence),
                "p_logreg": float(prediction.logreg_prob),
                "p_tree": float(prediction.tree_prob),
                "stake": float(stake),
                "status": "pending",
                "model_version": prediction.version,
                "discord_sent": 0,
            })
            notifier_bets.append({
                "wallet": name,
                "stake": float(stake),
                "stake_pct": float(stake_pct),
            })
            available[name] = before - stake
        return bets, notifier_bets

    def _record(
        self,
        fixture: Fixture,
        now: datetime,
        *,
        status: str,
        odds_source: str,
        selection: str | None,
        execution_odds: float | None,
        confidence: float | None,
        prediction: ModelPrediction | None,
        feature_snapshot: OddsSnapshot | None,
        consensus_attempt: SnapshotAttempt | None,
        execution: OddsSnapshot | None,
        fallback_reason: str | None,
        reason: str | None,
        bets: list[dict],
    ) -> DecisionOutcome:
        decision = self._decision_row(
            fixture=fixture,
            now=now,
            status=status,
            odds_source=odds_source,
            selection=selection,
            execution_odds=execution_odds,
            confidence=confidence,
            prediction=prediction,
            feature_snapshot=feature_snapshot,
            consensus_attempt=consensus_attempt,
            execution=execution,
            fallback_reason=fallback_reason,
            reason=reason,
        )
        decision_id = db.insert_decision_and_bets(self.conn, decision, bets)
        if decision_id is None:
            return DecisionOutcome(
                fixture_id=fixture.event_id,
                status="already_decided",
                odds_source=odds_source,
                selection=selection,
                recorded=False,
            )
        try:
            self._archive(fixture.event_id, consensus_attempt, execution)
        except Exception:
            # The decision and bets are already committed.  A failed audit
            # write must not roll back or hide a live paper bet.
            logger.exception("snapshot archive failed for %s", fixture.event_id)
        return DecisionOutcome(
            fixture_id=fixture.event_id,
            status=status,
            decision_id=decision_id,
            odds_source=odds_source,
            selection=selection,
            reason=reason,
            fallback_reason=fallback_reason,
            bets=bets,
            recorded=True,
        )

    def _decision_row(
        self,
        *,
        fixture: Fixture,
        now: datetime,
        status: str,
        odds_source: str,
        selection: str | None,
        execution_odds: float | None,
        confidence: float | None,
        prediction: ModelPrediction | None,
        feature_snapshot: OddsSnapshot | None,
        consensus_attempt: SnapshotAttempt | None,
        execution: OddsSnapshot | None,
        fallback_reason: str | None,
        reason: str | None,
    ) -> dict:
        consensus_snapshot = (
            consensus_attempt.snapshot if consensus_attempt else None
        )
        staleness = None
        if feature_snapshot is not None and feature_snapshot.observed_at:
            staleness = int(
                (now - feature_snapshot.observed_at).total_seconds()
            )
        raw_consensus = None
        if consensus_snapshot is not None:
            raw_consensus = consensus_snapshot.raw
        elif consensus_attempt is not None:
            raw_consensus = consensus_attempt.raw
        return {
            "fixture_id": fixture.event_id,
            "created_at": iso_utc(now),
            "decision_at_utc": iso_utc(
                fixture.decision_at(self.settings.decision_lead_minutes)
            ),
            "actual_lead_seconds": int(
                (fixture.kickoff_utc - now).total_seconds()
            ),
            "odds_source": odds_source,
            "consensus_attempted_at": iso_utc(now),
            "consensus_snapshot_at": iso_utc(
                consensus_snapshot.observed_at
            ) if consensus_snapshot else None,
            "fallback_snapshot_at": iso_utc(execution.observed_at)
            if odds_source == FALLBACK_SOURCE and execution else None,
            "fallback_reason": fallback_reason
            if odds_source == FALLBACK_SOURCE else None,
            "snapshot_at": iso_utc(feature_snapshot.observed_at)
            if feature_snapshot else None,
            "staleness_seconds": staleness,
            "model_version": prediction.version if prediction else None,
            "selection": selection,
            "home_odds": feature_snapshot.home if feature_snapshot else None,
            "draw_odds": feature_snapshot.draw if feature_snapshot else None,
            "away_odds": feature_snapshot.away if feature_snapshot else None,
            "execution_odds": execution_odds,
            "confidence": confidence,
            "p_logreg": prediction.logreg_prob if prediction else None,
            "p_tree": prediction.tree_prob if prediction else None,
            "status": status,
            "reason": reason,
            "raw_consensus": raw_consensus,
            "raw_bet365": execution.raw if execution else None,
        }

    def _archive(
        self,
        fixture_id: str,
        consensus_attempt: SnapshotAttempt | None,
        execution: OddsSnapshot | None,
    ) -> None:
        if consensus_attempt and consensus_attempt.snapshot:
            db.insert_snapshot(
                self.conn, fixture_id, consensus_attempt.snapshot
            )
        if execution is not None:
            db.insert_snapshot(self.conn, fixture_id, execution)

    def _retry_or_skip(
        self,
        fixture: Fixture,
        now: datetime,
        *,
        reason: str,
        consensus_attempt: SnapshotAttempt | None = None,
        execution: OddsSnapshot | None = None,
        odds_source: str = NO_SOURCE,
        feature_snapshot: OddsSnapshot | None = None,
        fallback_reason: str | None = None,
    ) -> DecisionOutcome:
        cutoff = fixture.kickoff_utc - self.settings.safety_margin
        if now < cutoff:
            logger.info(
                "retry %s (%s); %ss to cutoff",
                fixture.event_id, reason,
                int((cutoff - now).total_seconds()),
            )
            return DecisionOutcome(
                fixture_id=fixture.event_id,
                status="retry",
                odds_source=odds_source,
                reason=reason,
                fallback_reason=fallback_reason,
                retry=True,
                recorded=False,
            )
        return self._record(
            fixture,
            now,
            status="skipped",
            odds_source=odds_source,
            selection=None,
            execution_odds=None,
            confidence=None,
            prediction=None,
            feature_snapshot=feature_snapshot,
            consensus_attempt=consensus_attempt,
            execution=execution,
            fallback_reason=fallback_reason,
            reason=reason,
            bets=[],
        )

    def _skip(
        self,
        fixture: Fixture,
        now: datetime,
        status: str,
        reason: str,
    ) -> DecisionOutcome:
        return self._record(
            fixture,
            now,
            status="skipped",
            odds_source=NO_SOURCE,
            selection=None,
            execution_odds=None,
            confidence=None,
            prediction=None,
            feature_snapshot=None,
            consensus_attempt=None,
            execution=None,
            fallback_reason=None,
            reason=f"{status}: {reason}",
            bets=[],
        )

    def _notify(
        self,
        fixture: Fixture,
        now: datetime,
        selection: str,
        execution_odds: float,
        confidence: float,
        prediction: ModelPrediction,
        feature_snapshot: OddsSnapshot,
        execution: OddsSnapshot,
        odds_source: str,
        fallback_reason: str | None,
        notifier_bets: list[dict],
        decision_id: int | None,
    ) -> None:
        if not notifier_bets:
            return
        message = {
            "match_id": fixture.event_id,
            "league": fixture.league,
            "home_team": fixture.home_team,
            "away_team": fixture.away_team,
            "kickoff": fixture.local_kickoff(
                self.settings.timezone
            ).strftime("%Y-%m-%d %H:%M"),
            "selection": selection,
            "odds_taken": float(execution_odds),
            "confidence": float(confidence),
            "p_logreg": float(prediction.logreg_prob),
            "p_tree": float(prediction.tree_prob),
            "odds_source": odds_source,
            "fallback_reason": fallback_reason,
            "snapshot_at": iso_utc(feature_snapshot.observed_at),
            "actual_lead_minutes": (
                fixture.kickoff_utc - now
            ).total_seconds() / 60.0,
        }
        sent = False
        try:
            sent = self.notifier.send_bets(message, notifier_bets)
        except Exception:
            logger.exception("notification failed for %s", fixture.event_id)
        if decision_id is not None:
            try:
                rows = self.conn.execute(
                    "SELECT id FROM bets WHERE decision_id = ?", (decision_id,)
                ).fetchall()
                db.update_discord_sent(
                    self.conn, [int(row["id"]) for row in rows], sent
                )
            except Exception:
                logger.exception(
                    "could not mark Discord delivery for %s", fixture.event_id
                )
