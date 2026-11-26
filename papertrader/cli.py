"""Command-line entry points for the paper trader.

Commands
--------
    train                  fit and save the market-only model
    tick                   run one scheduler tick
    run                    run the scheduler loop (default 30s)
    fixtures               refresh today's fixtures once
    settle                 settle pending bets from provider results
    stats                  per-wallet and per-source performance
    status                 next decisions, pending exposure, last sync
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from . import db
from .config import Settings, load_env_file
from .domain import iso_utc, parse_utc
from .engine import DecisionEngine
from .model import load_market_model, train_market_model
from .notifier import DiscordNotifier
from .providers.bzzoiro import BzzoiroClient, BzzoiroFixtureProvider
from .providers.moviodds import MoviOddsBet365Provider, MoviOddsClient
from .scheduler import Scheduler, SystemClock, SYNC_DATE_KEY, SYNC_LAST_KEY
from .settlement import SettlementService
from .stats import (
    all_stats,
    pending_table,
    source_stats,
    status_report,
    wallet_stats,
)


def configure_logging(path: Path) -> logging.Logger:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("papertrader")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.FileHandler(path),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def _settings(args) -> Settings:
    settings = Settings.from_env()
    overrides = {}
    if getattr(args, "db", None):
        overrides["db_path"] = Path(args.db)
    if getattr(args, "model", None):
        overrides["model_path"] = Path(args.model)
    if getattr(args, "capital", None) is not None:
        overrides["starting_capital"] = float(args.capital)
    if getattr(args, "timezone", None):
        overrides["timezone"] = args.timezone
    return replace(settings, **overrides) if overrides else settings


def _require(settings: Settings) -> None:
    missing = []
    if not settings.bzzoiro_api_key:
        missing.append("BZZOIRO-API-KEY")
    if not settings.moviodds_api_key:
        missing.append("ODDS-API-KEY")
    if missing:
        raise SystemExit(
            "Missing provider API key(s): " + ", ".join(missing)
        )
    if not settings.model_path.exists():
        raise SystemExit(
            f"Model not found: {settings.model_path}. Run 'paper_trader.py "
            "train' first."
        )


def build_scheduler(settings: Settings):
    """Wire real providers, model, ledger, and scheduler."""
    _require(settings)
    conn = db.connect(settings.db_path, settings.starting_capital)
    bzzoiro = BzzoiroClient(
        settings.bzzoiro_api_key,
        base_url=settings.bzzoiro_base_url,
    )
    fixtures = BzzoiroFixtureProvider(
        bzzoiro,
        league=settings.league,
        league_id=settings.league_id,
        timezone=settings.timezone,
    )
    bet365 = MoviOddsBet365Provider(
        MoviOddsClient(
            settings.moviodds_api_key,
            base_url=settings.moviodds_base_url,
        ),
        board_league=settings.moviodds_league,
    )
    model = load_market_model(settings.model_path)
    notifier = DiscordNotifier(settings.webhook_url)
    engine = DecisionEngine(conn, model, fixtures, settings, notifier)
    settlement = SettlementService(conn, fixtures)
    scheduler = Scheduler(
        settings, conn, engine, fixtures, bet365, settlement
    )
    return scheduler, conn


def cmd_train(args) -> None:
    settings = _settings(args)
    model_path = Path(args.model) if args.model else settings.model_path
    metadata = train_market_model(model_path)
    print(f"Model saved to {model_path}")
    print(json.dumps(metadata, indent=2))


def cmd_run(args) -> None:
    settings = _settings(args)
    configure_logging(settings.log_path)
    scheduler, conn = build_scheduler(settings)
    try:
        scheduler.run(once=args.once, interval_seconds=args.interval)
    finally:
        conn.close()


def cmd_tick(args) -> None:
    settings = _settings(args)
    configure_logging(settings.log_path)
    scheduler, conn = build_scheduler(settings)
    try:
        now = parse_utc(args.now) if args.now else SystemClock().now()
        if now is None:
            raise SystemExit("--now must be an ISO-8601 timestamp")
        result = scheduler.tick(now)
        print(json.dumps({
            "now": result.now.isoformat(timespec="seconds"),
            "due": result.due_fixtures,
            "deferred": result.deferred,
            "placed": result.processed,
            "retries": result.retries,
            "skipped": result.skipped,
            "missed": result.missed,
            "settled": result.settled,
            "errors": result.errors,
        }, indent=2, default=str))
    finally:
        conn.close()


def cmd_fixtures(args) -> None:
    settings = _settings(args)
    configure_logging(settings.log_path)
    if not settings.bzzoiro_api_key:
        raise SystemExit("Missing provider API key: BZZOIRO-API-KEY")
    now = parse_utc(args.now) if args.now else SystemClock().now()
    if now is None:
        raise SystemExit("--now must be an ISO-8601 timestamp")
    local_today = now.astimezone(
        ZoneInfo(settings.timezone)
    ).date()
    conn = db.connect(settings.db_path, settings.starting_capital)
    try:
        client = BzzoiroClient(
            settings.bzzoiro_api_key,
            base_url=settings.bzzoiro_base_url,
        )
        provider = BzzoiroFixtureProvider(
            client,
            league=settings.league,
            league_id=settings.league_id,
            timezone=settings.timezone,
        )
        fixtures = provider.fetch_fixtures(
            local_today - timedelta(days=1),
            local_today,
            include_all=True,
        )
        count = db.upsert_fixtures(conn, fixtures, updated_at=now)
        db.set_meta(conn, SYNC_LAST_KEY, iso_utc(now))
        db.set_meta(conn, SYNC_DATE_KEY, local_today.isoformat())
        print(json.dumps({"synced": count, "now": now.isoformat()},
                         indent=2))
    finally:
        conn.close()


def cmd_settle(args) -> None:
    settings = _settings(args)
    configure_logging(settings.log_path)
    if not settings.bzzoiro_api_key:
        raise SystemExit("Missing provider API key: BZZOIRO-API-KEY")
    conn = db.connect(settings.db_path, settings.starting_capital)
    try:
        client = BzzoiroClient(
            settings.bzzoiro_api_key,
            base_url=settings.bzzoiro_base_url,
        )
        provider = BzzoiroFixtureProvider(
            client,
            league=settings.league,
            league_id=settings.league_id,
            timezone=settings.timezone,
        )
        summary = SettlementService(conn, provider).settle_pending()
        print(json.dumps(summary, indent=2))
    finally:
        conn.close()


def cmd_stats(args) -> None:
    settings = _settings(args)
    conn = db.connect(settings.db_path, settings.starting_capital)
    try:
        if args.wallet:
            wallet = db.get_wallet(conn, args.wallet)
            if wallet is None:
                raise SystemExit(f"Unknown wallet: {args.wallet}")
            stats, pending, settled = wallet_stats(conn, wallet)
            if args.json:
                print(json.dumps(stats, indent=2, default=str))
            else:
                print(f"Wallet: {stats['wallet']}")
                print(json.dumps(stats, indent=2, default=str))
                if not pending.empty:
                    print("\nPending")
                    print(pending[[
                        "kickoff", "home_team", "away_team", "selection",
                        "odds_taken", "stake", "confidence",
                    ]].round(2).to_string(index=False))
                if not settled.empty:
                    print("\nRecent settled")
                    print(settled[[
                        "kickoff", "home_team", "away_team", "selection",
                        "odds_taken", "stake", "status", "profit",
                    ]].tail(10).round(2).to_string(index=False))
            return

        table = all_stats(conn)
        sources = source_stats(conn)
        pending = pending_table(conn)
        if args.json:
            print(json.dumps({
                "wallets": table.to_dict(orient="records"),
                "sources": sources.to_dict(orient="records"),
                "pending": pending.to_dict(orient="records")
                if not pending.empty else [],
            }, indent=2, default=str))
            return
        print("Paper wallets — same selections, different sizers")
        print(table.round(2).to_string(index=False))
        if not sources.empty:
            print("\nBy price source")
            print(sources.round(2).to_string(index=False))
    finally:
        conn.close()


def cmd_status(args) -> None:
    settings = _settings(args)
    conn = db.connect(settings.db_path, settings.starting_capital)
    try:
        report = status_report(
            conn, lead_minutes=settings.decision_lead_minutes
        )
        report["pending_rows"] = pending_table(conn).to_dict(orient="records")
        print(json.dumps(report, indent=2, default=str))
    finally:
        conn.close()


def _add_common(parser, *, model: bool = True) -> None:
    parser.add_argument("--db", default=None)
    if model:
        parser.add_argument("--model", default=None)
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--timezone", default=None)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Per-fixture paper trader with T-30m decisions."
    )
    parser.add_argument(
        "--env", default=".env",
        help="Optional .env file (existing environment wins)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Fit and save the model.")
    p_train.add_argument("--model", default=None)
    p_train.set_defaults(func=cmd_train)

    p_run = sub.add_parser("run", help="Run the scheduler loop.")
    _add_common(p_run)
    p_run.add_argument("--interval", type=int, default=None,
                       help="Tick interval in seconds (default 30)")
    p_run.add_argument("--once", action="store_true",
                       help="Run one tick and exit")
    p_run.set_defaults(func=cmd_run)

    p_tick = sub.add_parser("tick", help="Run exactly one tick.")
    _add_common(p_tick)
    p_tick.add_argument("--now", help="Override the clock (ISO-8601)")
    p_tick.set_defaults(func=cmd_tick)

    p_fixtures = sub.add_parser("fixtures", help="Refresh fixtures once.")
    _add_common(p_fixtures)
    p_fixtures.add_argument("--now", help="Override the clock (ISO-8601)")
    p_fixtures.set_defaults(func=cmd_fixtures)

    p_settle = sub.add_parser("settle", help="Settle pending bets.")
    _add_common(p_settle)
    p_settle.set_defaults(func=cmd_settle)

    p_stats = sub.add_parser("stats", help="Show wallet statistics.")
    _add_common(p_stats, model=False)
    p_stats.add_argument("--wallet", help="Show one wallet")
    p_stats.add_argument("--json", action="store_true")
    p_stats.set_defaults(func=cmd_stats)

    p_status = sub.add_parser("status", help="Show scheduler status.")
    _add_common(p_status, model=False)
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    load_env_file(args.env)
    args.func(args)


if __name__ == "__main__":
    main()
