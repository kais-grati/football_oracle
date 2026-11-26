# Docker deployment on a VPS

## What runs

One long-running container runs the per-fixture scheduler:

```bash
python paper_trader.py run --interval 30
```

Every tick it:

1. settles finished bets (at most every `PAPER_SETTLEMENT_INTERVAL_SECONDS`),
2. refreshes today's La Liga fixtures every `PAPER_FIXTURES_REFRESH_MINUTES`,
3. decides each fixture at its own `kickoff_utc - PAPER_DECISION_LEAD_MINUTES`,
4. sizes six paper wallets and writes the SQLite ledger,
5. sends one Discord message per fixture with placed bets.

Persistent state lives in `./data/paper` on the host. This scheduler is
paper-only: it never places a real bet and never handles real money.

## Prerequisites

Docker Engine and the Compose plugin. From the repository root:

```bash
mkdir -p data/paper
sudo chown -R 1000:1000 data/paper
```

The UID/GID must match the values in `.env` (default `1000`). If your VPS user
is different, update `UID` and `GID` in `.env` and use the same values in the
`chown`.

## Configuration

Create `.env` from `.env.example` and set the Discord webhook, Bzzoiro key,
MoviOdds key, timezone, and UID/GID:

```bash
cp .env.example .env
chmod 600 .env
```

`.env` is gitignored, so it does not arrive with `git pull`. If you create it
locally, transfer it over a secure channel (for example `scp`); never commit
it.

Required keys:

- `BZZOIRO-API-KEY` — fixtures, results, and consensus 1X2.
- `ODDS-API-KEY` — MoviOdds Bet365 board for execution and fallback.
- `DISCORD_WEBHOOK_URL` — optional; without it the ledger still records bets.
- `TZ` — local calendar timezone; default `Europe/Madrid`.

## First run

```bash
docker compose build
docker compose run --rm paper-trader python paper_trader.py train
docker compose up -d
docker compose logs -f paper-trader
```

- `train` fits the market-only logreg + tree model and writes
  `data/paper/model.joblib` to the bind mount.
- `up` starts the scheduler. It runs a tick immediately and every
  `PAPER_TICK_SECONDS` (30s).
- If the model is missing, the container exits; train first.

## Decision timing

- `decision_at = kickoff_utc - 30m`.
- The scheduler accepts a decision while `kickoff_utc > now + 10m`.
- A missed T-30m (process restart, API outage) is decided late if it is still
  before the 10-minute cutoff; otherwise it is recorded as `skipped`.
- Transient provider failures retry after `PAPER_RETRY_INTERVAL_SECONDS` (120s).

Consensus is preferred. It is valid only when all three outcomes are present,
the snapshot is within `PAPER_MAX_STALENESS_MINUTES` (45m), and the odds are
sane. Otherwise the engine rebuilds the features from the Bet365 board, flags
`odds_source=bet365_fallback`, halves the stake by default
(`PAPER_FALLBACK_STAKE_MULTIPLIER=0.5`), and records the fallback reason. If
Bet365 is also unavailable, the decision is skipped.

## Operational commands

```bash
docker compose exec paper-trader python paper_trader.py status
docker compose exec paper-trader python paper_trader.py stats
docker compose exec paper-trader python paper_trader.py stats --wallet Conservative
docker compose exec paper-trader python paper_trader.py stats --json
docker compose exec paper-trader python paper_trader.py tick
docker compose exec paper-trader python paper_trader.py fixtures
docker compose exec paper-trader python paper_trader.py settle
docker compose logs -f paper-trader
docker compose restart
docker compose down
```

`stats` shows per-wallet performance and a separate table by `odds_source`
(`consensus` vs `bet365_fallback`). `status` shows the next decision times,
open exposure, and the last fixture sync.

`data/paper/` contains the ledger, model, log, and snapshots. Back it up when
the container is stopped.

## Updating

```bash
git pull
docker compose up -d --build
```

If the training CSVs changed, retrain and restart:

```bash
docker compose run --rm paper-trader python paper_trader.py train
docker compose restart
```

## Settings

- `TZ` controls the local calendar used by the model features; storage and
  comparisons are UTC.
- `PAPER_TICK_SECONDS=30`, `PAPER_RETRY_INTERVAL_SECONDS=120`,
  `PAPER_SETTLEMENT_INTERVAL_SECONDS=300`.
- `PAPER_FIXTURES_REFRESH_MINUTES=360`.
- `PAPER_MAX_STALENESS_MINUTES=45`, `PAPER_DECISION_LEAD_MINUTES=30`,
  `PAPER_SAFETY_MARGIN_MINUTES=10`.
- `PAPER_FALLBACK_STAKE_MULTIPLIER=0.5`.
- `PYTHON_VERSION=3.14`; switch to `3.13` if the 3.14 slim image is unavailable
  on your registry, then rebuild.

## Notes

- Run one container only; SQLite has a single writer.
- Logs go to `data/paper/scheduler.log` and Docker's stdout.
- The research scripts (`market_only_eval.py`, `walk_forward_eval.py`, etc.) are
  not copied into the runtime image.
- No real money is involved; this is a paper ledger.
