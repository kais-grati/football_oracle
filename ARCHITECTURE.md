# Paper Trader Architecture — Per-Fixture T−30m Decisions

**Date:** 2026-11-25
**Status:** implemented, covered by the offline test suite, and running in paper mode. No real money is involved.
**Companion:** `STRATEGY.md`

## Implemented package

The implementation lives in `papertrader/`:

| Module | Responsibility |
|---|---|
| `config.py` | Environment/`.env` settings; secrets excluded from repr |
| `domain.py` | UTC fixtures, snapshots, results, predictions |
| `features.py` | 16-column market-only feature vector, shared by training and live |
| `model.py` | Logreg + tree training, loading, versioning |
| `sizing.py` | Six pseudo-linear wallet profiles |
| `db.py` | SQLite migrations, fixtures, decisions, snapshots, bets, exposure |
| `providers/bzzoiro.py` | Fixtures, results, consensus |
| `providers/moviodds.py` | Bet365 board and kickoff+team matching |
| `engine.py` | T−30m decision, fallback, sizing, Discord |
| `scheduler.py` | Tick loop, sync, retry/backoff, settlement throttle |
| `settlement.py` | Result settlement and bankroll updates |
| `stats.py` | Per-wallet and per-source reporting |
| `cli.py` | `train`, `tick`, `run`, `fixtures`, `settle`, `stats`, `status` |

`paper_trader.py` is a thin wrapper around `papertrader.cli`.

### Default policy

- Decision at `kickoff_utc - 30m`; cutoff at `kickoff_utc - 10m`.
- Consensus accepted only if complete, within 45 minutes, and sane.
- Otherwise Bet365 fallback with consensus feature rebuild, a 0.5 stake
  multiplier, and `odds_source=bet365_fallback`.
- 30-second tick; 120-second retry backoff; 300-second settlement interval;
  6-hour fixture refresh.
- One SQLite writer; one scheduler container.

### Commands

```bash
python paper_trader.py train
python paper_trader.py run --interval 30
python paper_trader.py tick
python paper_trader.py fixtures
python paper_trader.py settle
python paper_trader.py stats
python paper_trader.py status
```

## Core shift

Replace the single daily 09:00 batch with an **event-driven scheduler**:

```text
fixtures -> decision_at = kickoff_utc - 30 min -> tick -> snapshot
         -> consensus, else Bet365 fallback -> model -> wallets -> Discord
```

Each match gets its own decision at its own T−30m. Multiple matches in a day are handled independently, not as one morning batch.

## Components

1. **Fixture sync**
   - Pulls upcoming fixtures from the free fixtures source.
   - Stores kickoff in UTC, league, teams, status, and provider IDs.
   - Refreshes every few hours and at the start of each day to catch new or rescheduled matches.

2. **Scheduler**
   - Long-running process with a 30–60 second tick.
   - Each tick settles finished bets, refreshes the schedule if stale, and selects due decisions.
   - A one-shot `tick` command can also be run from cron/systemd.

3. **Snapshot fetcher**
   - At T−30m, tries **consensus** first.
   - If consensus is missing, stale, or incomplete, falls back to **Bet365** for the model features.
   - Also fetches Bet365 execution odds for the selected outcome.
   - Stores raw snapshots, the source used, and their timestamps.

4. **Decision engine**
   - Builds the market-only feature set from the consensus snapshot.
   - Loads the current model version.
   - Requires logreg and tree to agree.
   - Selects the market favorite; takes the Bet365 price for execution.
   - Records the decision even when it is a no-bet.

5. **Wallet ledger**
   - Existing `wallets` and `bets` tables.
   - Each decision produces up to six wallet bets.
   - Stakes are sized against `bankroll - pending exposure`.

6. **Settler**
   - Fetches results for completed matches.
   - Updates bet status, profit, wallet bankroll, and CLV.
   - Runs on the same tick loop, before new decisions.

7. **Notifier**
   - One Discord message per fixture at decision time, listing every wallet that placed a bet.
   - Include the decision lead time and consensus snapshot time.

8. **Snapshot archive / observability**
   - Persist the raw consensus and execution snapshots used for each decision.
   - This gives an audit trail and a dataset to retrain on actual T−30m consensus data.

## Data model additions

- `fixtures`: provider fixture ID, Bzzoiro event ID, league, teams, kickoff_utc, status, updated_at.
- `decisions`: fixture_id, decision_at_utc, snapshot_at_utc, staleness_seconds, actual_lead_seconds, model_version, consensus probabilities, execution odds, selection, status, reason, **odds_source** (`consensus` or `bet365_fallback`), **consensus_snapshot_at**, **fallback_snapshot_at**, **fallback_reason**.
- `bets`: add `decision_id`; keep `UNIQUE(wallet_id, match_id)`.
- One `decision` per fixture; one bet per wallet per fixture. These unique keys make retries and restarts idempotent.

## Tick loop

Every 30–60 seconds:

1. Settle bets whose matches have results.
2. Refresh fixtures if the schedule is stale.
3. Query fixtures where:
   - status is scheduled,
   - `decision_at <= now`,
   - `kickoff_utc > now + safety_margin` (e.g., 10 minutes),
   - no decision row exists yet.
4. Claim the due fixtures in a transaction.
5. Fetch consensus/execution snapshots.
6. Validate freshness and the kickoff window.
7. Build features, run both models, size for each wallet, insert the decision and bets.
8. Send Discord messages.
9. Mark the decision done, skipped, or failed.

A single process is enough. SQLite has one writer; keep the settler and scheduler in the same loop to avoid write contention.

## Multiple matches in a day

This falls out of per-fixture decision times:

- A 14:00 match decides at 13:30; a 21:00 match decides at 20:30.
- The 14:00 result is settled before the 20:30 decision, so the later stake uses the updated bankroll.
- For overlapping matches (16:15 and 18:30), the 16:15 stake is still pending at 18:00. The 18:30 stake is sized on `bankroll - pending exposure`, so you never overcommit.
- If several matches share a decision time, claim them together, fetch in a small parallel batch, and process wallet reservations in a deterministic order (decision time, kickoff, match ID).
- Do **not** process the whole day as one batch; that would use the wrong odds and the wrong bankroll state.

## Snapshot timing and freshness

- Store everything in UTC. `decision_at = kickoff_utc - 30m`.
- Fetch at T−30m; record `snapshot_at` from the provider’s `last_update_at`.
- Bzzoiro’s consensus updates every 30 minutes within 24h of kickoff. At T−30m, the latest snapshot may be up to 30 minutes old. Define a policy:
  - Accept the freshest snapshot at T−30m.
  - If it is older than, say, 45 minutes, retry every 2–5 minutes.
  - Stop retrying at T−10m. If still stale, skip the match and log it, or place with an explicit stale flag — decide this policy up front.
- Fetch the Bet365 execution price from the same snapshot moment. If consensus and execution timestamps differ materially, record both and apply a maximum mismatch rule.
- For CLV, either schedule a closing snapshot at T−1m/T+0 or use football-data closing odds after the match. This is separate from the decision and does not affect it.

## Consensus fallback to Bet365

Order of preference at T−30m:

1. **Consensus** — valid if all three 1X2 outcomes are present and >1, the snapshot is within the freshness policy, and the overround is finite and sane.
2. **Bet365 fallback** — if consensus is missing, stale, or incomplete, fetch the Bet365 1X2 snapshot and use it to build the same feature vector.
3. **Skip** — if neither source yields a full 1X2 set and an execution price, record a skipped decision.

Feature impact:

- The model's two probability logits and `market_overround` are recomputed from Bet365's normalized implied probabilities and Bet365's overround.
- The feature names and schema stay the same; only the source changes.
- This is an out-of-distribution input: the model was trained on consensus, and Bet365 typically has a higher margin and different probabilities. Treat fallback decisions as degraded.

Risk controls:

- Set `odds_source = bet365_fallback`; store the failed consensus attempt, its reason, and both snapshot times.
- Do not blend sources silently.
- Reduce fallback stakes, e.g. 50% of the normal sizer output or cap at the Conservative profile, until fallback performance is validated.
- Flag the Discord message as a Bet365 fallback.
- Split stats by `odds_source`; monitor fallback yield, hit rate, drawdown, and CLV separately.
- Optionally train a dedicated fallback model on Bet365-only features and select it by source; the market-only model should not be assumed to transfer unchanged.

Edge rules:

- If consensus has only two outcomes, treat it as missing; fallback.
- If the Bet365 snapshot is incomplete or lacks the selected outcome's price, skip.
- If consensus is stale but Bet365 is fresh, use the Bet365 fallback.
- If consensus is fresh but Bet365 execution is unavailable, skip; there is no bet without an executable price.
- Never place a bet after kickoff, regardless of source.

## Provider abstraction

Use interfaces, not hard-coded calls:

- `FixtureSource`: free fixtures source or Bzzoiro `/events`.
- `ConsensusSource`: Bzzoiro free consensus.
- `ExecutionSource`: MoviOdds Bet365, or Bzzoiro paid per-book Bet365.
- `ResultsSource`: free results source/football-data.

Map matches on **kickoff_utc + normalized team names**, and store provider IDs. Do not rely on team names alone.

The cleanest production setup is Bzzoiro paid: one event request can provide both consensus and Bet365 per-book odds. The free setup uses Bzzoiro consensus + MoviOdds Bet365 and needs the two sources joined at decision time. The Bet365 source serves two roles: execution price and fallback feature source when consensus is unavailable. Keep the fallback and execution snapshots from the same provider and timestamp where possible.

## Failure and edge cases

- **Missed T−30m:** on restart, decide only if the match is still before `kickoff - 10m`; record the actual lead. Otherwise skip.
- **Stale/unavailable API:** retry with backoff, isolate failures per fixture, continue other matches.
- **Consensus unavailable/stale/incomplete:** switch to the Bet365 fallback if a complete Bet365 snapshot and execution price exist; mark the decision and apply the reduced-stake policy. If Bet365 is also unavailable, skip.
- **Fallback feature drift:** monitor fallback performance separately; do not treat it as equivalent to consensus.
- **Postponed/kickoff changed:** refresh fixtures; move `decision_at`; void or skip bets if a match is cancelled.
- **Partial wallet writes:** retry only the missing wallet bets; never duplicate a `(wallet, match)`.
- **Rate limits:** Bzzoiro 25 req/s and 7,500/day; MoviOdds 30/min. Use a small concurrency limit and respect `Retry-After`/429.
- **After kickoff:** never place a new bet; mark the decision skipped.

## Operations

- The Docker command is `python paper_trader.py run --interval 30`; the same loop can be invoked as a one-shot `tick` from cron/systemd.
- One container, `restart: unless-stopped`, persistent `data/paper/` volume, UTC internally.
- Validate the new scheduler in shadow mode before replacing a prior version; run only one scheduler container against a ledger.
- Add `status`/`schedule` commands showing the next decision times, pending decisions, stale snapshots, and last errors.

## Testing and rollout

The `tests/` suite injects a fake clock and fake providers and covers:

- live/historical feature consistency, including local-midnight kickoffs;
- provider parsing from saved API payloads, retries, pagination, team aliases;
- T−30m decisions, 10-minute cutoff, stale consensus, Bet365 fallback,
  incomplete snapshots, model disagreement, and missing execution prices;
- one board request for simultaneous fixtures, wallet reservations, settlement
  order, lost/void bets, source-separated stats, and idempotent restarts;
- retry backoff, settlement throttling, fixture refresh, and malformed provider
  rows;
- CLI training and reporting.

Live smoke checks against the real Bzzoiro and MoviOdds APIs validated the
current La Liga fixture list, consensus payloads, results, the Bet365 board,
and 12/12 fixture-to-board matching. The live scheduler was replayed with a
future clock and correctly rejected stale snapshots.

Before deployment, run in shadow mode alongside the current daily paper trader
for a few weeks, then compare decisions, odds, and results.

## Open operational decisions (defaults implemented)

1. Max staleness and cutoff: how old can consensus be at T−30m, and when do you skip?
2. Execution provider: MoviOdds Bet365, Bzzoiro paid per-book, or another book.
3. Whether to archive every T−30m snapshot and retrain the model on production consensus data.
4. Whether to capture a closing snapshot for CLV or rely on football-data after the match.
5. Fallback stake multiplier and whether to train a dedicated Bet365 fallback model.
