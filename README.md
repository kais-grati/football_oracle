# Football Oracle

[![tests](https://github.com/kais-grati/football_oracle/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/kais-grati/football_oracle/actions/workflows/tests.yml)

Football Oracle is a research and **paper-trading** system for a La Liga
football model. It combines market-average 1X2 odds, an engineered feature
vector, and a logistic-regression + decision-tree ensemble. A bet is simulated
only when both models agree; the selected outcome is the market favorite and
Bet365 is the execution price.

The scheduler decides each fixture independently at `kickoff_utc - 30m`,
fetches a fresh consensus snapshot, falls back to Bet365 when consensus is
unusable, sizes six independent paper wallets, records every decision in
SQLite, and optionally sends a Discord alert.

> **Research only.** This project does not place real bets, does not handle
> real money, and is not financial, betting, or investment advice. Past
> performance does not guarantee future results. If you use anything here,
> comply with local law and with the terms of your data and odds providers.

## What is implemented

- Per-fixture scheduling at T−30m with a strict T−10m cutoff and no bets after
  kickoff.
- Consensus-first features with a flagged Bet365 fallback, separate source
  statistics, and a reduced fallback stake.
- Six pseudo-linear sizing profiles: Conservative, Moderate, Medium,
  Medium-High, High, Aggressive.
- A six-wallet SQLite ledger with pending-exposure reservations, settlement,
  drawdown, hit rate, yield, ROI, and per-source reporting.
- Idempotent decisions and wallet bets, restart safety, retry backoff, and
  settlement throttling.
- Bzzoiro fixtures/results/consensus and MoviOdds Bet365 adapters.
- A fake-clock/fake-provider test suite and a GitHub Actions test workflow.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the scheduler design and
[`STRATEGY.md`](STRATEGY.md) for the research results and limitations.

## Repository layout

```text
papertrader/             Production scheduler, providers, ledger, CLI, stats
tests/                   Offline unit and integration tests
paper_trader.py          Thin CLI wrapper
simulator.py             Research simulator, selectors, sizers
market_only_eval.py      Market-average model vs the earlier model
market_feature_ablation.py
walk_forward_eval.py     Season walk-forward evaluation
bookmaker_walk_forward.py
tune_pseudo_linear.py    Sizing optimization on validation/test
run_discriminators.py    Notebook discriminator experiments
ablation_features.py
build_features.py        Historical feature builder
build_player_ratings.py  FIFA/EA FC player-rating aggregation
download_data.py         Historical match-data downloader
experiments.ipynb        Research notebook
model_comparison.ipynb   Model comparison notebook
api_experiments.ipynb    Provider/API exploration
reports/                 Generated research reports
Dockerfile               Paper-trader image
docker-compose.yml       Single-container deployment
```

The generated `data/` tree, EA/FIFA CSVs, SQLite ledgers, model artifacts, and
logs are intentionally ignored. Historical match CSVs at the repository root
are included to make the research scripts reproducible.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env
chmod 600 .env
# Fill in BZZOIRO-API-KEY and ODDS-API-KEY. Discord is optional.

# Fit the current market-only model.
python paper_trader.py train

# Run one tick, or the scheduler loop.
python paper_trader.py tick
python paper_trader.py run --interval 30

# Inspect the paper ledger.
python paper_trader.py stats
python paper_trader.py stats --wallet Conservative
python paper_trader.py status
```

The scheduler writes to `data/paper/ledger.sqlite` and
`data/paper/model.joblib` by default. Those files are local state, not source.

## Decision policy

1. At `kickoff_utc - 30m`, fetch consensus. Accept it only when all three
   outcomes exist, the snapshot is at most 45 minutes old, and the prices are
   sane.
2. If consensus is missing, stale, or incomplete, fetch the Bet365 board.
   Recompute the same feature vector from Bet365, mark
   `odds_source=bet365_fallback`, and apply the fallback stake multiplier.
3. If neither source yields a complete 1X2 price and an executable selected
   outcome, skip the fixture and record the reason.
4. Require both models to agree. The pick is the market favorite; the stake is
   sized from `bankroll - pending exposure`.
5. Never place a new bet after the safety cutoff or after kickoff.

## Tests

```bash
python -m pytest -q tests
```

The suite is offline. It uses fake clocks, fake providers, and saved provider
payloads. It covers feature consistency, model training/loading, ledger
migrations, provider parsing and retries, T−30m timing, multiple fixtures per
day, failures and fallbacks, bankroll reservations, settlement, source-level
stats, snapshots, CLI behavior, retry backoff, and restart idempotency.

GitHub Actions runs the same suite on pushes and pull requests.

## Deployment

`Dockerfile` and `docker-compose.yml` run one scheduler container with a
persistent `./data/paper` volume. The scheduler is paper-only. See
[`DOCKER.md`](DOCKER.md) for setup, model training, health checks, backups, and
the operational commands.

## Data and attribution

- Historical results and odds: [football-data.co.uk](https://www.football-data.co.uk/).
- Player ratings: public FIFA/EA FC datasets referenced in
  `build_player_ratings.py`; those datasets retain their own terms.
- Live decision-time prices: Bzzoiro and MoviOdds, subject to their API terms
  and plan limits.
- The repository does not include API keys or downloaded provider snapshots.

If you redistribute data or derived reports, check the upstream licences and
terms first.

## Limitations

- La Liga only. Other leagues were negative in the research evaluation.
- Small sample: roughly four seasons of walk-forward results.
- The market-only model is a credible peer to the earlier model, not a proven
  edge.
- Bet365 fallback is out-of-distribution relative to consensus-trained
  features; it is flagged, haircut, and reported separately.
- No commission, live liquidity, bookmaker limits, or closing-line value are
  modeled.
- Provider coverage, update frequency, and terms can change.

## License

The code in this repository is released under the MIT License; see
[`LICENSE`](LICENSE). Third-party data, datasets, and API responses are not
covered by that license and remain subject to their original terms.
