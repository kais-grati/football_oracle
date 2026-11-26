# Football Oracle — Strategy Summary

**Date:** 2026-11-24
**Status:** research / paper-trading candidate. The per-fixture scheduler is implemented and running in paper mode. No real money is involved.

## Universe and data

- Train on all five leagues' prior seasons.
- Bet **La Liga only**. Every other league was negative across every tested variant; the all-league portfolio loses money.
- Features: **market-average odds only**.
- Execution: **Bet365 price** for the selected outcome.
- Pre-match only, one decision snapshot per fixture at T−30m.

## Upgraded feature set

- Two probability logits from the market-average implied probabilities:
  - `log(p_home / p_away)`
  - `log(p_draw / p_away)`
- `market_overround`
- Raw calendar: `month`, `day`, `time_mins`
- Cyclical calendar: sin/cos for month, day-of-week, and time-of-day
- Flags: `is_weekend`, `is_midweek`, `matchday` (weekly season proxy)
- League: ordinal `league_enc`

No raw odds, no duplicated probability columns, no one-hot league, and no Bet365 features. Bet365 is only the execution price.

## Model and decision

- **Logistic regression** (standardized pipeline) and **decision tree** (`max_depth=4`, `min_samples_leaf=50`).
- Both models must predict the pick wins.
- **Target:** whether the market-average favorite wins.
- **Pick:** shortest odds by the market average.
- **Bet price:** Bet365 odds for that pick.
- **Training:** season-based walk-forward; train on all prior seasons; refit each season.

## Sizing

The tested results use the **Medium pseudo-linear sizer** currently configured in `market_only_eval.py`:

- 10% maximum stake
- `x_trans=0.65`, `y_trans=0.04`, `slope=0.50`

For live paper trading, start with the **Conservative 2%** profile or the six-wallet setup. Medium produces large drawdowns.

## Performance — walk-forward, all-league training, La Liga only

| Metric | Value |
|---|---:|
| Bets | 570 |
| Hit rate | 70.53% |
| Yield | **5.53%** |
| Yield 95% CI | +0.0% .. +10.7% |
| ROI (Medium sizer) | +304.2% |
| Max drawdown (Medium) | −31.6% |
| Seasons positive | 4/4 |
| Per-season yield | +2.76%, +5.22%, +3.08%, +8.45% |

## Like-for-like vs the actual strategy

Same Conservative 2% sizer, same La Liga universe, same Bet365 execution, same walk-forward:

| Strategy | Bets | Hit % | Yield % | Yield 95% CI | ROI % | Max DD % |
|---|---:|---:|---:|---|---:|---:|
| Actual: Bet365 + engineered | 554 | 70.76 | 5.45 | +0.2% .. +10.6% | +40.3 | −9.9 |
| Upgrade: market-only logits + calendar + cyclical | 570 | 70.53 | 5.51 | +0.3% .. +10.5% | +44.1 | −8.0 |

Per-season yield:

| Season | Actual | Upgrade |
|---|---:|---:|
| 2022-23 | +0.34% | +3.90% |
| 2023-24 | +5.96% | +6.26% |
| 2024-25 | +3.29% | +3.03% |
| 2025-26 | +12.14% | +8.82% |

The two are statistically indistinguishable. The upgrade has a slightly better confidence-interval lower bound and drawdown, plus a much better worst season (3.03% vs 0.34%). The actual has the stronger 2025-26 upside (12.14% vs 8.82%). The earlier upgrade ROI of +304% and −31.6% drawdown came from the Medium sizer, not from a better model; with the same Conservative sizer the upgrade is +44.1% ROI and −8.0% drawdown.

## Feature-set comparison

- Two logits alone: **5.31%**, CI −0.2% .. +10.4%.
- Calendar flags alone: **5.22%**.
- The full stack adds only ~0.2 pp over two logits alone — within noise.
- One-hot league, log-odds, and the full combined sets all reduce La Liga performance.

The upgrade is a credible peer to the actual strategy, not a proven improvement. Its main advantages are slightly lower drawdown and better season-to-season stability; the actual has stronger recent upside.

## Deployment

`papertrader/` implements:

- six independent wallets (Conservative through Aggressive)
- per-fixture decisions at T−30m, not one daily batch
- Bzzoiro consensus features, with a flagged Bet365 fallback
- MoviOdds Bet365 execution prices
- a SQLite ledger, snapshot archive, Discord alerts, and per-source stats

The scheduler is implemented in `papertrader/` and runs in paper mode with
the six-wallet ledger. To reproduce the deployment:

1. Build the image (the Dockerfile copies `papertrader/`).
2. Set `BZZOIRO-API-KEY`, `ODDS-API-KEY`, and optionally
   `DISCORD_WEBHOOK_URL`.
3. Run `python paper_trader.py train` once inside the container to create the
   market-only model artifact.
4. Start with `python paper_trader.py run --interval 30` and monitor the
   ledger and logs. No real money is involved.

### Consensus fallback

If consensus odds are missing, stale, or incomplete at decision time, the scheduler can rebuild the same feature vector from Bet365 odds. That is a **degraded fallback**: it is out-of-distribution versus the consensus-trained model, so it is flagged as `odds_source=bet365_fallback`, uses a reduced stake, and is monitored separately. A dedicated Bet365 fallback model is preferable once enough fallback samples exist. If Bet365 is also unavailable, the decision is skipped. See `ARCHITECTURE.md`.

## Main risks

- **La Liga only.** The model loses in the Premier League, Bundesliga, Serie A, and Ligue 1; the all-league aggregate is −2.47% yield.
- **Small sample.** About 570 bets over four seasons; the confidence interval is only marginally above zero.
- **Redundant time features.** The full stack includes both raw and cyclical time; the gain over two logits alone is small and may be overfit.
- **Medium sizing is aggressive.** ROI and drawdown are not comparable to Conservative; use a capped profile for live.
- **Data dependency.** Bzzoiro free gives consensus but not Bet365 per-book; MoviOdds gives Bet365 but no consensus. A hybrid means two feeds to join; Bzzoiro paid gives both.
- **No SLA or provenance clarity.** Bzzoiro is the better fit but has reliability and provenance caveats. No commission, limits, or slippage are modeled.
- **No CLV yet.** Free tiers do not give closing lines; use football-data closing odds or snapshot near kickoff.

> Research and paper-trading only. This is not betting, financial, or investment advice; no result is guaranteed. Past performance does not predict future performance.

## Bottom line

The upgraded market-only model is a credible **La Liga peer** to the current Bet365 + consensus strategy: about 5.5% yield, four positive seasons, slightly lower drawdown, and a better worst season. The actual strategy has stronger recent upside. Neither is statistically proven, and neither should be relied on across leagues. Keep the upgrade as a challenger, paper-trade it, and choose between them based on which feature dependency you want to operate.

## Reproducibility

- `market_only_eval.py` — market-average vs current model comparison.
- `market_feature_ablation.py` — preprocessing and feature ablation.
- `papertrader/` — per-fixture scheduler, providers, ledger, CLI, and tests.
- `paper_trader.py` — thin CLI wrapper.
- `reports/market_feature_ablation_*.csv` — full ablation metrics.
