# Walk-forward — bookmaker variants

- **Date:** 2026-11-24
- **Strategy:** Logistic + tree agree, shortest-odds pick
- **Features:** Book + engineered (19 features)
- **Sizer:** Conservative pseudo-linear, 2% max stake
- **League:** La Liga
- **Walk-forward:** train on all prior seasons; test 2022-23 through 2025-26; refit each season
- **Odds source:** football-data.co.uk raw 1X2 files, cached in `data/odds_raw`

## Coverage

```
        Book  Matches with odds  La Liga matches with odds  Coverage %
      Bet365               8907                       1900       99.99
    Pinnacle               8056                       1708       90.44
     Bet&Win               8199                       1739       92.04
William Hill               6727                       1419       75.52
      VC Bet               5402                       1140       60.64
   Max price               8907                       1900       99.99
```

`Max price` is the maximum odds across the books in each file, not a single
bookmaker; treat it as a line-shopping upper bound.

## Primary — Bet365

```
 Bets  Hit %  Avg odds  Yield %    Yield 95% CI  ROI %  Final  Max DD % Seasons with bets
  554  70.76      1.52     5.45 +0.2% .. +10.6%  40.33 140.33     -9.91               4/4
```

**Per season**

```
 Season  Bets  Hit %  Yield %  ROI %  Max DD %
2022-23   149  66.44     0.34   0.57     -9.91
2023-24   157  69.43     5.96  11.30     -6.70
2024-25   143  69.93     3.29   5.98     -8.00
2025-26   105  80.00    12.14  18.29     -2.73
```

## Bookmaker comparison — common period 2022-23 to 2024-25

```
        Book  Common Bets  Common Hit %  Common Yield %  Common ROI %  Common Max DD %
      Bet365          449         68.60            3.32         18.63            -9.91
    Pinnacle          459         66.88            2.04         11.71            -9.88
     Bet&Win          358         66.76            0.60          2.58            -9.41
William Hill          374         70.86            4.47         22.06            -7.74
   Max price          433         69.05            6.91         42.80            -6.91
```

VC Bet is excluded because it has no 2024-25 odds; its full available period
is shown below. Bet&Win, Pinnacle and William Hill have partial coverage within
these seasons, so their bet counts differ.

## Full available period per book

```
        Book  Bets  Hit %  Avg odds  Yield %    Yield 95% CI  ROI %  Final  Max DD % Seasons with bets
      Bet365   554  70.76      1.52     5.45 +0.2% .. +10.6%  40.33 140.33     -9.91               4/4
    Pinnacle   513  68.23      1.53     3.23  -2.3% .. +8.7%  21.38 121.38     -9.88               4/4
     Bet&Win   458  69.21      1.51     3.07  -2.8% .. +8.6%  17.60 117.60     -9.41               4/4
William Hill   374  70.86      1.51     4.47 -2.1% .. +10.7%  22.06 122.06     -7.74               3/4
      VC Bet   313  67.09      1.55     1.63  -5.8% .. +8.9%   5.70 105.70    -11.45               2/4
   Max price   547  70.38      1.56     7.73 +2.3% .. +13.0%  65.76 165.76     -6.91               4/4
```

`Seasons with bets` reflects bookmaker coverage, not strategy choice.

## Per-season yield (%)

```
 Season  Bet365  Pinnacle  Bet&Win  William Hill  VC Bet  Max price
2022-23    0.34     -1.09    -1.93          2.88   -2.83       4.58
2023-24    5.96      6.09     5.22          8.08    5.86       6.92
2024-25    3.29      1.31    -3.69          2.22     NaN       8.53
2025-26   12.14     10.98    10.29           NaN     NaN       9.94
```

## Key findings

- The Bet365 baseline reproduces the ablation's `Book + engineered` result:
  554 bets, 70.8% hit rate, 5.45% yield, +40.3% ROI, -9.9% max drawdown.
- Only Bet365 (+0.2% to +10.6%) and the max-price line (+2.3% to +13.0%) have
  yield confidence intervals above zero. Pinnacle (3.23%, -2.3% to +8.7%),
  Bet&Win (3.07%, -2.8% to +8.6%), William Hill (4.47%, -2.1% to +10.7%) and
  VC Bet (1.63%, -5.8% to +8.9%) all include zero.
- Pinnacle is the sharper book, but its yield is lower than Bet365's. The edge
  does not simply improve with better prices, which suggests it may be tied to
  Bet365's specific pricing rather than a general market inefficiency.
- The max-price upper bound (7.73%) shows line shopping could help, but it is
  not a single-book strategy and cannot be executed at one bookmaker.
- 2022-23 is the weak season for most books; 2025-26 is strong where data
  exists. The edge is not uniform across seasons.
- William Hill (3/4 test seasons) and VC Bet (2/4) have partial coverage; their
  full-period rows are not directly comparable.

## Caveats

- Each variant is retrained on that bookmaker's odds, not just settled at its
  prices. Differences therefore mix price effects with model/selection effects.
- Missing book odds shrink the sample; `Seasons with bets` and the per-season
  table show where coverage is incomplete.
- The historical odds snapshot timing is the source's pre-match snapshot, not a
  controlled live decision time. Live execution and closing-line value are not
  modeled.
- No commission, stake limits, rejected bets, or slippage are included.
- One prepared match has missing market averages; rows with incomplete
  engineered features are skipped by the model.
- Betfair, BetMGM, BetVictor, Coral and Ladbrokes only appear in the 2025-26
  raw files; they have no historical training coverage and are not included.

## Files

- `reports/bookmaker_pooled.csv`
- `reports/bookmaker_common.csv`
- `reports/bookmaker_per_season.csv`
- `reports/bookmaker_coverage.csv`
- `reports/bookmaker_season_yields.csv`
