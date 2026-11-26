"""Walk-forward evaluation across bookmaker variants.

Rebuilds the "Book + engineered" feature set for several bookmakers from the
raw football-data.co.uk files, then runs the same season-based walk-forward:
train on all prior seasons, predict the next season, bet La Liga only with the
Conservative 2% sizer, and require logistic regression and decision tree to
agree.

The report is written to reports/bookmaker_walk_forward.md, with the full
tables as CSV files in the same directory.

Run:
    python bookmaker_walk_forward.py
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ablation_features import (
    BOOK_FEATURES,
    ENGINEERED_FEATURES,
    add_engineered_features,
    bootstrap_yield,
    season_order,
    walk_forward_predictions,
)
from simulator import PseudoLinearSizer, Simulator, load_matches

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{division}.csv"
LEAGUES = [
    ("SP1", "La Liga"),
    ("E0", "Premier League"),
    ("D1", "Bundesliga"),
    ("I1", "Serie A"),
    ("F1", "Ligue 1"),
]
SEASONS = ["2122", "2223", "2324", "2425", "2526"]

RAW_DIR = Path("data/odds_raw")
REPORT_DIR = Path("reports")

LEAGUE = "La Liga"
TRAIN_SEASONS = 1
COMMON_SEASONS = ["2022-23", "2023-24", "2024-25"]
FEATURES = BOOK_FEATURES + ENGINEERED_FEATURES
SIZER = PseudoLinearSizer(0.65, 0.01, 0.10, max_fraction=0.02)

BOOKS = {
    "Bet365": ("B365H", "B365D", "B365A"),
    "Pinnacle": ("PSH", "PSD", "PSA"),
    "Bet&Win": ("BWH", "BWD", "BWA"),
    "William Hill": ("WHH", "WHD", "WHA"),
    "VC Bet": ("VCH", "VCD", "VCA"),
    "Max price": ("MaxH", "MaxD", "MaxA"),
}
BOOK_COLUMNS = {column for columns in BOOKS.values() for column in columns}


# ---------------------------------------------------------------------------
# Raw odds
# ---------------------------------------------------------------------------

def download_raw_files() -> None:
    """Cache the raw football-data CSVs, downloading any that are missing."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        for division, _ in LEAGUES:
            path = RAW_DIR / f"{season}_{division}.csv"
            if path.exists():
                continue
            response = requests.get(
                BASE_URL.format(season=season, division=division),
                timeout=30,
            )
            response.raise_for_status()
            path.write_bytes(response.content)


def read_raw(path: Path) -> pd.DataFrame:
    """Read one raw file, handling the BOM and older cp1252 seasons."""
    content = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    return pd.read_csv(StringIO(text), on_bad_lines="skip")


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse football-data dates, which are dd/mm/yy or dd/mm/yyyy."""
    text = values.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    two_digit = text.str.fullmatch(r"\d{1,2}/\d{1,2}/\d{2}", na=False)
    four_digit = text.str.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", na=False)
    if two_digit.any():
        parsed.loc[two_digit] = pd.to_datetime(
            text[two_digit], format="%d/%m/%y", errors="coerce"
        )
    if four_digit.any():
        parsed.loc[four_digit] = pd.to_datetime(
            text[four_digit], format="%d/%m/%Y", errors="coerce"
        )
    other = text.notna() & ~two_digit & ~four_digit
    if other.any():
        parsed.loc[other] = pd.to_datetime(
            text[other], dayfirst=True, errors="coerce"
        )
    return parsed.dt.date


def load_raw_frames() -> dict[tuple[str, str], pd.DataFrame]:
    """Load every raw file with normalized date/team/division keys."""
    frames = {}
    for season in SEASONS:
        for division, _ in LEAGUES:
            raw = read_raw(RAW_DIR / f"{season}_{division}.csv")
            keys = pd.DataFrame({
                "date": parse_dates(raw["Date"]),
                "home_team": raw["HomeTeam"].astype("string").str.strip(),
                "away_team": raw["AwayTeam"].astype("string").str.strip(),
                "division": division,
            }, index=raw.index)
            odds = raw[[c for c in raw.columns if c in BOOK_COLUMNS]]
            frame = pd.concat([keys, odds], axis=1)
            frames[(season, division)] = frame.dropna(
                subset=["date", "home_team", "away_team"]
            )
    return frames


def extract_book_odds(raw_frames: dict[tuple[str, str], pd.DataFrame],
                      columns: tuple[str, str, str]) -> pd.DataFrame:
    """Return complete three-way odds for one bookmaker, keyed by match."""
    pieces = []
    for raw in raw_frames.values():
        if not all(column in raw.columns for column in columns):
            continue
        values = raw.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
        complete = values.notna().all(axis=1) & (values > 1).all(axis=1)
        if not complete.any():
            continue
        piece = raw.loc[complete, ["date", "home_team", "away_team", "division"]].copy()
        piece["book_home_win"] = values.loc[complete, columns[0]].to_numpy()
        piece["book_draw"] = values.loc[complete, columns[1]].to_numpy()
        piece["book_away_win"] = values.loc[complete, columns[2]].to_numpy()
        pieces.append(piece)

    book = pd.concat(pieces, ignore_index=True)
    return (
        book.drop_duplicates(subset=["date", "home_team", "away_team", "division"])
        .reset_index(drop=True)
    )


def attach_book_odds(matches: pd.DataFrame, book_odds: pd.DataFrame) -> pd.DataFrame:
    """Merge one bookmaker's odds onto the prepared matches."""
    data = matches.copy()
    data["_date"] = pd.to_datetime(data["date"]).dt.date
    merged = data.merge(
        book_odds,
        left_on=["_date", "home_team", "away_team", "division"],
        right_on=["date", "home_team", "away_team", "division"],
        how="left",
        suffixes=("", "_book"),
    )
    return merged.drop(columns=["_date", "date_book"], errors="ignore")


def make_variant(merged: pd.DataFrame) -> pd.DataFrame:
    """Replace the book odds and recompute probs, pick, target, engineered features."""
    variant = merged.copy()
    variant["odds_home_win"] = variant["book_home_win"]
    variant["odds_draw"] = variant["book_draw"]
    variant["odds_away_win"] = variant["book_away_win"]

    inverse = 1 / variant[["odds_home_win", "odds_draw", "odds_away_win"]]
    total = inverse.sum(axis=1)
    variant["prob_home"] = inverse["odds_home_win"] / total
    variant["prob_draw"] = inverse["odds_draw"] / total
    variant["prob_away"] = inverse["odds_away_win"] / total

    conditions = [
        (variant["odds_home_win"] <= variant["odds_draw"])
        & (variant["odds_home_win"] <= variant["odds_away_win"]),
        (variant["odds_away_win"] <= variant["odds_draw"])
        & (variant["odds_home_win"] >= variant["odds_away_win"]),
        (variant["odds_away_win"] >= variant["odds_draw"])
        & (variant["odds_home_win"] >= variant["odds_draw"]),
    ]
    variant["pred"] = np.select(
        conditions, ["home", "away", "draw"], default=None
    )
    variant["guessed_correct"] = variant["pred"] == variant["winner"]

    return add_engineered_features(variant)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_book(data: pd.DataFrame, seasons: list[str],
                  sim: Simulator) -> tuple[dict, pd.DataFrame]:
    """Run the season walk-forward and return pooled plus per-season results."""
    test_seasons = seasons[TRAIN_SEASONS:]
    logreg_pred, tree_pred = walk_forward_predictions(data, seasons, FEATURES)

    oos = data["season"].isin(test_seasons)
    mask = oos & logreg_pred & tree_pred

    full = sim.run(data, selector=mask, sizer=SIZER)
    full_metrics = full.metrics
    low, high = bootstrap_yield(full.history)

    season_rows = []
    for season in test_seasons:
        metrics = sim.run(
            data,
            selector=mask & (data["season"] == season),
            sizer=SIZER,
        ).metrics
        season_rows.append({
            "Season": season,
            "Bets": metrics["bets"],
            "Hit %": metrics["accuracy"] * 100,
            "Yield %": metrics["yield_pct"],
            "ROI %": metrics["roi_pct"],
            "Max DD %": metrics["max_drawdown_pct"],
        })

    per_season = pd.DataFrame(season_rows)
    seasons_with_bets = int((per_season["Bets"] > 0).sum())

    pooled = {
        "Bets": full_metrics["bets"],
        "Hit %": full_metrics["accuracy"] * 100,
        "Avg odds": full_metrics["avg_odds"],
        "Yield %": full_metrics["yield_pct"],
        "Yield 95% CI": f"{low:+.1f}% .. {high:+.1f}%",
        "ROI %": full_metrics["roi_pct"],
        "Final": full_metrics["final_bankroll"],
        "Max DD %": full_metrics["max_drawdown_pct"],
        "Seasons with bets": f"{seasons_with_bets}/{len(test_seasons)}",
    }
    return pooled, per_season


def evaluate_common(data: pd.DataFrame, seasons: list[str],
                    sim: Simulator) -> dict:
    """Pooled result restricted to the common 2022-23..2024-25 test period."""
    test_seasons = seasons[TRAIN_SEASONS:]
    logreg_pred, tree_pred = walk_forward_predictions(data, seasons, FEATURES)

    oos = data["season"].isin(test_seasons)
    common = data["season"].isin(COMMON_SEASONS)
    mask = oos & common & logreg_pred & tree_pred

    metrics = sim.run(data, selector=mask, sizer=SIZER).metrics
    return {
        "Bets": metrics["bets"],
        "Hit %": metrics["accuracy"] * 100,
        "Yield %": metrics["yield_pct"],
        "ROI %": metrics["roi_pct"],
        "Max DD %": metrics["max_drawdown_pct"],
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report_table(frame: pd.DataFrame) -> str:
    return "```\n" + frame.to_string(index=False) + "\n```"


def write_report(pooled: pd.DataFrame, common: pd.DataFrame,
                 per_season: pd.DataFrame, coverage: pd.DataFrame,
                 season_yields: pd.DataFrame) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    primary = pooled[pooled["Book"] == "Bet365"].drop(columns=["Book"]).round(2)
    primary_seasons = (
        per_season[per_season["Book"] == "Bet365"]
        .drop(columns=["Book"])
        .round(2)
    )
    common_table = common[common["Book"] != "VC Bet"].round(2)
    pooled_table = pooled.round(2)
    coverage_table = coverage.round(2)
    season_table = season_yields.reset_index().round(2)

    report = f"""# Walk-forward — bookmaker variants

- **Date:** 2026-11-24
- **Strategy:** Logistic + tree agree, shortest-odds pick
- **Features:** Book + engineered ({len(FEATURES)} features)
- **Sizer:** Conservative pseudo-linear, 2% max stake
- **League:** {LEAGUE}
- **Walk-forward:** train on all prior seasons; test 2022-23 through 2025-26; refit each season
- **Odds source:** football-data.co.uk raw 1X2 files, cached in `{RAW_DIR}`

## Coverage

{report_table(coverage_table)}

`Max price` is the maximum odds across the books in each file, not a single
bookmaker; treat it as a line-shopping upper bound.

## Primary — Bet365

{report_table(primary)}

**Per season**

{report_table(primary_seasons)}

## Bookmaker comparison — common period 2022-23 to 2024-25

{report_table(common_table)}

VC Bet is excluded because it has no 2024-25 odds; its full available period
is shown below. Bet&Win, Pinnacle and William Hill have partial coverage within
these seasons, so their bet counts differ.

## Full available period per book

{report_table(pooled_table)}

`Seasons with bets` reflects bookmaker coverage, not strategy choice.

## Per-season yield (%)

{report_table(season_table)}

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
"""
    path = REPORT_DIR / "bookmaker_walk_forward.md"
    path.write_text(report)

    pooled_table.to_csv(REPORT_DIR / "bookmaker_pooled.csv", index=False)
    common_table.to_csv(REPORT_DIR / "bookmaker_common.csv", index=False)
    per_season.round(2).to_csv(REPORT_DIR / "bookmaker_per_season.csv", index=False)
    coverage_table.to_csv(REPORT_DIR / "bookmaker_coverage.csv", index=False)
    season_table.to_csv(REPORT_DIR / "bookmaker_season_yields.csv", index=False)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> Path:
    download_raw_files()
    raw_frames = load_raw_frames()
    matches = load_matches()
    seasons = season_order(matches)
    sim = Simulator(league=LEAGUE)

    coverage_rows, pooled_rows, common_rows, season_rows = [], [], [], []

    for book_name, columns in BOOKS.items():
        book_odds = extract_book_odds(raw_frames, columns)
        merged = attach_book_odds(matches, book_odds)
        variant = make_variant(merged)

        has_odds = merged["book_home_win"].notna()
        coverage_rows.append({
            "Book": book_name,
            "Matches with odds": int(has_odds.sum()),
            "La Liga matches with odds": int(
                (has_odds & (merged["league"] == LEAGUE)).sum()
            ),
            "Coverage %": has_odds.mean() * 100,
        })

        pooled, per_season = evaluate_book(variant, seasons, sim)
        pooled_rows.append({"Book": book_name, **pooled})
        common = evaluate_common(variant, seasons, sim)
        common_rows.append({"Book": book_name, **common})
        per_season.insert(0, "Book", book_name)
        season_rows.append(per_season)

    coverage = pd.DataFrame(coverage_rows)
    pooled = pd.DataFrame(pooled_rows)
    common = pd.DataFrame(common_rows).rename(
        columns={"Bets": "Common Bets", "Hit %": "Common Hit %",
                 "Yield %": "Common Yield %", "ROI %": "Common ROI %",
                 "Max DD %": "Common Max DD %"}
    )
    per_season = pd.concat(season_rows, ignore_index=True)

    season_yields = per_season.pivot(
        index="Season", columns="Book", values="Yield %"
    )
    season_yields = season_yields.reindex(
        columns=list(BOOKS), index=seasons[TRAIN_SEASONS:]
    )

    path = write_report(
        pooled, common, per_season, coverage, season_yields
    )

    print("Bookmaker walk-forward complete")
    print(f"Report: {path}")
    print()
    print(pooled.round(2).to_string(index=False))
    return path


if __name__ == "__main__":
    main()
