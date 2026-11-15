"""Build a football match + 1X2 odds dataset from football-data.co.uk.

Each downloaded CSV covers one division and season. The output CSV has one row
per match with the teams, final score, winner, and home/draw/away odds.

Run:
    python download_data.py

Add more data by extending LEAGUES and SEASONS below.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{division}.csv"

# (division code, human-readable league name)
LEAGUES = [
    ("SP1", "La Liga"),
    ("E0", "Premier League"),
    ("D1", "Bundesliga"),
    ("I1", "Serie A"),
    ("F1", "Ligue 1"),
]

# football-data season codes: 2425 = 2024/25
SEASONS = ["2425"]

OUTPUT_PATH = Path(__file__).resolve().parent / "football_matches_odds.csv"

REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]

# Preferred bookmakers, in order. For each match, the first source with all
# three odds present becomes the unified odds_home_win/draw/away_win columns.
ODDS_SOURCES = [
    ("b365", ("B365H", "B365D", "B365A")),
    ("pinnacle_close", ("PSCH", "PSCD", "PSCA")),
    ("market_avg", ("AvgH", "AvgD", "AvgA")),
]

# Kept alongside the unified columns for reference.
EXTRA_ODDS_COLUMNS = {
    "AvgH": "market_avg_home_win",
    "AvgD": "market_avg_draw",
    "AvgA": "market_avg_away_win",
}


def season_label(code: str) -> str:
    """Convert a football-data season code to a readable label: '2425' -> '2024-25'."""
    start = int(code[:2])
    start += 2000 if start < 90 else 1900
    return f"{start}-{str(start + 1)[-2:]}"


def download_division(season: str, division: str) -> pd.DataFrame:
    url = BASE_URL.format(season=season, division=division)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # The files start with a UTF-8 BOM. Decoding as utf-8-sig strips it;
    # otherwise pandas names the first column "﻿Div" and selects fail.
    # Older seasons are cp1252 rather than UTF-8, so fall back for those.
    content = response.content
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
    two_digit_year = text.str.fullmatch(r"\d{1,2}/\d{1,2}/\d{2}", na=False)
    four_digit_year = text.str.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", na=False)
    if two_digit_year.any():
        parsed.loc[two_digit_year] = pd.to_datetime(
            text[two_digit_year], format="%d/%m/%y", errors="coerce"
        )
    if four_digit_year.any():
        parsed.loc[four_digit_year] = pd.to_datetime(
            text[four_digit_year], format="%d/%m/%Y", errors="coerce"
        )
    # Anything else: let pandas try its best rather than silently dropping it.
    other = text.notna() & ~two_digit_year & ~four_digit_year
    if other.any():
        parsed.loc[other] = pd.to_datetime(
            text[other], dayfirst=True, errors="coerce"
        )
    return parsed.dt.date


def extract_matches(
    raw: pd.DataFrame, league: str, season: str, division: str
) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            f"{league} {season_label(season)} ({division}) is missing "
            f"required columns: {missing}"
        )

    out = pd.DataFrame(index=raw.index)
    out["date"] = parse_dates(raw["Date"])
    out["time"] = (
        raw["Time"].astype("string").str.strip()
        if "Time" in raw.columns
        else pd.Series(pd.NA, index=raw.index, dtype="string")
    )
    out["home_team"] = raw["HomeTeam"].astype("string").str.strip()
    out["away_team"] = raw["AwayTeam"].astype("string").str.strip()
    out["home_goals"] = pd.to_numeric(raw["FTHG"], errors="coerce")
    out["away_goals"] = pd.to_numeric(raw["FTAG"], errors="coerce")

    # Winner: trust FTR when present, fall back to the final score.
    if "FTR" in raw.columns:
        winner = (
            raw["FTR"]
            .astype("string")
            .str.strip()
            .str.upper()
            .map({"H": "home", "D": "draw", "A": "away"})
            .astype("string")
        )
    else:
        winner = pd.Series(pd.NA, index=raw.index, dtype="string")

    from_score = pd.Series(pd.NA, index=raw.index, dtype="string")
    from_score.loc[out["home_goals"] > out["away_goals"]] = "home"
    from_score.loc[out["home_goals"] < out["away_goals"]] = "away"
    from_score.loc[out["home_goals"] == out["away_goals"]] = "draw"
    out["winner"] = winner.fillna(from_score)

    # Unified 1X2 odds with a fallback chain across bookmakers.
    for column in ("odds_home_win", "odds_draw", "odds_away_win"):
        out[column] = float("nan")
    out["odds_source"] = pd.Series(pd.NA, index=raw.index, dtype="string")

    for source, columns in ODDS_SOURCES:
        if not all(c in raw.columns for c in columns):
            continue
        values = raw.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
        # Decimal odds must be > 1; treat anything else as missing.
        complete = values.notna().all(axis=1) & (values > 1).all(axis=1)
        complete &= out["odds_home_win"].isna()
        if not complete.any():
            continue
        out.loc[complete, ["odds_home_win", "odds_draw", "odds_away_win"]] = (
            values.loc[complete].to_numpy()
        )
        out.loc[complete, "odds_source"] = source

    for source_column, target_column in EXTRA_ODDS_COLUMNS.items():
        if source_column in raw.columns:
            out[target_column] = pd.to_numeric(raw[source_column], errors="coerce")

    out["league"] = league
    out["season"] = season_label(season)
    out["division"] = division
    return out


def main() -> None:
    frames = []
    for season in SEASONS:
        for division, league in LEAGUES:
            raw = download_division(season, division)
            matches = extract_matches(raw, league, season, division)
            downloaded = len(matches)
            matches = matches.dropna(
                subset=[
                    "date",
                    "home_team",
                    "away_team",
                    "home_goals",
                    "away_goals",
                    "winner",
                    "odds_home_win",
                    "odds_draw",
                    "odds_away_win",
                ]
            )
            dropped = downloaded - len(matches)
            note = f" ({dropped} dropped for missing score/odds)" if dropped else ""
            print(
                f"{league} {season_label(season)} ({division}): "
                f"{len(matches)} matches{note}"
            )
            frames.append(matches)

    if not frames:
        raise SystemExit("No leagues configured in LEAGUES.")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"])
    df = df.sort_values(["date", "league"]).reset_index(drop=True)
    df.insert(0, "match_id", range(1, len(df) + 1))
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    leading = [
        "match_id", "date", "time", "league", "season",
        "home_team", "away_team", "home_goals", "away_goals", "winner",
        "odds_home_win", "odds_draw", "odds_away_win", "odds_source",
    ]
    trailing = [c for c in df.columns if c not in leading]
    df = df[leading + trailing]

    df.to_csv(OUTPUT_PATH, index=False)

    counts = df["winner"].value_counts()
    summary = ", ".join(f"{k} {v}" for k, v in counts.items())
    print(f"\nSaved {len(df)} matches to {OUTPUT_PATH}")
    print(f"Winners: {summary}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
