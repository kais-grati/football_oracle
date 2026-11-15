"""Build a pre-match feature dataset for football match prediction.

Every feature for a match is computed exclusively from matches played before
it, so the output can be used for training without temporal leakage.

Features per match
------------------
- att_avg:   decayed average goals scored per match over the team's last
             WINDOW matches (higher is better).
- def_avg:   decayed average goals conceded per match (lower is better).
- mid_avg:   decayed average goal difference per match, used as a proxy for
             midfield control (higher is better).
- gk_rating: decayed clean-sheet percentage over the team's recent matches,
             scaled 0-100 (goalkeeper proxy; higher is better).
- elo:       World Football Elo-style rating, starting at 1500 and updated
             after every match with K=20, a goal-difference multiplier and a
             fixed home advantage of 65 points.
- form5:     average points (3 win / 1 draw / 0 loss) over the team's last
             five matches; NaN until the team has played at least one match.
- prob_home_win / prob_draw / prob_away_win: bookmaker decimal odds
             converted to implied probabilities (1 / odds) and normalized to
             sum to 1, i.e. with the bookmaker overround removed.
- diff_*:    home value minus away value for def, mid and att.

The gk/def/mid/att ratings are result-based proxies by default. Pass
--ratings team_ratings.csv (built by build_player_ratings.py from per-season
FIFA / EA FC player ratings) to replace them with real position-level ratings,
matched by season.

Proxy ratings decay exponentially with match age (half-life HALF_LIFE
matches) and are shrunk toward the league's running averages (weighted as
PRIOR_WEIGHT matches), so every team has a usable pre-match rating from its
first game. Running league averages are computed only from earlier matches of
the same league.

Elo and form5 are always computed from results, so behaviour is identical
whether or not real position ratings are supplied.

target: 0 = home win, 1 = draw, 2 = away win.

Run:
    python build_features.py
    python build_features.py --input football_matches_odds.csv --output match_features.csv
    python build_features.py --ratings team_ratings.csv
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

WINDOW = 10  # matches used for team ratings
HALF_LIFE = 5.0  # match-count half-life of the rating decay
PRIOR_WEIGHT = 5.0  # weight, in matches, of the league-average prior

ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 65.0

DEFAULT_GOALS = 1.35  # goals per team per match before the league has data
DEFAULT_CLEAN_SHEET = 0.25  # clean-sheet rate before the league has data

TARGET_MAP = {"home": 0, "draw": 1, "away": 2}

OUTPUT_COLUMNS = [
    "match_id",
    "date",
    "home_team",
    "away_team",
    "home_gk_rating",
    "home_def_avg",
    "home_mid_avg",
    "home_att_avg",
    "away_gk_rating",
    "away_def_avg",
    "away_mid_avg",
    "away_att_avg",
    "diff_def",
    "diff_mid",
    "diff_att",
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_form5",
    "away_form5",
    "prob_home_win",
    "prob_draw",
    "prob_away_win",
    "target",
]

REQUIRED_COLUMNS = [
    "date",
    "league",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "winner",
    "odds_home_win",
    "odds_draw",
    "odds_away_win",
]

NON_FEATURE_COLUMNS = {"match_id", "date", "home_team", "away_team", "target"}

PROBABILITY_COLUMNS = ["prob_home_win", "prob_draw", "prob_away_win"]


class TeamMatch(NamedTuple):
    goals_for: int
    goals_against: int
    goal_diff: int
    clean_sheet: int


@dataclass
class TeamState:
    elo: float = ELO_START
    recent: deque[TeamMatch] = field(default_factory=lambda: deque(maxlen=WINDOW))
    form: deque[int] = field(default_factory=lambda: deque(maxlen=5))


@dataclass
class LeagueState:
    """Running league totals, used to build leak-free priors."""

    matches: int = 0
    goals: int = 0
    clean_sheets: int = 0

    def priors(self) -> tuple[float, float, float, float]:
        """Priors for goals for, goals against, goal diff and clean sheets."""
        if self.matches == 0:
            goals_per_team = DEFAULT_GOALS
            clean_sheet_rate = DEFAULT_CLEAN_SHEET
        else:
            team_matches = 2 * self.matches
            goals_per_team = self.goals / team_matches
            clean_sheet_rate = self.clean_sheets / team_matches
        return goals_per_team, goals_per_team, 0.0, clean_sheet_rate

    def update(self, home_goals: int, away_goals: int) -> None:
        self.matches += 1
        self.goals += home_goals + away_goals
        self.clean_sheets += int(away_goals == 0) + int(home_goals == 0)


def decayed_mean(matches: deque[TeamMatch], attribute: str, prior: float) -> float:
    """Exponentially weighted mean of a field, shrunk toward a prior."""
    total = 0.0
    weight_sum = 0.0
    for age, match in enumerate(reversed(matches)):
        weight = 0.5 ** (age / HALF_LIFE)
        total += weight * getattr(match, attribute)
        weight_sum += weight
    return (PRIOR_WEIGHT * prior + total) / (PRIOR_WEIGHT + weight_sum)


def team_features(
    state: TeamState, priors: tuple[float, float, float, float]
) -> dict[str, float]:
    goals_for_prior, goals_against_prior, goal_diff_prior, clean_sheet_prior = priors
    form5 = sum(state.form) / len(state.form) if state.form else float("nan")
    return {
        "gk_rating": 100.0 * decayed_mean(state.recent, "clean_sheet", clean_sheet_prior),
        "def_avg": decayed_mean(state.recent, "goals_against", goals_against_prior),
        "mid_avg": decayed_mean(state.recent, "goal_diff", goal_diff_prior),
        "att_avg": decayed_mean(state.recent, "goals_for", goals_for_prior),
        "elo": state.elo,
        "form5": form5,
    }


def pre_match_features(
    state: TeamState,
    priors: tuple[float, float, float, float],
    team_ratings: dict[tuple[str, str, str], dict[str, float]] | None,
    season: str,
    league: str,
    team: str,
) -> dict[str, float]:
    features = team_features(state, priors)
    if team_ratings is not None:
        static = team_ratings.get((season, league, team))
        if static is None:
            raise ValueError(
                f"No pre-built ratings for {team} ({league}, {season}); "
                "run build_player_ratings.py first."
            )
        features.update(static)
    return features


def points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def elo_update(
    home_elo: float, away_elo: float, home_goals: int, away_goals: int
) -> tuple[float, float]:
    """Update both Elo ratings with a World Football Elo-style formula."""
    expected_home = 1.0 / (
        1.0 + 10 ** ((away_elo - (home_elo + ELO_HOME_ADVANTAGE)) / 400.0)
    )
    if home_goals > away_goals:
        score_home = 1.0
    elif home_goals == away_goals:
        score_home = 0.5
    else:
        score_home = 0.0

    goal_diff = abs(home_goals - away_goals)
    if goal_diff <= 1:
        multiplier = 1.0
    elif goal_diff == 2:
        multiplier = 1.5
    else:
        multiplier = (11.0 + goal_diff) / 8.0

    delta = ELO_K * multiplier * (score_home - expected_home)
    return home_elo + delta, away_elo - delta


def implied_probabilities(
    home_odds: float, draw_odds: float, away_odds: float
) -> tuple[float, float, float]:
    """Normalize decimal odds to implied probabilities that sum to one."""
    if any(pd.isna(odds) or odds <= 1 for odds in (home_odds, draw_odds, away_odds)):
        raise ValueError(
            f"Invalid odds: home={home_odds}, draw={draw_odds}, away={away_odds}"
        )
    raw = (1.0 / home_odds, 1.0 / draw_odds, 1.0 / away_odds)
    total = sum(raw)
    return tuple(value / total for value in raw)


def build_features(
    matches: pd.DataFrame,
    team_ratings: dict[tuple[str, str, str], dict[str, float]] | None = None,
) -> pd.DataFrame:
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="raise")
    if matches["date"].isna().any():
        raise ValueError("Input contains unparseable dates.")
    if "match_id" not in matches.columns:
        matches["match_id"] = range(1, len(matches) + 1)

    unknown_winners = set(matches["winner"].dropna().unique()) - set(TARGET_MAP)
    if unknown_winners or matches["winner"].isna().any():
        raise ValueError(f"Unexpected winner values: {sorted(unknown_winners)} or missing.")

    sort_columns = ["date"]
    if "time" in matches.columns:
        sort_columns.append("time")
    sort_columns.append("match_id")
    matches = matches.sort_values(sort_columns, na_position="last").reset_index(drop=True)

    teams: dict[str, TeamState] = {}
    leagues: dict[str, LeagueState] = {}
    rows: list[dict[str, object]] = []

    for match in matches.itertuples(index=False):
        home = teams.setdefault(match.home_team, TeamState())
        away = teams.setdefault(match.away_team, TeamState())
        league = leagues.setdefault(match.league, LeagueState())

        priors = league.priors()
        home_features = pre_match_features(
            home, priors, team_ratings, match.season, match.league, match.home_team
        )
        away_features = pre_match_features(
            away, priors, team_ratings, match.season, match.league, match.away_team
        )

        home_goals = int(match.home_goals)
        away_goals = int(match.away_goals)
        prob_home_win, prob_draw, prob_away_win = implied_probabilities(
            match.odds_home_win, match.odds_draw, match.odds_away_win
        )

        rows.append(
            {
                "match_id": int(match.match_id),
                "date": match.date.strftime("%Y-%m-%d"),
                "home_team": match.home_team,
                "away_team": match.away_team,
                "home_gk_rating": home_features["gk_rating"],
                "home_def_avg": home_features["def_avg"],
                "home_mid_avg": home_features["mid_avg"],
                "home_att_avg": home_features["att_avg"],
                "away_gk_rating": away_features["gk_rating"],
                "away_def_avg": away_features["def_avg"],
                "away_mid_avg": away_features["mid_avg"],
                "away_att_avg": away_features["att_avg"],
                "diff_def": home_features["def_avg"] - away_features["def_avg"],
                "diff_mid": home_features["mid_avg"] - away_features["mid_avg"],
                "diff_att": home_features["att_avg"] - away_features["att_avg"],
                "home_elo": home_features["elo"],
                "away_elo": away_features["elo"],
                "elo_diff": home_features["elo"] - away_features["elo"],
                "home_form5": home_features["form5"],
                "away_form5": away_features["form5"],
                "prob_home_win": prob_home_win,
                "prob_draw": prob_draw,
                "prob_away_win": prob_away_win,
                "target": TARGET_MAP[match.winner],
            }
        )

        # Update state only after the row is recorded, so features stay pre-match.
        home.recent.append(
            TeamMatch(home_goals, away_goals, home_goals - away_goals, int(away_goals == 0))
        )
        away.recent.append(
            TeamMatch(away_goals, home_goals, away_goals - home_goals, int(home_goals == 0))
        )
        home.form.append(points(home_goals, away_goals))
        away.form.append(points(away_goals, home_goals))

        home.elo, away.elo = elo_update(home.elo, away.elo, home_goals, away_goals)
        league.update(home_goals, away_goals)

    features = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    feature_columns = [
        c
        for c in OUTPUT_COLUMNS
        if c not in NON_FEATURE_COLUMNS and c not in PROBABILITY_COLUMNS
    ]
    features[feature_columns] = features[feature_columns].round(3)
    features[PROBABILITY_COLUMNS] = features[PROBABILITY_COLUMNS].round(4)

    # Derive the difference columns from the rounded ratings so that
    # diff_x == home_x - away_x holds exactly in the written CSV.
    for position in ("def", "mid", "att"):
        features[f"diff_{position}"] = (
            features[f"home_{position}_avg"] - features[f"away_{position}_avg"]
        ).round(3)
    features["elo_diff"] = (features["home_elo"] - features["away_elo"]).round(3)
    return features


def load_team_ratings(path: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    ratings = pd.read_csv(path)
    required = ["season", "league", "team", "gk_rating", "def_avg", "mid_avg", "att_avg"]
    missing = [c for c in required if c not in ratings.columns]
    if missing:
        raise SystemExit(f"{path} is missing required columns: {missing}")

    return {
        (row.season, row.league, row.team): {
            "gk_rating": float(row.gk_rating),
            "def_avg": float(row.def_avg),
            "mid_avg": float(row.mid_avg),
            "att_avg": float(row.att_avg),
        }
        for row in ratings.itertuples(index=False)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the pre-match feature dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE_DIR / "football_matches_odds.csv",
        help="Match results CSV (default: football_matches_odds.csv next to this script).",
    )
    parser.add_argument(
        "--ratings",
        type=Path,
        default=None,
        help="Optional per-season team ratings CSV from build_player_ratings.py; "
        "replaces the result-based position proxies.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "match_features.csv",
        help="Where to write the feature dataset (default: match_features.csv).",
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise SystemExit(f"{args.input} is missing required columns: {missing}")

    team_ratings = load_team_ratings(args.ratings) if args.ratings else None
    features = build_features(raw, team_ratings)
    features.to_csv(args.output, index=False)

    counts = features["target"].value_counts()
    labels = {value: name for name, value in TARGET_MAP.items()}
    summary = ", ".join(
        f"{labels[value]} {counts.get(value, 0)}" for value in sorted(TARGET_MAP.values())
    )
    print(f"Saved {len(features)} rows x {len(features.columns)} columns to {args.output}")
    if team_ratings is None:
        print("Position ratings: result-based proxies (pass --ratings team_ratings.csv for per-season FIFA / EA FC ratings)")
    else:
        print(f"Position ratings: {args.ratings}")
    print(f"Targets: {summary}")
    print(f"Missing values: {int(features.isna().sum().sum())} (form5 before a team's first match)")


if __name__ == "__main__":
    main()
