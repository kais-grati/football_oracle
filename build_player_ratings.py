"""Build per-season team position ratings from FIFA / EA FC player ratings.

One ratings snapshot per season, so a season's matches are rated with the
squad ratings known before that season started:

- 2021/22: FIFA 22 complete player dataset (CC0)
- 2022/23: FIFA 23 player data (CC0)
- 2023/24: EA Sports FC 24 complete player dataset (CC0)
- 2024/25: EA Sports FC 25 database (Apache 2.0)
- 2025/26: EA Sports FC 26 player ratings (CC0)

Datasets are downloaded from Kaggle on first use and cached under data/.

For every club in each season's matches, the script selects a likely starting
XI in a 4-3-3 and aggregates the players' overall ratings:

- gk_rating: best goalkeeper's rating
- def_avg:   mean rating of the four best defenders (CB / LB / RB / LWB / RWB)
- mid_avg:   mean rating of the three best central midfielders (CDM / CM / CAM)
- att_avg:   mean rating of the three best attackers (ST / CF / LW / RW / LM / RM)

Output columns: season, league, team, gk_rating, def_avg, mid_avg, att_avg.

Run:
    python build_player_ratings.py
"""

from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
MATCHES_PATH = BASE_DIR / "football_matches_odds.csv"
OUTPUT_PATH = BASE_DIR / "team_ratings.csv"

POSITION_GROUPS = {
    "gk": {"GK"},
    "def": {"CB", "LB", "RB", "LWB", "RWB"},
    "mid": {"CDM", "CM", "CAM"},
    "att": {"ST", "CF", "LW", "RW", "LM", "RM"},
}

# Starting XI shape: 4-3-3.
STARTERS = {"gk": 1, "def": 4, "mid": 3, "att": 3}

OUTPUT_COLUMNS = ["season", "league", "team", "gk_rating", "def_avg", "mid_avg", "att_avg"]

KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/{ref}"
DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

FC24_LEAGUES = {"La Liga", "Premier League", "Bundesliga", "Serie A", "Ligue 1"}
FIFA22_LEAGUES = {
    "Spain Primera Division",
    "English Premier League",
    "German 1. Bundesliga",
    "Italian Serie A",
    "French Ligue 1",
}
EA_LEAGUES = {
    "LALIGA EA SPORTS",
    "Premier League",
    "Bundesliga",
    "Serie A Enilive",
    "Ligue 1 McDonald's",
}


def normalize_name(value: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = text.lower().replace("&", " and ").replace(".", "").replace("'", "").replace("’", "")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def first_position(values: pd.Series) -> pd.Series:
    return values.astype("string").str.split(",").str[0].str.strip().str.upper()


def normalized_players(
    players: pd.Series, clubs: pd.Series, positions: pd.Series, overall: pd.Series
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "player": players.astype("string").str.strip(),
            "club": clubs.astype("string").str.strip(),
            "position": positions.astype("string").str.strip().str.upper(),
            "overall": pd.to_numeric(overall, errors="coerce"),
        }
    )
    return frame.dropna(subset=["player", "club", "position", "overall"])


def load_fifa22(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        usecols=["short_name", "club_name", "league_name", "player_positions", "overall"],
        low_memory=False,
    )
    raw = raw[raw["league_name"].isin(FIFA22_LEAGUES)]
    return normalized_players(
        raw["short_name"], raw["club_name"], first_position(raw["player_positions"]),
        raw["overall"],
    )


def load_fifa23(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    return normalized_players(
        raw["Known As"], raw["Club Name"], raw["Best Position"], raw["Overall"]
    )


def load_fc24(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        usecols=["fifa_version", "short_name", "club_name", "league_name",
                 "player_positions", "overall"],
        low_memory=False,
    )
    raw = raw[(raw["fifa_version"] == 24) & raw["league_name"].isin(FC24_LEAGUES)]
    return normalized_players(
        raw["short_name"], raw["club_name"], first_position(raw["player_positions"]),
        raw["overall"],
    )


def load_fc25(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, usecols=["Name", "Team", "League", "Position", "OVR"])
    raw = raw[raw["League"].isin(EA_LEAGUES)]
    return normalized_players(raw["Name"], raw["Team"], raw["Position"], raw["OVR"])


def load_fc26(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        path, usecols=["commonName", "firstName", "lastName", "team", "leagueName",
                       "position", "overallRating"]
    )
    raw = raw[raw["leagueName"].isin(EA_LEAGUES)]
    names = raw["commonName"].fillna(
        raw["firstName"].astype("string") + " " + raw["lastName"].astype("string")
    )
    return normalized_players(names, raw["team"], raw["position"], raw["overallRating"])


@dataclass(frozen=True)
class DataSource:
    kaggle_ref: str
    directory: str
    filename: str
    loader: Callable[[Path], pd.DataFrame]


SOURCES = {
    "2021-22": DataSource(
        "stefanoleone992/fifa-22-complete-player-dataset",
        "data/fifa22_full",
        "players_22.csv",
        load_fifa22,
    ),
    "2022-23": DataSource(
        "babatundezenith/fifa-archive",
        "data/fifa23",
        "Fifa_23_Players_Data.csv",
        load_fifa23,
    ),
    "2023-24": DataSource(
        "stefanoleone992/ea-sports-fc-24-complete-player-dataset",
        "data/fc24",
        "male_players.csv",
        load_fc24,
    ),
    "2024-25": DataSource(
        "nyagami/ea-sports-fc-25-database-ratings-and-stats",
        "data/fc25",
        "male_players.csv",
        load_fc25,
    ),
    "2025-26": DataSource(
        "justdhia/ea-sports-fc-26-player-ratings",
        "data/fc26",
        "ea_fc26_players.csv",
        load_fc26,
    ),
}

# football-data.co.uk team name -> club names used by the ratings datasets.
TEAM_ALIASES = {
    # Spain
    "Alaves": ["deportivo alaves", "d alaves", "alaves"],
    "Almeria": ["ud almeria", "union deportiva almeria", "almeria"],
    "Ath Bilbao": ["athletic club de bilbao", "athletic club", "athletic bilbao"],
    "Ath Madrid": ["atletico de madrid", "atletico madrid"],
    "Barcelona": ["fc barcelona", "barcelona"],
    "Betis": ["real betis balompie", "real betis"],
    "Cadiz": ["cadiz cf", "cadiz"],
    "Celta": ["rc celta de vigo", "rc celta", "celta de vigo", "celta"],
    "Elche": ["elche cf", "elche"],
    "Espanol": ["rcd espanyol de barcelona", "rcd espanyol", "espanyol"],
    "Getafe": ["getafe cf", "getafe"],
    "Girona": ["girona fc", "girona"],
    "Granada": ["granada cf", "granada"],
    "Las Palmas": ["ud las palmas", "las palmas"],
    "Leganes": ["cd leganes", "leganes"],
    "Levante": ["levante union deportiva", "levante ud", "levante"],
    "Mallorca": ["rcd mallorca", "mallorca"],
    "Osasuna": ["ca osasuna", "osasuna"],
    "Oviedo": ["r oviedo", "real oviedo", "oviedo"],
    "Real Madrid": ["real madrid cf", "real madrid"],
    "Sociedad": ["real sociedad"],
    "Sevilla": ["sevilla fc", "sevilla"],
    "Valencia": ["valencia cf", "valencia"],
    "Valladolid": ["real valladolid cf", "real valladolid", "r valladolid cf", "valladolid"],
    "Vallecano": ["rayo vallecano"],
    "Villarreal": ["villarreal cf", "villarreal"],
    # Germany
    "Augsburg": ["fc augsburg", "augsburg"],
    "Bayern Munich": ["fc bayern munchen", "bayern munchen", "bayern munich"],
    "Bielefeld": ["dsc arminia bielefeld", "arminia bielefeld", "bielefeld"],
    "Bochum": ["vfl bochum 1848", "vfl bochum", "bochum"],
    "Darmstadt": ["darmstadt 98", "sv darmstadt 98", "darmstadt"],
    "Dortmund": ["borussia dortmund", "dortmund"],
    "Ein Frankfurt": ["eintracht frankfurt", "frankfurt"],
    "FC Koln": ["1 fc koln", "fc koln", "koln", "cologne"],
    "Freiburg": ["sport club freiburg", "sc freiburg", "freiburg"],
    "Greuther Furth": ["spvgg greuther furth", "greuther furth"],
    "Hamburg": ["hamburger sv", "hamburg"],
    "Heidenheim": ["1 fc heidenheim 1846", "1 fc heidenheim", "fc heidenheim", "heidenheim"],
    "Hertha": ["hertha bsc", "hertha"],
    "Hoffenheim": ["tsg 1899 hoffenheim", "tsg hoffenheim", "hoffenheim"],
    "Holstein Kiel": ["holstein kiel", "kiel"],
    "Leverkusen": ["bayer 04 leverkusen", "bayer leverkusen", "leverkusen"],
    "M'gladbach": [
        "borussia monchengladbach", "borussia mgladbach", "monchengladbach", "mgladbach"
    ],
    "Mainz": ["1 fsv mainz 05", "fsv mainz 05", "mainz 05", "mainz"],
    "RB Leipzig": ["rb leipzig", "leipzig"],
    "Schalke 04": ["fc schalke 04", "schalke 04", "schalke"],
    "St Pauli": ["fc st pauli", "st pauli"],
    "Stuttgart": ["vfb stuttgart", "stuttgart"],
    "Union Berlin": ["1 fc union berlin", "fc union berlin", "union berlin"],
    "Werder Bremen": ["sv werder bremen", "werder bremen"],
    "Wolfsburg": ["vfl wolfsburg", "wolfsburg"],
    # England
    "Arsenal": ["arsenal"],
    "Aston Villa": ["aston villa"],
    "Bournemouth": ["afc bournemouth", "bournemouth"],
    "Brentford": ["brentford"],
    "Brighton": ["brighton and hove albion", "brighton hove albion", "brighton"],
    "Burnley": ["burnley"],
    "Chelsea": ["chelsea"],
    "Crystal Palace": ["crystal palace"],
    "Everton": ["everton"],
    "Fulham": ["fulham"],
    "Ipswich": ["ipswich town", "ipswich"],
    "Leeds": ["leeds united", "leeds"],
    "Leicester": ["leicester city", "leicester"],
    "Liverpool": ["liverpool"],
    "Luton": ["luton town", "luton"],
    "Man City": ["manchester city"],
    "Man United": ["manchester united", "man utd"],
    "Newcastle": ["newcastle united", "newcastle utd", "newcastle"],
    "Norwich": ["norwich city", "norwich"],
    "Nott'm Forest": ["nottingham forest", "nottm forest"],
    "Sheffield United": ["sheffield united", "sheffield utd"],
    "Southampton": ["southampton"],
    "Sunderland": ["sunderland"],
    "Tottenham": ["tottenham hotspur", "spurs", "tottenham"],
    "Watford": ["watford"],
    "West Ham": ["west ham united", "west ham"],
    "Wolves": ["wolverhampton wanderers", "wolves"],
    # France
    "Ajaccio": ["ac ajaccio", "ajaccio"],
    "Angers": ["angers sco", "angers"],
    "Auxerre": ["aj auxerre", "auxerre"],
    "Bordeaux": ["fc girondins de bordeaux", "girondins de bordeaux", "girondins bordeaux", "bordeaux"],
    "Brest": ["stade brestois 29", "stade brestois", "brest"],
    "Clermont": ["clermont foot 63", "clermont foot", "clermont"],
    "Le Havre": ["le havre ac", "le havre", "havre ac", "havre"],
    "Lens": ["racing club de lens", "rc lens", "lens"],
    "Lille": ["losc lille", "lille"],
    "Lorient": ["fc lorient", "lorient"],
    "Lyon": ["olympique lyonnais", "lyon", "ol"],
    "Marseille": ["olympique de marseille", "olympique marseille", "marseille", "om"],
    "Metz": ["fc metz", "metz"],
    "Monaco": ["as monaco", "monaco"],
    "Montpellier": ["montpellier herault sc", "montpellier hsc", "montpellier herault", "montpellier"],
    "Nantes": ["fc nantes", "nantes"],
    "Nice": ["ogc nice", "nice"],
    "Paris FC": ["paris fc"],
    "Paris SG": ["paris saint germain", "paris sg", "psg"],
    "Reims": ["stade de reims", "reims"],
    "Rennes": ["stade rennais fc", "stade rennais", "rennes"],
    "St Etienne": ["as saint etienne", "saint etienne", "st etienne"],
    "Strasbourg": ["rc strasbourg alsace", "rc strasbourg", "strasbourg"],
    "Toulouse": ["toulouse football club", "toulouse fc", "toulouse"],
    "Troyes": ["estac troyes", "troyes"],
    # Italy
    "Atalanta": ["atalanta", "bergamo calcio"],
    "Bologna": ["bologna"],
    "Cagliari": ["cagliari"],
    "Como": ["como"],
    "Cremonese": ["us cremonese", "cremonese"],
    "Empoli": ["empoli"],
    "Fiorentina": ["fiorentina"],
    "Frosinone": ["frosinone"],
    "Genoa": ["genoa"],
    "Inter": ["internazionale", "inter", "lombardia fc"],
    "Juventus": ["juventus"],
    "Lazio": ["lazio", "latium"],
    "Lecce": ["lecce"],
    "Milan": ["ac milan", "milan", "milano fc"],
    "Monza": ["ac monza", "monza"],
    "Napoli": ["ssc napoli", "napoli"],
    "Parma": ["parma"],
    "Pisa": ["pisa"],
    "Roma": ["as roma", "roma"],
    "Salernitana": ["us salernitana 1919", "us salernitana", "salernitana"],
    "Sampdoria": ["uc sampdoria", "sampdoria"],
    "Sassuolo": ["us sassuolo calcio", "sassuolo"],
    "Spezia": ["la spezia", "spezia"],
    "Torino": ["torino fc", "torino"],
    "Udinese": ["udinese calcio", "udinese"],
    "Venezia": ["venezia fc", "venezia"],
    "Verona": ["hellas verona", "verona"],
}


def download_dataset(source: DataSource) -> Path:
    directory = BASE_DIR / source.directory
    path = directory / source.filename
    if path.exists():
        return path

    directory.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {source.kaggle_ref}...")
    response = requests.get(
        KAGGLE_URL.format(ref=source.kaggle_ref), headers=DOWNLOAD_HEADERS, timeout=600
    )
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for member in archive.namelist():
            parts = Path(member).parts
            if member.endswith("/") or ".." in parts or Path(member).is_absolute():
                continue
            archive.extract(member, directory)

    if not path.exists():
        raise RuntimeError(f"{source.filename} not found after downloading {source.kaggle_ref}")
    return path


def map_teams(teams: list[str], players: pd.DataFrame) -> dict[str, str]:
    index: dict[str, set[str]] = {}
    for club in players["club"].unique():
        index.setdefault(normalize_name(club), set()).add(club)

    mapping, problems = {}, []
    for team in teams:
        aliases = TEAM_ALIASES.get(team, [normalize_name(team)])
        matches = set()
        for alias in aliases:
            matches |= index.get(normalize_name(alias), set())
        if len(matches) == 1:
            mapping[team] = matches.pop()
        elif not matches:
            problems.append(f"{team}: no club matches {aliases}")
        else:
            problems.append(f"{team}: ambiguous between {sorted(matches)}")

    if problems:
        raise ValueError("Team mapping failed:\n  " + "\n  ".join(problems))
    return mapping


def group_rating(squad: pd.DataFrame, group: str) -> float:
    candidates = squad[squad["position"].isin(POSITION_GROUPS[group])]
    if candidates.empty:
        raise ValueError(f"no players for position group '{group}'")

    starters = candidates.sort_values(
        ["overall", "player"], ascending=[False, True]
    ).head(STARTERS[group])
    return round(float(starters["overall"].mean()), 3)


def season_team_leagues(matches: pd.DataFrame) -> dict[str, str]:
    leagues = {}
    for row in matches.itertuples(index=False):
        leagues[row.home_team] = row.league
        leagues[row.away_team] = row.league
    return leagues


def build_ratings(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in sorted(matches["season"].unique()):
        source = SOURCES.get(season)
        if source is None:
            raise ValueError(f"No ratings source configured for season {season}")

        players = source.loader(download_dataset(source))
        team_leagues = season_team_leagues(matches[matches["season"] == season])
        mapping = map_teams(sorted(team_leagues), players)

        for team, league in sorted(team_leagues.items()):
            club = mapping[team]
            squad = players[players["club"] == club]
            rows.append(
                {
                    "season": season,
                    "league": league,
                    "team": team,
                    "gk_rating": group_rating(squad, "gk"),
                    "def_avg": group_rating(squad, "def"),
                    "mid_avg": group_rating(squad, "mid"),
                    "att_avg": group_rating(squad, "att"),
                }
            )
        print(f"{season}: {len(team_leagues)} teams rated")

    ratings = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    ratings[["gk_rating", "def_avg", "mid_avg", "att_avg"]] = ratings[
        ["gk_rating", "def_avg", "mid_avg", "att_avg"]
    ].round(3)
    return ratings.sort_values(["season", "league", "team"]).reset_index(drop=True)


def main() -> None:
    matches = pd.read_csv(MATCHES_PATH)
    ratings = build_ratings(matches)
    ratings.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved ratings for {len(ratings)} team-seasons to {OUTPUT_PATH}")
    print(
        ratings[["gk_rating", "def_avg", "mid_avg", "att_avg"]]
        .describe()
        .loc[["min", "mean", "max"]]
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
