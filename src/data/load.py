"""Carga de datasets crudos."""
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_results(filename: str = "results.csv") -> pd.DataFrame:
    """Carga el dataset de resultados de partidos internacionales.

    Columnas esperadas: date, home_team, away_team, home_score, away_score,
    tournament, city, country, neutral.
    """
    path = RAW_DIR / filename
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_fifa_ranking(filename: str = "fifa_ranking.csv") -> pd.DataFrame:
    """Carga el dataset de ranking FIFA histórico.

    Columnas esperadas: rank, country_full, total_points, rank_date.
    """
    path = RAW_DIR / filename
    df = pd.read_csv(path, parse_dates=["rank_date"])
    return df


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """Genera la columna target a partir de los goles.

    0 = away_win, 1 = draw, 2 = home_win.
    """
    df = df.copy()
    conditions = [
        df["home_score"] > df["away_score"],
        df["home_score"] == df["away_score"],
        df["home_score"] < df["away_score"],
    ]
    choices = [2, 1, 0]
    df["result"] = np.select(conditions, choices)
    return df
