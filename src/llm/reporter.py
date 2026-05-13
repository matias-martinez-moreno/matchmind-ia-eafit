"""Genera reportes narrativos de partidos combinando predicción ML + SHAP + LLM."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.llm.client import GroqClient
from src.llm.prompts import (
    PROMPT_PRE_MATCH,
    PROMPT_EXPLANATION,
    format_shap_features,
)

RESULT_LABELS = {0: "victoria visitante", 1: "empate", 2: "victoria local"}

# Nombres en español para los features (mejor calidad de prompts)
FEATURE_NAMES_ES = {
    "home_recent_wins": "victorias recientes del local",
    "away_recent_wins": "victorias recientes del visitante",
    "home_goals_avg_10": "promedio de goles anotados (local)",
    "home_goals_conceded_avg_10": "promedio de goles recibidos (local)",
    "away_goals_avg_10": "promedio de goles anotados (visitante)",
    "away_goals_conceded_avg_10": "promedio de goles recibidos (visitante)",
    "h2h_home_wins": "victorias historicas del local en H2H",
    "h2h_draws": "empates historicos H2H",
    "h2h_away_wins": "victorias historicas del visitante en H2H",
    "home_team_rank": "ranking FIFA del local",
    "away_team_rank": "ranking FIFA del visitante",
    "rank_diff": "diferencia de ranking FIFA",
    "points_diff": "diferencia de puntos FIFA",
    "neutral": "campo neutral",
}


def get_top_shap_features(explainer, x_row: pd.DataFrame,
                          predicted_class: int, top_k: int = 5) -> list[tuple[str, float]]:
    """Extrae los top-k features con mayor |SHAP value| para una predicción.

    Soporta tanto la API vieja (lista de arrays) como la nueva (array 3D).
    """
    shap_vals = explainer.shap_values(x_row)
    if isinstance(shap_vals, list):
        sv = shap_vals[predicted_class][0]
    else:
        # array 3D: (n_samples, n_features, n_classes)
        sv = shap_vals[0, :, predicted_class]

    idx = np.argsort(np.abs(sv))[::-1][:top_k]
    return [(x_row.columns[j], float(sv[j])) for j in idx]


def generate_match_report(
    home_team: str,
    away_team: str,
    tournament: str,
    date: str,
    home_recent_wins: int,
    away_recent_wins: int,
    home_rank: int,
    away_rank: int,
    h2h_home_wins: int,
    h2h_draws: int,
    h2h_away_wins: int,
    predicted_class: int,
    confidence: float,
    shap_top_features: list,
    client: GroqClient | None = None,
) -> str:
    """Genera un análisis táctico narrativo del partido."""
    if client is None:
        client = GroqClient()

    # Traducir nombres técnicos a español para el prompt
    pretty = [(FEATURE_NAMES_ES.get(name, name), value)
              for name, value in shap_top_features]

    prompt = PROMPT_PRE_MATCH.format(
        home_team=home_team,
        away_team=away_team,
        tournament=tournament,
        date=date,
        home_recent_wins=home_recent_wins,
        away_recent_wins=away_recent_wins,
        home_rank=home_rank,
        away_rank=away_rank,
        h2h_home_wins=h2h_home_wins,
        h2h_draws=h2h_draws,
        h2h_away_wins=h2h_away_wins,
        predicted_result=RESULT_LABELS[predicted_class],
        confidence=confidence,
        shap_top_features=format_shap_features(pretty),
    )
    return client.generate(prompt)


def generate_explanation(
    home_team: str,
    away_team: str,
    predicted_class: int,
    top_3_factors: list,
    client: GroqClient | None = None,
) -> str:
    """Explicación breve de la predicción para aficionado."""
    if client is None:
        client = GroqClient()

    pretty = [(FEATURE_NAMES_ES.get(name, name), value)
              for name, value in top_3_factors]
    factors_text = "\n".join(
        f"- {name}: impacto {value:+.3f}" for name, value in pretty
    )
    prompt = PROMPT_EXPLANATION.format(
        home_team=home_team,
        away_team=away_team,
        predicted_result=RESULT_LABELS[predicted_class],
        top_3_factors=factors_text,
    )
    return client.generate(prompt, max_tokens=200)
