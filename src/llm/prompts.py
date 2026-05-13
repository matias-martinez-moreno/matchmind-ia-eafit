"""Prompts del LLM usados en el sistema MatchMind.

Estos prompts están documentados y forman parte de la arquitectura
(ver sección de Buenas Prácticas en la guía: los prompts no son detalle,
son parte central del sistema).
"""

# Prompt 1: Análisis pre-partido completo usando datos del modelo + SHAP.
PROMPT_PRE_MATCH = """Eres un analista táctico de fútbol experto en
selecciones nacionales. Analiza el siguiente partido internacional y genera
un pronóstico narrativo en español (MÁXIMO 150 palabras).

DATOS DEL PARTIDO:
- Equipo local: {home_team}
- Equipo visitante: {away_team}
- Torneo: {tournament}
- Fecha: {date}

ESTADÍSTICAS RECIENTES:
- {home_team}: {home_recent_wins}/5 victorias recientes, ranking FIFA {home_rank}
- {away_team}: {away_recent_wins}/5 victorias recientes, ranking FIFA {away_rank}
- Head-to-head: {h2h_home_wins} victorias {home_team}, {h2h_draws} empates, {h2h_away_wins} victorias {away_team}

PREDICCIÓN DEL MODELO ML:
- Resultado más probable: {predicted_result} (probabilidad: {confidence:.1%})

FACTORES MÁS INFLUYENTES (SHAP values):
{shap_top_features}

INSTRUCCIONES:
1. Identifica el favorito y justifica con los datos.
2. Menciona 1-2 factores clave decisivos.
3. Termina con una predicción narrativa concisa.
4. NO inventes información que no esté en el contexto.
5. Tono: profesional, conciso, en español.
"""


# Prompt 2: Explicación breve y didáctica de la predicción para aficionado.
PROMPT_EXPLANATION = """Eres un comentarista deportivo. Explica en
lenguaje sencillo (MÁXIMO 80 palabras, en español) por qué el modelo
predice que el resultado más probable será: {predicted_result}
en el partido {home_team} vs {away_team}.

Los 3 factores más importantes según el modelo son:
{top_3_factors}

Explícalo como si hablaras con un aficionado al fútbol, sin jerga técnica.
NO inventes información fuera del contexto.
"""


def format_shap_features(top_features: list[tuple[str, float]]) -> str:
    """Convierte una lista de (feature_name, shap_value) en texto para el prompt."""
    lines = []
    for name, value in top_features:
        direction = "favorece local" if value > 0 else "favorece visitante"
        lines.append(f"- {name}: {value:+.3f} ({direction})")
    return "\n".join(lines)
