"""Demo Streamlit — MatchMind.

Ejecutar: streamlit run app/main.py

Permite seleccionar dos equipos y obtener:
- Probabilidad de cada resultado
- Gráfico SHAP de los factores más influyentes
- Reporte narrativo generado por LLM (Groq + Llama-3.1-8b)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

from src.models.train import load_model
from src.llm.client import GroqClient
from src.llm.reporter import (
    generate_match_report,
    get_top_shap_features,
    FEATURE_NAMES_ES,
)

st.set_page_config(page_title="MatchMind", page_icon="⚽", layout="wide")

st.title("⚽ MatchMind")
st.caption("Predicción de resultados de partidos internacionales · IA EAFIT 2026-1")


@st.cache_resource
def load_resources():
    model = load_model("xgboost_model.pkl")
    with open(ROOT / "data" / "processed" / "train_test_split.pkl", "rb") as f:
        split = pickle.load(f)
    # Tambien cargamos features.csv para buscar estadisticas historicas
    features = pd.read_csv(ROOT / "data" / "processed" / "features.csv",
                           parse_dates=["date"])
    explainer = shap.TreeExplainer(model)
    return model, split, features, explainer


@st.cache_resource
def get_llm_client():
    return GroqClient()


try:
    model, split, all_features, explainer = load_resources()
except FileNotFoundError:
    st.error("No se encontró el modelo. Ejecuta primero scripts/run_preprocessing.py "
             "y scripts/run_tuning_shap.py")
    st.stop()

# Lista combinada de equipos del dataset completo, no solo del test
teams = sorted(set(all_features["home_team"]) | set(all_features["away_team"]))


col1, col2 = st.columns(2)
with col1:
    home = st.selectbox(
        "Equipo local",
        teams,
        index=teams.index("Brazil") if "Brazil" in teams else 0,
    )
with col2:
    away = st.selectbox(
        "Equipo visitante",
        teams,
        index=teams.index("Argentina") if "Argentina" in teams else 1,
    )

is_neutral = st.checkbox("Campo neutral", value=False)
tournament = st.text_input("Tipo de torneo (para el reporte)", value="Internacional")

st.divider()


def build_x_for_match(home: str, away: str, is_neutral: bool) -> pd.DataFrame:
    """Construye un vector de features para un partido entre home y away.

    Estrategia: tomar la estadística MAS RECIENTE de cada equipo en el dataset.
    Esto es lo que el modelo usaría si tuviera que predecir un nuevo partido.
    """
    cols = split["X_test"].columns.tolist()
    x = pd.DataFrame([{c: 0.0 for c in cols}])

    # Buscar el partido mas reciente del home como local
    home_games = all_features[all_features["home_team"] == home].sort_values("date")
    if not home_games.empty:
        row = home_games.iloc[-1]
        x["home_recent_wins"] = row["home_recent_wins"]
        x["home_goals_avg_10"] = row["home_goals_avg_10"]
        x["home_goals_conceded_avg_10"] = row["home_goals_conceded_avg_10"]
        x["home_team_rank"] = row["home_team_rank"]

    # Buscar el partido mas reciente del away como visitante
    away_games = all_features[all_features["away_team"] == away].sort_values("date")
    if not away_games.empty:
        row = away_games.iloc[-1]
        x["away_recent_wins"] = row["away_recent_wins"]
        x["away_goals_avg_10"] = row["away_goals_avg_10"]
        x["away_goals_conceded_avg_10"] = row["away_goals_conceded_avg_10"]
        x["away_team_rank"] = row["away_team_rank"]

    # H2H historico
    h2h = all_features[
        ((all_features["home_team"] == home) & (all_features["away_team"] == away))
        | ((all_features["home_team"] == away) & (all_features["away_team"] == home))
    ]
    if not h2h.empty:
        last_h2h = h2h.sort_values("date").iloc[-1]
        # los valores acumulados estan al ultimo partido jugado entre ellos
        # Si el ultimo partido fue home-away igual, los valores son los buenos.
        # Si fue invertido, hay que swapear.
        if last_h2h["home_team"] == home:
            x["h2h_home_wins"] = last_h2h["h2h_home_wins"] + (1 if last_h2h["result"] == 2 else 0)
            x["h2h_away_wins"] = last_h2h["h2h_away_wins"] + (1 if last_h2h["result"] == 0 else 0)
            x["h2h_draws"] = last_h2h["h2h_draws"] + (1 if last_h2h["result"] == 1 else 0)
        else:
            x["h2h_home_wins"] = last_h2h["h2h_away_wins"] + (1 if last_h2h["result"] == 0 else 0)
            x["h2h_away_wins"] = last_h2h["h2h_home_wins"] + (1 if last_h2h["result"] == 2 else 0)
            x["h2h_draws"] = last_h2h["h2h_draws"] + (1 if last_h2h["result"] == 1 else 0)

    # Diferencias
    x["rank_diff"] = x["home_team_rank"] - x["away_team_rank"]
    x["neutral"] = int(is_neutral)
    return x[cols]


if st.button("Predecir resultado", type="primary"):
    x = build_x_for_match(home, away, is_neutral)

    proba = model.predict_proba(x)[0]
    pred = int(np.argmax(proba))
    labels = ["Visitante gana", "Empate", "Local gana"]
    color_pred = ["#9b1b1b", "#b45309", "#006b3c"][pred]

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Gana {away}", f"{proba[0]:.1%}")
    c2.metric("Empate", f"{proba[1]:.1%}")
    c3.metric(f"Gana {home}", f"{proba[2]:.1%}")

    st.markdown(
        f"<h4 style='color:{color_pred};'>Predicción: {labels[pred]} "
        f"(confianza {proba[pred]:.1%})</h4>",
        unsafe_allow_html=True,
    )

    # SHAP
    with st.spinner("Calculando factores influyentes..."):
        top_features = get_top_shap_features(explainer, x, pred, top_k=8)

    st.subheader("Factores más influyentes (SHAP)")
    fig, ax = plt.subplots(figsize=(8, 4))
    names_es = [FEATURE_NAMES_ES.get(n, n) for n, _ in top_features]
    values = [v for _, v in top_features]
    colors = ["#006b3c" if v > 0 else "#9b1b1b" for v in values]
    ax.barh(names_es, values, color=colors)
    ax.set_xlabel("SHAP value (positivo favorece la clase predicha)")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.invert_yaxis()
    st.pyplot(fig)

    # LLM
    st.subheader("Análisis táctico generado por LLM")
    with st.spinner("Generando análisis con Llama-3.1-8b vía Groq..."):
        try:
            llm_client = get_llm_client()
            top5 = top_features[:5]
            report = generate_match_report(
                home_team=home, away_team=away,
                tournament=tournament,
                date="próximo partido",
                home_recent_wins=int(x["home_recent_wins"].values[0]),
                away_recent_wins=int(x["away_recent_wins"].values[0]),
                home_rank=int(x["home_team_rank"].values[0]),
                away_rank=int(x["away_team_rank"].values[0]),
                h2h_home_wins=int(x["h2h_home_wins"].values[0]),
                h2h_draws=int(x["h2h_draws"].values[0]),
                h2h_away_wins=int(x["h2h_away_wins"].values[0]),
                predicted_class=pred,
                confidence=float(proba[pred]),
                shap_top_features=top5,
                client=llm_client,
            )
            st.info(report)
        except RuntimeError as e:
            st.error(f"Error LLM: {e}. Verifica que .env tenga GROQ_API_KEY.")

st.divider()
with st.expander("Acerca del modelo"):
    st.markdown(
        """
**MatchMind** usa XGBoost entrenado sobre 30,511 partidos internacionales (1993–2026).

- **Métricas en test (5,876 partidos)**: Accuracy 0.55, F1-macro 0.45, AUC 0.71
- **Top feature**: diferencia de ranking FIFA (impacto SHAP 0.385)
- **LLM**: Llama-3.1-8b-instant vía Groq API (gratuito)
- **Limitación**: el modelo subpredice empates (1,360 reales vs 578 predichos en test)
        """
    )
