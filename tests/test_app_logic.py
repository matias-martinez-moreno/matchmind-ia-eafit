"""Test end-to-end de la lógica de la demo (sin Streamlit UI)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pickle
import numpy as np
import pandas as pd
import shap

from src.models.train import load_model
from src.llm.client import GroqClient
from src.llm.reporter import generate_match_report, get_top_shap_features


def test_full_pipeline():
    """Replica el flujo de Streamlit para un partido nuevo: Brasil vs Argentina, neutral=False."""
    print("\n=== Test: Brasil vs Argentina ===\n")
    model = load_model("xgboost_model.pkl")
    with open(ROOT / "data" / "processed" / "train_test_split.pkl", "rb") as f:
        split = pickle.load(f)
    all_features = pd.read_csv(ROOT / "data" / "processed" / "features.csv",
                               parse_dates=["date"])
    explainer = shap.TreeExplainer(model)

    home, away = "Brazil", "Argentina"

    # Construir vector
    cols = split["X_test"].columns.tolist()
    x = pd.DataFrame([{c: 0.0 for c in cols}])

    home_games = all_features[all_features["home_team"] == home].sort_values("date")
    row = home_games.iloc[-1]
    x["home_recent_wins"] = row["home_recent_wins"]
    x["home_goals_avg_10"] = row["home_goals_avg_10"]
    x["home_goals_conceded_avg_10"] = row["home_goals_conceded_avg_10"]
    x["home_team_rank"] = row["home_team_rank"]

    away_games = all_features[all_features["away_team"] == away].sort_values("date")
    row = away_games.iloc[-1]
    x["away_recent_wins"] = row["away_recent_wins"]
    x["away_goals_avg_10"] = row["away_goals_avg_10"]
    x["away_goals_conceded_avg_10"] = row["away_goals_conceded_avg_10"]
    x["away_team_rank"] = row["away_team_rank"]

    h2h = all_features[
        ((all_features["home_team"] == home) & (all_features["away_team"] == away))
        | ((all_features["home_team"] == away) & (all_features["away_team"] == home))
    ]
    if not h2h.empty:
        last = h2h.sort_values("date").iloc[-1]
        if last["home_team"] == home:
            x["h2h_home_wins"] = last["h2h_home_wins"]
            x["h2h_away_wins"] = last["h2h_away_wins"]
            x["h2h_draws"] = last["h2h_draws"]
        else:
            x["h2h_home_wins"] = last["h2h_away_wins"]
            x["h2h_away_wins"] = last["h2h_home_wins"]
            x["h2h_draws"] = last["h2h_draws"]

    x["rank_diff"] = x["home_team_rank"] - x["away_team_rank"]
    x["neutral"] = 0

    # Predecir
    proba = model.predict_proba(x[cols])[0]
    pred = int(np.argmax(proba))
    labels = ["Visitante", "Empate", "Local"]

    print(f"Probabilidades: Visitante={proba[0]:.1%}, Empate={proba[1]:.1%}, Local={proba[2]:.1%}")
    print(f"Prediccion: {labels[pred]} ({proba[pred]:.1%})")
    print(f"Vector usado:\n{x[cols].T}")

    # SHAP
    top_features = get_top_shap_features(explainer, x[cols], pred, top_k=5)
    print(f"\nTop 5 features SHAP:")
    for name, value in top_features:
        print(f"  {name}: {value:+.4f}")

    # LLM
    client = GroqClient()
    report = generate_match_report(
        home_team=home, away_team=away,
        tournament="Internacional",
        date="proximo partido",
        home_recent_wins=int(x["home_recent_wins"].values[0]),
        away_recent_wins=int(x["away_recent_wins"].values[0]),
        home_rank=int(x["home_team_rank"].values[0]),
        away_rank=int(x["away_team_rank"].values[0]),
        h2h_home_wins=int(x["h2h_home_wins"].values[0]),
        h2h_draws=int(x["h2h_draws"].values[0]),
        h2h_away_wins=int(x["h2h_away_wins"].values[0]),
        predicted_class=pred,
        confidence=float(proba[pred]),
        shap_top_features=top_features,
        client=client,
    )
    print(f"\n--- Reporte LLM ---\n{report}\n")
    return True


if __name__ == "__main__":
    ok = test_full_pipeline()
    print(f"\n[{'OK' if ok else 'FAIL'}] Test end-to-end de la app")
