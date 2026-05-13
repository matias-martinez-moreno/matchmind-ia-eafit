"""Integración LLM (Groq + Llama-3.1-8b) y evaluación cuantitativa.

1. Genera 3 reportes narrativos para partidos icónicos del test set.
2. Evalúa cuantitativamente el LLM sobre 20 casos con criterios automáticos.
3. Guarda los resultados en docs/llm_outputs/ y un resumen en docs/.

Ejecutar: python scripts/run_llm_integration.py
"""
import sys
import time
import pickle
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import shap

from src.models.train import load_model
from src.llm.client import GroqClient
from src.llm.reporter import generate_match_report, get_top_shap_features
from src.evaluation.llm_eval import (
    LLMTestCase, evaluate_response, summarize,
)

OUT_DIR = ROOT / "docs" / "llm_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def section(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")


# ============================================================
section("1. CARGAR MODELO, SPLIT Y CLIENTE GROQ")
# ============================================================
xgb = load_model("xgboost_model.pkl")
with open(ROOT / "data" / "processed" / "train_test_split.pkl", "rb") as f:
    split = pickle.load(f)

X_test = split["X_test"]
y_test = split["y_test"]
meta_test = split["meta_test"]

try:
    client = GroqClient()
    print(f"Cliente Groq OK. Modelo: {client.model}")
except RuntimeError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

explainer = shap.TreeExplainer(xgb)
print(f"Test set: {len(X_test):,} partidos")


# ============================================================
section("2. SELECCIONAR PARTIDOS ICONICOS PARA EJEMPLOS")
# ============================================================
icons = [
    ("Brazil", "Argentina"),
    ("Germany", "France"),
    ("Spain", "Italy"),
    ("Colombia", "Argentina"),
    ("Mexico", "United States"),
    ("Portugal", "Spain"),
    ("England", "Germany"),
    ("Netherlands", "Belgium"),
]

selected = []
for h, a in icons:
    mask = (meta_test["home_team"] == h) & (meta_test["away_team"] == a)
    if mask.any():
        idx = meta_test[mask].index[0]
        selected.append((h, a, idx))

# Si no hay 3, llenamos con los primeros del test
i = 0
while len(selected) < 3:
    row = meta_test.iloc[i]
    if (row["home_team"], row["away_team"], i) not in selected:
        selected.append((row["home_team"], row["away_team"], i))
    i += 1

print(f"Encontrados {len(selected)} partidos iconicos.")
selected_demo = selected[:3]


# ============================================================
section("3. GENERAR 3 REPORTES DE EJEMPLO")
# ============================================================
demo_results = []

for h, a, idx in selected_demo:
    row = meta_test.iloc[idx]
    x = X_test.iloc[[idx]]
    proba = xgb.predict_proba(x)[0]
    pred = int(np.argmax(proba))
    true_label = int(y_test[idx])

    top_features = get_top_shap_features(explainer, x, pred, top_k=5)

    print(f"\n--- {h} vs {a} ({row['date'].date()}, {row['tournament']}) ---")
    print(f"Real: {['Visitante','Empate','Local'][true_label]} | "
          f"Predicho: {['Visitante','Empate','Local'][pred]} ({proba[pred]:.1%})")

    t0 = time.time()
    try:
        report = generate_match_report(
            home_team=h, away_team=a,
            tournament=row["tournament"],
            date=str(row["date"].date()),
            home_recent_wins=int(x["home_recent_wins"].values[0]),
            away_recent_wins=int(x["away_recent_wins"].values[0]),
            home_rank=int(x["home_team_rank"].values[0]),
            away_rank=int(x["away_team_rank"].values[0]),
            h2h_home_wins=int(x["h2h_home_wins"].values[0]),
            h2h_draws=int(x["h2h_draws"].values[0]),
            h2h_away_wins=int(x["h2h_away_wins"].values[0]),
            predicted_class=pred,
            confidence=proba[pred],
            shap_top_features=top_features,
            client=client,
        )
        dt = time.time() - t0
        print(f"\n[Latencia: {dt:.1f}s]\n{report}\n")
        demo_results.append({
            "home": h, "away": a,
            "date": str(row["date"].date()),
            "tournament": row["tournament"],
            "predicted": pred,
            "real": true_label,
            "confidence": float(proba[pred]),
            "report": report,
            "latency_sec": round(dt, 2),
        })
    except Exception as e:
        print(f"ERROR en {h} vs {a}: {e}")

# Guardar
with open(OUT_DIR / "demo_reports.json", "w", encoding="utf-8") as f:
    json.dump(demo_results, f, indent=2, ensure_ascii=False)
print(f"\nReportes guardados: {OUT_DIR / 'demo_reports.json'}")


# ============================================================
section("4. EVALUACION CUANTITATIVA — 20 CASOS")
# ============================================================
n_cases = 20
# Tomamos partidos variados del test, no solo los primeros
sample_idx = np.linspace(0, len(X_test) - 1, n_cases, dtype=int)

eval_results = []
print(f"Evaluando {n_cases} casos del test set...\n")

for i, idx in enumerate(sample_idx):
    row = meta_test.iloc[idx]
    x = X_test.iloc[[idx]]
    proba = xgb.predict_proba(x)[0]
    pred = int(np.argmax(proba))

    favorite = row["home_team"] if pred == 2 else (row["away_team"] if pred == 0 else "empate")
    top_features = get_top_shap_features(explainer, x, pred, top_k=3)

    try:
        report = generate_match_report(
            home_team=row["home_team"], away_team=row["away_team"],
            tournament=row["tournament"],
            date=str(row["date"].date()),
            home_recent_wins=int(x["home_recent_wins"].values[0]),
            away_recent_wins=int(x["away_recent_wins"].values[0]),
            home_rank=int(x["home_team_rank"].values[0]),
            away_rank=int(x["away_team_rank"].values[0]),
            h2h_home_wins=int(x["h2h_home_wins"].values[0]),
            h2h_draws=int(x["h2h_draws"].values[0]),
            h2h_away_wins=int(x["h2h_away_wins"].values[0]),
            predicted_class=pred, confidence=float(proba[pred]),
            shap_top_features=top_features, client=client,
        )

        # Mapear nombres tecnicos a español para chequear
        factors_es = []
        for fname, _ in top_features:
            # ej. h2h_home_wins -> "h2h", "victorias"
            if "h2h" in fname:
                factors_es.append("h2h")
                factors_es.append("historico")
            if "rank" in fname:
                factors_es.append("ranking")
                factors_es.append("FIFA")
            if "goals" in fname:
                factors_es.append("goles")
            if "recent" in fname:
                factors_es.append("racha")
                factors_es.append("recientes")
            if "neutral" in fname:
                factors_es.append("neutral")
            if "points" in fname:
                factors_es.append("puntos")

        tc = LLMTestCase(
            home_team=row["home_team"],
            away_team=row["away_team"],
            expected_favorite=favorite,
            expected_factors=list(set(factors_es)),
        )
        checks = evaluate_response(report, tc)
        eval_results.append({
            "idx": int(idx),
            "match": f"{row['home_team']} vs {row['away_team']}",
            "date": str(row["date"].date()),
            "pred": pred, "favorite": favorite,
            "checks": checks,
            "report_length_words": len(report.split()),
        })
        marks = "".join("[+]" if v else "[-]" for v in checks.values())
        print(f"  {i+1:2d}. {row['home_team'][:14]:14} vs {row['away_team'][:14]:14} "
              f"-> {favorite[:18]:18} {marks}")
    except Exception as e:
        print(f"  {i+1:2d}. ERROR: {e}")
        continue


# ============================================================
section("5. RESUMEN DE EVALUACION")
# ============================================================
checks_only = [r["checks"] for r in eval_results]
summary = summarize(checks_only)
print(f"\nEvaluacion sobre {len(eval_results)} casos:\n")
print(f"  Menciona el equipo predicho como favorito: {summary['mentions_favorite']:.1%}")
print(f"  Menciona algun factor SHAP relevante:      {summary['mentions_factor']:.1%}")
print(f"  Esta en español:                           {summary['is_spanish']:.1%}")
print(f"  Cumple limite de palabras (<=200):         {summary['within_length']:.1%}")

avg_words = np.mean([r["report_length_words"] for r in eval_results])
print(f"\nLongitud promedio de respuesta: {avg_words:.0f} palabras")

# ============================================================
section("6. GUARDAR RESUMEN PARA EL INFORME")
# ============================================================
summary_data = {
    "n_cases": len(eval_results),
    "metrics": {k: round(v * 100, 1) for k, v in summary.items()},
    "avg_words": round(avg_words, 1),
    "model_groq": client.model,
    "details": eval_results,
}
with open(OUT_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=2, ensure_ascii=False)

# Guardar tabla CSV para el informe LaTeX
tab = pd.DataFrame([
    {"Criterio": "Menciona favorito predicho",
     "Aciertos (%)": f"{summary['mentions_favorite']*100:.1f}"},
    {"Criterio": "Menciona factor SHAP",
     "Aciertos (%)": f"{summary['mentions_factor']*100:.1f}"},
    {"Criterio": "Respuesta en espanol",
     "Aciertos (%)": f"{summary['is_spanish']*100:.1f}"},
    {"Criterio": "Dentro del limite de palabras",
     "Aciertos (%)": f"{summary['within_length']*100:.1f}"},
])
tab.to_csv(ROOT / "docs" / "llm_evaluation_summary.csv", index=False)
print(f"Guardado: docs/llm_evaluation_summary.csv")

print("\n[OK] Integracion LLM completa.")
