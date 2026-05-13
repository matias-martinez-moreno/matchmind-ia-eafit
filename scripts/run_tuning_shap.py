"""Tuning de hiperparámetros XGBoost + análisis SHAP global y local.

Ejecutar: python scripts/run_tuning_shap.py
"""
import sys
import time
import pickle
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

from src.models.train import save_model

FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"


def section(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")


# ============================================================
section("1. CARGAR SPLIT")
# ============================================================
with open(ROOT / "data" / "processed" / "train_test_split.pkl", "rb") as f:
    split = pickle.load(f)

X_train, y_train = split["X_train"], split["y_train"]
X_val, y_val = split["X_val"], split["y_val"]
X_test, y_test = split["X_test"], split["y_test"]
meta_test = split["meta_test"]
print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# ============================================================
section("2. GRID SEARCH (manual sobre val)")
# ============================================================
param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200, 500],
    "subsample": [0.8, 1.0],
}
combinations = list(product(*param_grid.values()))
print(f"Probando {len(combinations)} combinaciones...\n")

best = {"f1_macro": -1, "params": None, "model": None}
results = []
t_start = time.time()

for max_depth, lr, n_est, sub in combinations:
    params = dict(
        max_depth=max_depth, learning_rate=lr, n_estimators=n_est,
        subsample=sub, random_state=42, objective="multi:softprob",
        num_class=3, eval_metric="mlogloss", n_jobs=-1,
    )
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_val_pred, average="macro")
    acc = accuracy_score(y_val, y_val_pred)
    results.append({"max_depth": max_depth, "lr": lr, "n_est": n_est,
                    "sub": sub, "val_f1": f1, "val_acc": acc})
    print(f"  d={max_depth} lr={lr} n={n_est} sub={sub} -> F1={f1:.4f} Acc={acc:.4f}")
    if f1 > best["f1_macro"]:
        best = {"f1_macro": f1, "params": params, "model": model}

print(f"\nTotal tuning: {time.time()-t_start:.1f}s")
print(f"\nMejor F1 (val): {best['f1_macro']:.4f}")
print("Mejores params:", {k: v for k, v in best["params"].items()
                          if k in ["max_depth", "learning_rate", "n_estimators", "subsample"]})

# ============================================================
section("3. EVALUACION FINAL EN TEST")
# ============================================================
xgb_best = best["model"]
y_pred = xgb_best.predict(X_test)
y_proba = xgb_best.predict_proba(X_test)
test_f1 = f1_score(y_test, y_pred, average="macro")
test_acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1-macro: {test_f1:.4f}")

# Guardar modelo tuneado
save_model(xgb_best, "xgboost_model.pkl")
print("Modelo final guardado: xgboost_model.pkl")

# ============================================================
section("4. SHAP — explainer global")
# ============================================================
print("Calculando SHAP values (muestra de 500 partidos)...")
explainer = shap.TreeExplainer(xgb_best)
sample_size = min(500, len(X_test))
X_sample = X_test.sample(sample_size, random_state=42)
shap_values = explainer.shap_values(X_sample)

# Para XGBoost multiclase, shap_values puede ser:
# - lista de 3 arrays (versiones viejas)
# - array 3D (versiones nuevas)
if isinstance(shap_values, list):
    print(f"SHAP shape: lista de {len(shap_values)} clases, cada una {shap_values[0].shape}")
    shap_vals_home = shap_values[2]  # clase 2 = home_win
else:
    print(f"SHAP shape: {shap_values.shape}")
    # array 3D: (n_samples, n_features, n_classes)
    shap_vals_home = shap_values[:, :, 2]

# ============================================================
section("5. SHAP — Summary plot (clase home_win)")
# ============================================================
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals_home, X_sample, show=False, plot_size=(10, 6))
plt.title("SHAP summary — clase Local gana")
plt.tight_layout()
out = FIG_DIR / "shap_summary_home.png"
plt.savefig(out)
plt.close()
print(f"Guardada: {out.name}")

# Bar plot global (importance)
plt.figure(figsize=(8, 5))
shap.summary_plot(shap_vals_home, X_sample, plot_type="bar", show=False, plot_size=(8, 5))
plt.title("Feature importance (SHAP global) — clase Local gana")
plt.tight_layout()
out = FIG_DIR / "shap_importance.png"
plt.savefig(out)
plt.close()
print(f"Guardada: {out.name}")

# Top 10 features ranking
mean_abs_shap = np.abs(shap_vals_home).mean(axis=0)
importance = pd.DataFrame({
    "feature": X_sample.columns,
    "mean_abs_shap": mean_abs_shap,
}).sort_values("mean_abs_shap", ascending=False)
print("\nTop 10 features mas importantes (SHAP):")
print(importance.head(10).to_string(index=False))

# ============================================================
section("6. SHAP — 3 partidos individuales")
# ============================================================
# Buscamos partidos icónicos del test set
icons = [
    ("Brazil", "Argentina"),
    ("Germany", "France"),
    ("Spain", "Italy"),
    ("Colombia", "Argentina"),
]

found = []
for h, a in icons:
    mask = (meta_test["home_team"] == h) & (meta_test["away_team"] == a)
    if mask.any():
        idx = meta_test[mask].index[0]
        found.append((h, a, idx))
        if len(found) == 3:
            break

# Si no encontramos 3 partidos icónicos, completamos con los primeros del test
while len(found) < 3:
    i = len(found)
    row = meta_test.iloc[i]
    found.append((row["home_team"], row["away_team"], i))

print(f"Partidos seleccionados:")
for h, a, idx in found:
    row = meta_test.iloc[idx]
    print(f"  - {h} vs {a} ({row['date'].date()}, {row['tournament']})")

# Waterfall plot por cada partido (clase predicha)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (h, a, idx) in zip(axes, found):
    x = X_test.iloc[[idx]]
    proba = xgb_best.predict_proba(x)[0]
    pred = int(np.argmax(proba))
    labels = ["Visitante", "Empate", "Local"]

    shap_one = explainer.shap_values(x)
    if isinstance(shap_one, list):
        sv = shap_one[pred][0]
    else:
        sv = shap_one[0, :, pred]

    # Bar plot horizontal
    contribs = pd.DataFrame({
        "feature": x.columns, "shap": sv, "value": x.values[0],
    }).sort_values("shap", key=abs, ascending=True).tail(8)

    colors = ["#006b3c" if v > 0 else "#9b1b1b" for v in contribs["shap"]]
    ax.barh(contribs["feature"], contribs["shap"], color=colors)
    ax.set_title(f"{h} vs {a}\nPrediccion: {labels[pred]} ({proba[pred]:.0%})",
                 fontsize=10)
    ax.set_xlabel("SHAP value")
    ax.axvline(0, color="black", linewidth=0.5)

plt.suptitle("SHAP local — explicacion de 3 partidos individuales", fontsize=13)
plt.tight_layout()
out = FIG_DIR / "shap_individuales.png"
plt.savefig(out)
plt.close()
print(f"\nGuardada: {out.name}")

# ============================================================
section("7. ANALISIS DE ERRORES")
# ============================================================
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(pd.DataFrame(cm,
                   index=["Real-Away", "Real-Draw", "Real-Home"],
                   columns=["Pred-Away", "Pred-Draw", "Pred-Home"]))

n_draws_real = (y_test == 1).sum()
n_draws_pred = (y_pred == 1).sum()
print(f"\nEmpates reales: {n_draws_real}")
print(f"Empates predichos: {n_draws_pred}  ->  el modelo subpredice empates")

# Tipos de error mas comunes
print("\nTipos de errores mas frecuentes (test):")
err_idx = np.where(y_test != y_pred)[0]
err_pairs = [(int(y_test[i]), int(y_pred[i])) for i in err_idx]
err_counter = pd.Series(err_pairs).value_counts().head(5)
label_map = {0: "Away", 1: "Draw", 2: "Home"}
for (real, pred_), cnt in err_counter.items():
    print(f"  Real={label_map[real]}, Pred={label_map[pred_]}: {cnt} casos ({cnt/len(err_idx):.1%})")

print("\n[OK] Tuning + SHAP completo.")
