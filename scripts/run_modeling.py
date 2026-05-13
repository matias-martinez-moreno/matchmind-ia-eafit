"""Entrena los 4 modelos, genera tabla comparativa, confusion matrix y guarda
el mejor modelo (XGBoost) para uso posterior.

Ejecutar: python scripts/run_modeling.py
"""
import sys
import time
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.models.train import (
    fit_baseline, fit_logreg, fit_random_forest, fit_xgboost,
    save_model, get_feature_columns,
)
from src.evaluation.metrics import evaluate, comparison_table, print_report

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
X_val,   y_val   = split["X_val"],   split["y_val"]
X_test,  y_test  = split["X_test"],  split["y_test"]
print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# ============================================================
section("2. BASELINE — Dummy (clase mayoritaria)")
# ============================================================
t0 = time.time()
dummy = fit_baseline(X_train, y_train)
y_pred_d = dummy.predict(X_test)
y_proba_d = dummy.predict_proba(X_test)
row_dummy = evaluate(y_test, y_pred_d, y_proba_d, "Dummy")
print(f"Tiempo: {time.time()-t0:.2f}s")
print(row_dummy)

# ============================================================
section("3. LOGISTIC REGRESSION")
# ============================================================
t0 = time.time()
logreg, scaler = fit_logreg(X_train, y_train)
X_test_s = scaler.transform(X_test)
y_pred_lr = logreg.predict(X_test_s)
y_proba_lr = logreg.predict_proba(X_test_s)
row_lr = evaluate(y_test, y_pred_lr, y_proba_lr, "LogReg")
print(f"Tiempo: {time.time()-t0:.2f}s")
print(row_lr)

# ============================================================
section("4. RANDOM FOREST")
# ============================================================
t0 = time.time()
rf = fit_random_forest(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)
row_rf = evaluate(y_test, y_pred_rf, y_proba_rf, "RandomForest")
print(f"Tiempo: {time.time()-t0:.2f}s")
print(row_rf)

# ============================================================
section("5. XGBOOST")
# ============================================================
t0 = time.time()
xgb = fit_xgboost(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_proba_xgb = xgb.predict_proba(X_test)
row_xgb = evaluate(y_test, y_pred_xgb, y_proba_xgb, "XGBoost")
print(f"Tiempo: {time.time()-t0:.2f}s")
print(row_xgb)

# ============================================================
section("6. TABLA COMPARATIVA")
# ============================================================
table = comparison_table([row_dummy, row_lr, row_rf, row_xgb])
print(table.to_string(index=False))

# Guardar tabla en CSV para el informe
out_csv = ROOT / "docs" / "comparativa_modelos.csv"
table.to_csv(out_csv, index=False)
print(f"\nTabla CSV: {out_csv}")

# ============================================================
section("7. REPORTE DETALLADO XGBOOST")
# ============================================================
print_report(y_test, y_pred_xgb)

# ============================================================
section("8. CONFUSION MATRIX")
# ============================================================
cm = confusion_matrix(y_test, y_pred_xgb)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Away", "Draw", "Home"],
            yticklabels=["Away", "Draw", "Home"], ax=ax)
ax.set_xlabel("Prediccion")
ax.set_ylabel("Real")
ax.set_title("Confusion Matrix — XGBoost (test set)")
plt.tight_layout()
out = FIG_DIR / "confusion_matrix.png"
plt.savefig(out)
plt.close()
print(f"Guardada: {out.name}")

# ============================================================
section("9. GRAFICO COMPARATIVO DE MODELOS")
# ============================================================
fig, ax = plt.subplots(figsize=(9, 4.5))
x_pos = np.arange(len(table))
width = 0.25
ax.bar(x_pos - width, table["accuracy"], width, label="Accuracy", color="#003d79")
ax.bar(x_pos, table["f1_macro"], width, label="F1-macro", color="#006b3c")
if "auc_ovr" in table.columns:
    ax.bar(x_pos + width, table["auc_ovr"], width, label="AUC (OvR)", color="#b45309")
ax.set_xticks(x_pos)
ax.set_xticklabels(table["model"])
ax.set_ylim(0, 1)
ax.set_title("Comparacion de modelos en test set")
ax.legend()
ax.grid(axis="y", alpha=0.3)
for i, (acc, f1) in enumerate(zip(table["accuracy"], table["f1_macro"])):
    ax.text(i - width, acc + 0.01, f"{acc:.2f}", ha="center", fontsize=8)
    ax.text(i, f1 + 0.01, f"{f1:.2f}", ha="center", fontsize=8)
plt.tight_layout()
out = FIG_DIR / "comparativa_modelos.png"
plt.savefig(out)
plt.close()
print(f"Guardada: {out.name}")

# ============================================================
section("10. GUARDAR MODELO FINAL Y SCALER")
# ============================================================
save_model(xgb, "xgboost_model.pkl")
print("Guardado: models/checkpoints/xgboost_model.pkl")

with open(ROOT / "models" / "checkpoints" / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Guardado: models/checkpoints/scaler.pkl")

print("\n[OK] Modelado completo.")
