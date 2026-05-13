"""Ejecuta el pipeline de feature engineering completo y guarda los datasets
procesados.

Ejecutar: python scripts/run_preprocessing.py
"""
import sys
import time
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data.load import load_results, load_fifa_ranking, make_target
from src.data.features import build_features
from src.models.train import temporal_split, get_feature_columns

PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def section(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")


# ============================================================
section("1. CARGA Y LIMPIEZA")
# ============================================================
df = load_results()
print(f"Partidos crudos: {len(df):,}")

df = df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
df = make_target(df)
df = df[df["date"] >= "1993-01-01"].reset_index(drop=True)
print(f"Post-1993 sin nulos: {len(df):,}")

ranking = load_fifa_ranking()
print(f"Filas ranking FIFA: {len(ranking):,}")

# ============================================================
section("2. FEATURE ENGINEERING (vectorizado)")
# ============================================================
t0 = time.time()
features = build_features(df, ranking_df=ranking, n_recent=5, window_goals=10)
dt = time.time() - t0
print(f"Tiempo: {dt:.1f}s para {len(features):,} partidos")
print(f"Columnas: {list(features.columns)}")

# ============================================================
section("3. IMPUTACION DE NULOS DEL RANKING")
# ============================================================
print("Nulos antes:")
print(features[["home_team_rank", "away_team_rank",
                "home_team_points", "away_team_points"]].isnull().sum())

med_rank = features["home_team_rank"].median()
features["home_team_rank"] = features["home_team_rank"].fillna(med_rank)
features["away_team_rank"] = features["away_team_rank"].fillna(med_rank)
features["home_team_points"] = features["home_team_points"].fillna(0)
features["away_team_points"] = features["away_team_points"].fillna(0)
features["rank_diff"] = features["home_team_rank"] - features["away_team_rank"]
features["points_diff"] = features["home_team_points"] - features["away_team_points"]

print(f"Mediana usada para imputar rank: {med_rank}")

# ============================================================
section("4. GUARDAR DATASET PROCESADO")
# ============================================================
out_csv = PROCESSED / "features.csv"
features.to_csv(out_csv, index=False)
print(f"Guardado: {out_csv}  ({out_csv.stat().st_size / 1024 / 1024:.1f} MB)")

# ============================================================
section("5. SPLIT TEMPORAL")
# ============================================================
train, val, test = temporal_split(features,
                                  train_end="2015-12-31",
                                  val_end="2019-12-31")
print(f"Train: {len(train):,} ({train['date'].min().date()} -> {train['date'].max().date()})")
print(f"Val:   {len(val):,}   ({val['date'].min().date()} -> {val['date'].max().date()})")
print(f"Test:  {len(test):,}  ({test['date'].min().date()} -> {test['date'].max().date()})")

feature_cols = get_feature_columns()
print(f"\nFeatures usadas ({len(feature_cols)}): {feature_cols}")

split = {
    "X_train": train[feature_cols].astype(float),
    "y_train": train["result"].values,
    "X_val":   val[feature_cols].astype(float),
    "y_val":   val["result"].values,
    "X_test":  test[feature_cols].astype(float),
    "y_test":  test["result"].values,
    "meta_train": train[["date", "home_team", "away_team", "tournament"]].reset_index(drop=True),
    "meta_val":   val[["date", "home_team", "away_team", "tournament"]].reset_index(drop=True),
    "meta_test":  test[["date", "home_team", "away_team", "tournament"]].reset_index(drop=True),
}

with open(PROCESSED / "train_test_split.pkl", "wb") as f:
    pickle.dump(split, f)
print(f"\nSplit guardado en: {PROCESSED / 'train_test_split.pkl'}")

# ============================================================
section("6. SANITY CHECK")
# ============================================================
print("Distribucion de clases en train:")
print(pd.Series(split["y_train"]).value_counts(normalize=True).sort_index().rename({0: "away", 1: "draw", 2: "home"}))
print("\nDistribucion de clases en test:")
print(pd.Series(split["y_test"]).value_counts(normalize=True).sort_index().rename({0: "away", 1: "draw", 2: "home"}))

print("\nPrimeras filas de X_train:")
print(split["X_train"].head().to_string())
print("\nNulos en splits:")
for name in ["X_train", "X_val", "X_test"]:
    print(f"  {name}: {split[name].isnull().sum().sum()} nulos")
