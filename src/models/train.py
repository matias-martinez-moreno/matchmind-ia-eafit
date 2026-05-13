"""Entrenamiento y persistencia del modelo XGBoost.

Nota: el import de xgboost esta dentro de fit_xgboost() para que los notebooks
de preprocesamiento (02) no requieran tener xgboost instalado.
"""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "checkpoints"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def get_feature_columns() -> list[str]:
    """Lista de columnas que el modelo usa como features."""
    return [
        "home_recent_wins",
        "away_recent_wins",
        "home_goals_avg_10",
        "home_goals_conceded_avg_10",
        "away_goals_avg_10",
        "away_goals_conceded_avg_10",
        "h2h_home_wins",
        "h2h_draws",
        "h2h_away_wins",
        "home_team_rank",
        "away_team_rank",
        "rank_diff",
        "points_diff",
        "neutral",
    ]


def temporal_split(df: pd.DataFrame, train_end: str = "2015-12-31",
                   val_end: str = "2019-12-31") -> tuple:
    """Split temporal sin data leakage.

    Train: hasta train_end, Val: hasta val_end, Test: posterior.
    """
    df = df.sort_values("date")
    train = df[df["date"] <= train_end]
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test = df[df["date"] > val_end]
    return train, val, test


def fit_baseline(X_train, y_train) -> DummyClassifier:
    """Modelo dummy: predice la clase mayoritaria. Sirve de piso absoluto."""
    model = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def fit_logreg(X_train, y_train) -> tuple[LogisticRegression, StandardScaler]:
    """Regresión logística con scaler ajustado solo en train."""
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_s, y_train)
    return model, scaler


def fit_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Random Forest con configuración razonable."""
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def fit_xgboost(X_train, y_train, **kwargs):
    """XGBoost para clasificacion multiclase (3 clases)."""
    from xgboost import XGBClassifier  # import lazy
    params = dict(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    params.update(kwargs)
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def save_model(model, name: str = "xgboost_model.pkl") -> Path:
    """Persiste el modelo entrenado."""
    path = MODELS_DIR / name
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(name: str = "xgboost_model.pkl"):
    """Carga el modelo entrenado."""
    path = MODELS_DIR / name
    with open(path, "rb") as f:
        return pickle.load(f)
