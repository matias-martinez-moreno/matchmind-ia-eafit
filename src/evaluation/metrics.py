"""Métricas para clasificación multiclase 3-clases."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def evaluate(y_true, y_pred, y_proba=None, model_name: str = "model") -> dict:
    """Calcula Accuracy, F1-macro y AUC (one-vs-rest) si hay probabilidades.

    Devuelve un dict listo para concatenar en una tabla comparativa.
    """
    out = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }
    if y_proba is not None:
        try:
            out["auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
        except ValueError:
            out["auc_ovr"] = np.nan
    return out


def comparison_table(rows: list[dict]) -> pd.DataFrame:
    """Construye DataFrame ordenado por f1_macro descendente."""
    df = pd.DataFrame(rows)
    return df.sort_values("f1_macro", ascending=False).reset_index(drop=True)


def print_report(y_true, y_pred, target_names=None) -> None:
    """Imprime classification_report + confusion_matrix."""
    if target_names is None:
        target_names = ["away_win", "draw", "home_win"]
    print(classification_report(y_true, y_pred, target_names=target_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
