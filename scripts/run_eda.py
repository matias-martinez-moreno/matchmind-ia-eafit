"""Ejecuta el EDA completo, genera figuras PNG en docs/figures/ y imprime
hallazgos clave para documentar en el informe.

Ejecutar: python scripts/run_eda.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.load import load_results, load_fifa_ranking, make_target

FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"


def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


# ============================================================
section("1. CARGA DE DATOS")
# ============================================================
df = load_results()
ranking = load_fifa_ranking()
print(f"Partidos: {len(df):,}")
print(f"Rango temporal: {df['date'].min().date()} -> {df['date'].max().date()}")
print(f"Filas ranking FIFA: {len(ranking):,}")
print(f"Rango ranking FIFA: {ranking['rank_date'].min().date()} -> {ranking['rank_date'].max().date()}")

# ============================================================
section("2. INSPECCION BASICA")
# ============================================================
print("\n--- df.info() ---")
df.info()
print("\n--- Valores nulos ---")
print(df.isnull().sum())
print("\n--- df.describe() ---")
print(df.describe())

# ============================================================
section("3. VARIABLE OBJETIVO")
# ============================================================
df = make_target(df)
counts = df["result"].value_counts(normalize=True).sort_index()
counts.index = ["away_win", "draw", "home_win"]
print("Distribucion de clases:")
print(counts.apply(lambda x: f"{x:.2%}"))

# ============================================================
section("4. FIGURA 1 — Distribucion del target")
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4))
labels = ["Visitante gana", "Empate", "Local gana"]
counts_abs = df["result"].value_counts().sort_index()
colors = ["#9b1b1b", "#b45309", "#006b3c"]
ax.bar(labels, counts_abs.values, color=colors)
ax.set_title("Distribucion del resultado del partido (1872-2024)")
ax.set_ylabel("Numero de partidos")
for i, v in enumerate(counts_abs.values):
    ax.text(i, v + 200, f"{v:,}\n({v/len(df):.1%})", ha="center", fontsize=10)
plt.tight_layout()
out = FIG_DIR / "eda_distribucion.png"
plt.savefig(out)
plt.close()
print(f"Guardada: {out.name}")

# ============================================================
section("5. FIGURA 2 — Goles por tipo de torneo")
# ============================================================
df["total_goals"] = df["home_score"] + df["away_score"]
top_tournaments = df["tournament"].value_counts().head(8).index
subset = df[df["tournament"].isin(top_tournaments)]
fig, ax = plt.subplots(figsize=(11, 5))
sns.boxplot(data=subset, x="tournament", y="total_goals", ax=ax,
            order=top_tournaments)
ax.set_title("Distribucion de goles totales por tipo de torneo (top 8)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.set_xlabel("")
ax.set_ylabel("Goles totales por partido")
plt.tight_layout()
out = FIG_DIR / "eda_goles_torneo.png"
plt.savefig(out)
plt.close()
print(f"Guardada: {out.name}")
print("Top tournaments:")
for t in top_tournaments:
    print(f"  - {t}: {(df['tournament']==t).sum():,} partidos")

# ============================================================
section("6. FIGURA 3 — Evolucion temporal por decada")
# ============================================================
df["decade"] = (df["date"].dt.year // 10) * 10
by_decade = df.groupby("decade").size()
fig, ax = plt.subplots(figsize=(10, 4))
by_decade.plot(kind="bar", ax=ax, color="#003d79")
ax.set_title("Partidos internacionales por decada")
ax.set_xlabel("Decada")
ax.set_ylabel("Numero de partidos")
plt.xticks(rotation=0)
plt.tight_layout()
out = FIG_DIR / "eda_temporal.png"
plt.savefig(out)
plt.close()
print(f"Guardada: {out.name}")
print("Partidos por decada:")
print(by_decade)

# ============================================================
section("7. FIGURA 4 — Ventaja de localia")
# ============================================================
result_by_neutral = df.groupby("neutral")["result"].value_counts(normalize=True).unstack()
result_by_neutral.columns = ["Visitante gana", "Empate", "Local gana"]
fig, ax = plt.subplots(figsize=(7, 4))
result_by_neutral.plot(kind="bar", stacked=True, ax=ax,
                       color=["#9b1b1b", "#b45309", "#006b3c"])
ax.set_title("Distribucion del resultado segun campo")
ax.set_xticklabels(["Campo del local", "Campo neutral"], rotation=0)
ax.set_ylabel("Proporcion")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
out = FIG_DIR / "eda_localia.png"
plt.savefig(out)
plt.close()
print(f"Guardada: {out.name}")
print("\nVentaja de localia:")
print(result_by_neutral.applymap(lambda x: f"{x:.2%}"))

# ============================================================
section("8. ANALISIS DE OUTLIERS EN GOLES")
# ============================================================
print(f"Maximo goles totales: {df['total_goals'].max()}")
extreme = df[df["total_goals"] > 10]
print(f"Partidos con >10 goles: {len(extreme)} ({len(extreme)/len(df):.2%})")
print("\nTop 5 goleadas historicas:")
print(extreme.nlargest(5, "total_goals")[
    ["date", "home_team", "away_team", "home_score", "away_score", "tournament"]
].to_string(index=False))

# ============================================================
section("9. HALLAZGOS CLAVE")
# ============================================================
home_pct = (df["result"] == 2).mean()
draw_pct = (df["result"] == 1).mean()
away_pct = (df["result"] == 0).mean()
neutral_home_pct = df[df["neutral"]==False]["result"].eq(2).mean()
nonneutral_home_pct = df[df["neutral"]==True]["result"].eq(2).mean()

print(f"""
1. Distribucion del target: {home_pct:.1%} victorias locales, {draw_pct:.1%} empates, {away_pct:.1%} victorias visitantes.
   -> Imbalance moderado (local mas frecuente). Justifica F1-macro sobre Accuracy.

2. Ventaja de localia: {neutral_home_pct:.1%} victorias locales en campo propio vs
   {nonneutral_home_pct:.1%} en campo neutral. -> diferencia de {(neutral_home_pct-nonneutral_home_pct)*100:.1f} puntos.
   -> 'neutral' es feature importante.

3. Sesgo historico: la mayoria de partidos pre-1950 son europeos. Partidos por decada
   crecen exponencialmente: {by_decade.iloc[0]} (1870s) -> {by_decade.iloc[-1]} (2020s).
   -> Justifica filtrar a partidos post-1993 (cuando empieza ranking FIFA).

4. Outliers: {len(extreme)} partidos con >10 goles ({len(extreme)/len(df):.2%}).
   -> No los eliminamos: son partidos reales, no errores.

5. Tipo de torneo: hay {df['tournament'].nunique()} torneos distintos.
   El torneo afecta la distribucion de goles (boxplot) -> feature relevante.
""")

print("\n[OK] EDA completo. Figuras en docs/figures/")
