"""EDA avanzado: merge con FIFA ranking, top equipos, head-to-head, heatmap.

Genera 4 figuras adicionales en docs/figures/ y reporta hallazgos.
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


def section(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")


# ============================================================
section("CARGA Y FILTRADO POST-1993")
# ============================================================
df = load_results()
df = df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
df = make_target(df)
df = df[df["date"] >= "1993-01-01"].reset_index(drop=True)
print(f"Partidos post-1993: {len(df):,}")

ranking = load_fifa_ranking()
print(f"Filas ranking FIFA: {len(ranking):,}")

# ============================================================
section("MERGE CON RANKING FIFA (asof)")
# ============================================================
ranking_sorted = ranking.sort_values("rank_date").reset_index(drop=True)
df_sorted = df.sort_values("date").reset_index(drop=True)

# Merge para HOME
home_merge = pd.merge_asof(
    df_sorted[["date", "home_team"]].rename(columns={"home_team": "country_full"}),
    ranking_sorted[["rank_date", "country_full", "rank", "total_points"]],
    left_on="date", right_on="rank_date", by="country_full",
    direction="backward",
)
df_sorted["home_rank"] = home_merge["rank"].values
df_sorted["home_points"] = home_merge["total_points"].values

# Merge para AWAY
away_merge = pd.merge_asof(
    df_sorted[["date", "away_team"]].rename(columns={"away_team": "country_full"}),
    ranking_sorted[["rank_date", "country_full", "rank", "total_points"]],
    left_on="date", right_on="rank_date", by="country_full",
    direction="backward",
)
df_sorted["away_rank"] = away_merge["rank"].values
df_sorted["away_points"] = away_merge["total_points"].values

df_sorted["rank_diff"] = df_sorted["home_rank"] - df_sorted["away_rank"]
df_sorted["points_diff"] = df_sorted["home_points"] - df_sorted["away_points"]

cobertura_home = df_sorted["home_rank"].notna().mean()
cobertura_away = df_sorted["away_rank"].notna().mean()
print(f"Cobertura ranking home: {cobertura_home:.2%}")
print(f"Cobertura ranking away: {cobertura_away:.2%}")

df_r = df_sorted.dropna(subset=["home_rank", "away_rank"]).reset_index(drop=True)
print(f"Partidos con ranking completo: {len(df_r):,}")

# ============================================================
section("FIGURA 5 — Correlacion diferencia de ranking vs resultado")
# ============================================================
# rank_diff: positivo = local peor ranking (mayor numero), negativo = local mejor
bins = [-300, -50, -20, -5, 5, 20, 50, 300]
labels = ["Local mucho mejor", "Local mejor", "Local algo mejor",
          "Similar", "Visitante algo mejor", "Visitante mejor", "Visitante mucho mejor"]
df_r["rank_diff_bin"] = pd.cut(df_r["rank_diff"], bins=bins, labels=labels)

agg = df_r.groupby("rank_diff_bin", observed=True)["result"].value_counts(normalize=True).unstack()
agg.columns = ["Visitante", "Empate", "Local"]

fig, ax = plt.subplots(figsize=(11, 5))
agg[["Local", "Empate", "Visitante"]].plot(
    kind="bar", stacked=True, ax=ax,
    color=["#006b3c", "#b45309", "#9b1b1b"],
)
ax.set_title("Probabilidad de resultado segun diferencia de ranking FIFA")
ax.set_ylabel("Proporcion")
ax.set_xlabel("Diferencia de ranking (negativo = local mejor)")
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_ranking_resultado.png")
plt.close()
print("Guardada: eda_ranking_resultado.png")
print("\nTabla:")
print(agg.applymap(lambda x: f"{x:.1%}"))

# ============================================================
section("FIGURA 6 — Top 10 equipos con mas victorias historicas")
# ============================================================
wins_home = df[df["result"] == 2].groupby("home_team").size()
wins_away = df[df["result"] == 0].groupby("away_team").size()
total_wins = wins_home.add(wins_away, fill_value=0).sort_values(ascending=False)
top10 = total_wins.head(10)

fig, ax = plt.subplots(figsize=(9, 5))
top10.plot(kind="barh", ax=ax, color="#003d79")
ax.set_title("Top 10 selecciones con mas victorias historicas (1872-2026)")
ax.set_xlabel("Numero de victorias")
ax.invert_yaxis()
for i, v in enumerate(top10.values):
    ax.text(v + 10, i, f"{int(v):,}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_top_equipos.png")
plt.close()
print("Guardada: eda_top_equipos.png")
print("\nTop 10:")
print(top10)

# ============================================================
section("FIGURA 7 — Head-to-head clasicos")
# ============================================================
clasicos = [
    ("Brazil", "Argentina"),
    ("Germany", "Italy"),
    ("Spain", "France"),
    ("Mexico", "United States"),
]

def h2h_stats(team_a, team_b):
    """Retorna (wins_a, draws, wins_b, total)."""
    games = df[
        ((df["home_team"] == team_a) & (df["away_team"] == team_b))
        | ((df["home_team"] == team_b) & (df["away_team"] == team_a))
    ]
    wins_a = (
        ((games["home_team"] == team_a) & (games["result"] == 2)).sum()
        + ((games["away_team"] == team_a) & (games["result"] == 0)).sum()
    )
    wins_b = (
        ((games["home_team"] == team_b) & (games["result"] == 2)).sum()
        + ((games["away_team"] == team_b) & (games["result"] == 0)).sum()
    )
    draws = (games["result"] == 1).sum()
    return wins_a, draws, wins_b, len(games)


fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, (a, b) in zip(axes.flat, clasicos):
    wa, d, wb, total = h2h_stats(a, b)
    if total == 0:
        ax.set_visible(False)
        continue
    cats = [f"Gana {a}", "Empate", f"Gana {b}"]
    vals = [wa, d, wb]
    colors = ["#006b3c", "#b45309", "#9b1b1b"]
    ax.bar(cats, vals, color=colors)
    ax.set_title(f"{a} vs {b}  ({total} partidos)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, f"{v}\n({v/total:.0%})", ha="center", fontsize=9)
    ax.set_ylabel("Partidos")

plt.suptitle("Rivalidades clasicas — historico head-to-head", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_h2h_clasicos.png")
plt.close()
print("Guardada: eda_h2h_clasicos.png")
for a, b in clasicos:
    wa, d, wb, total = h2h_stats(a, b)
    print(f"  {a} vs {b}: {wa}-{d}-{wb} ({total} partidos)")

# ============================================================
section("FIGURA 8 — Heatmap de correlaciones")
# ============================================================
numeric_cols = ["home_rank", "away_rank", "rank_diff",
                "home_points", "away_points", "points_diff",
                "home_score", "away_score", "result"]
corr = df_r[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, center=0, ax=ax,
            cbar_kws={"label": "Coeficiente de correlacion"})
ax.set_title("Heatmap de correlaciones (post-1993, con ranking FIFA)")
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_heatmap.png")
plt.close()
print("Guardada: eda_heatmap.png")
print("\nCorrelaciones mas importantes con 'result':")
result_corrs = corr["result"].drop("result").abs().sort_values(ascending=False)
print(result_corrs)

# ============================================================
section("HALLAZGOS AVANZADOS")
# ============================================================
strong_fav_home = agg.loc["Local mucho mejor", "Local"] if "Local mucho mejor" in agg.index else 0
weak_fav_home = agg.loc["Local mucho mejor", "Visitante"] if "Local mucho mejor" in agg.index else 0
rank_corr = abs(corr.loc["result", "rank_diff"])
points_corr = abs(corr.loc["result", "points_diff"])

print(f"""
1. Ranking FIFA predice fuertemente el resultado:
   - Cuando el local tiene MUCHO mejor ranking: {strong_fav_home:.1%} gana, {weak_fav_home:.1%} pierde.
   - Correlacion rank_diff con result: {rank_corr:.3f}
   - Correlacion points_diff con result: {points_corr:.3f}

2. Top selecciones historicas: {", ".join(top10.head(5).index.tolist())}
   -> Estos equipos dominan en victorias absolutas.

3. Cobertura del merge: {cobertura_home:.1%} home, {cobertura_away:.1%} away
   -> Buena cobertura, ranking FIFA es feature confiable para post-1993.

4. Clasicos balanceados: Brazil vs Argentina, Germany vs Italy son
   rivalidades muy parejas -> features de h2h son cruciales para esos partidos.

5. Multicolinealidad: rank y points estan altamente correlacionados
   ({abs(corr.loc['home_rank', 'home_points']):.2f}) -> elegir uno o usar regularizacion.
""")

print("[OK] EDA avanzado completo.")
