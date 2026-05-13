"""Feature engineering vectorizado para predicción de partidos.

Estrategia: convertir el dataframe wide (un partido = 1 fila con home y away)
a formato long (un partido = 2 filas, una por equipo). Sobre el long calculamos
rolling/cumsum por equipo con shift(1) para evitar data leakage. Luego
volvemos a wide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte el dataframe wide a formato long (1 partido = 2 filas).

    Para cada partido genera dos filas: una con perspectiva del home y otra
    con perspectiva del away. Esto permite calcular estadísticas por equipo
    independientemente de si jugó como local o visitante.
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["match_id"] = df.index

    home = pd.DataFrame({
        "match_id": df["match_id"],
        "date": df["date"],
        "team": df["home_team"],
        "opponent": df["away_team"],
        "is_home": 1,
        "goals_scored": df["home_score"],
        "goals_conceded": df["away_score"],
        "result": df["result"],  # 0=away, 1=draw, 2=home
    })
    home["won"] = (home["result"] == 2).astype(int)
    home["drew"] = (home["result"] == 1).astype(int)

    away = pd.DataFrame({
        "match_id": df["match_id"],
        "date": df["date"],
        "team": df["away_team"],
        "opponent": df["home_team"],
        "is_home": 0,
        "goals_scored": df["away_score"],
        "goals_conceded": df["home_score"],
        "result": df["result"],
    })
    away["won"] = (away["result"] == 0).astype(int)
    away["drew"] = (away["result"] == 1).astype(int)

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)
    return long


def _add_team_features(long: pd.DataFrame, n_recent: int = 5,
                       window_goals: int = 10) -> pd.DataFrame:
    """Calcula racha y promedios de goles por equipo con rolling + shift(1)."""
    g = long.groupby("team", group_keys=False)

    # Victorias en los últimos `n_recent` partidos (sin incluir el actual)
    long["recent_wins"] = g["won"].apply(
        lambda s: s.shift(1).rolling(n_recent, min_periods=1).sum()
    ).reset_index(level=0, drop=True)

    # Promedio de goles anotados y recibidos en ventana
    long["goals_avg"] = g["goals_scored"].apply(
        lambda s: s.shift(1).rolling(window_goals, min_periods=1).mean()
    ).reset_index(level=0, drop=True)

    long["goals_conceded_avg"] = g["goals_conceded"].apply(
        lambda s: s.shift(1).rolling(window_goals, min_periods=1).mean()
    ).reset_index(level=0, drop=True)

    long = long.fillna({"recent_wins": 0, "goals_avg": 0, "goals_conceded_avg": 0})
    return long


def _long_to_wide(long: pd.DataFrame, df_original: pd.DataFrame) -> pd.DataFrame:
    """Vuelve a wide: junta las features del home y del away en una sola fila."""
    home_long = long[long["is_home"] == 1][[
        "match_id", "recent_wins", "goals_avg", "goals_conceded_avg"
    ]].rename(columns={
        "recent_wins": "home_recent_wins",
        "goals_avg": "home_goals_avg_10",
        "goals_conceded_avg": "home_goals_conceded_avg_10",
    })

    away_long = long[long["is_home"] == 0][[
        "match_id", "recent_wins", "goals_avg", "goals_conceded_avg"
    ]].rename(columns={
        "recent_wins": "away_recent_wins",
        "goals_avg": "away_goals_avg_10",
        "goals_conceded_avg": "away_goals_conceded_avg_10",
    })

    out = df_original.merge(home_long, on="match_id", how="left")
    out = out.merge(away_long, on="match_id", how="left")
    return out


def compute_h2h_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Head-to-head acumulado entre pares de equipos, sin data leakage.

    Para cada partido entre A y B, cuenta cuántos partidos previos hubo y
    quién ganó. Usa un identificador de par no ordenado.
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["match_id"] = df.index

    # Identificador de par: ordenado alfabéticamente
    df["pair"] = df.apply(
        lambda r: tuple(sorted([r["home_team"], r["away_team"]])), axis=1
    )

    # Indicadores de quién ganó cada partido
    df["_home_won"] = (df["result"] == 2).astype(int)
    df["_away_won"] = (df["result"] == 0).astype(int)
    df["_drew"] = (df["result"] == 1).astype(int)

    # Cumsum por par, con shift para que el partido actual NO se cuente
    g = df.groupby("pair", group_keys=False)
    df["_cum_home_won"] = g["_home_won"].apply(lambda s: s.shift(1).cumsum().fillna(0))
    df["_cum_away_won"] = g["_away_won"].apply(lambda s: s.shift(1).cumsum().fillna(0))
    df["_cum_drew"] = g["_drew"].apply(lambda s: s.shift(1).cumsum().fillna(0))

    # OJO: el cumsum cuenta "home won" desde la perspectiva del que fue local
    # en cada partido pasado. Pero el "home_team" del partido actual puede ser
    # cualquiera de los dos. Necesitamos recalcular en perspectiva del local actual.
    # Solución: calcular total de victorias por equipo dentro del par.

    # Reconvertir: para cada partido, # victorias del home_team actual en h2h pasados
    g = df.groupby("pair", group_keys=False)

    def victories_for_team(group):
        """Para cada partido del par, cuenta victorias previas de cada equipo."""
        teams = list(set(group["home_team"]) | set(group["away_team"]))
        # Si solo hay 2 equipos en el par (lo normal):
        if len(teams) > 2:
            # Caso raro por nombres con cambio histórico; tratamos como 2
            teams = teams[:2]

        # acumulado de victorias de cada equipo del par
        wins_a = []
        wins_b = []
        draws = []
        cum = {t: 0 for t in teams}
        cum_drew = 0
        for _, row in group.iterrows():
            h, a = row["home_team"], row["away_team"]
            wins_a.append(cum.get(h, 0))
            wins_b.append(cum.get(a, 0))
            draws.append(cum_drew)
            if row["result"] == 2:
                cum[h] = cum.get(h, 0) + 1
            elif row["result"] == 0:
                cum[a] = cum.get(a, 0) + 1
            else:
                cum_drew += 1
        group = group.copy()
        group["h2h_home_wins"] = wins_a
        group["h2h_away_wins"] = wins_b
        group["h2h_draws"] = draws
        return group

    df = g.apply(victories_for_team).reset_index(drop=True)
    df = df.drop(columns=["_home_won", "_away_won", "_drew",
                          "_cum_home_won", "_cum_away_won", "_cum_drew", "pair"])
    return df


def merge_fifa_rank(df: pd.DataFrame, ranking_df: pd.DataFrame) -> pd.DataFrame:
    """Hace merge del ranking FIFA con cada partido por fecha más cercana anterior."""
    df = df.sort_values("date").reset_index(drop=True)
    ranking_df = ranking_df.sort_values("rank_date").reset_index(drop=True)

    home = pd.merge_asof(
        df[["date", "home_team"]].rename(columns={"home_team": "country_full"}),
        ranking_df[["rank_date", "country_full", "rank", "total_points"]],
        left_on="date", right_on="rank_date", by="country_full", direction="backward",
    )
    df["home_team_rank"] = home["rank"].values
    df["home_team_points"] = home["total_points"].values

    away = pd.merge_asof(
        df[["date", "away_team"]].rename(columns={"away_team": "country_full"}),
        ranking_df[["rank_date", "country_full", "rank", "total_points"]],
        left_on="date", right_on="rank_date", by="country_full", direction="backward",
    )
    df["away_team_rank"] = away["rank"].values
    df["away_team_points"] = away["total_points"].values

    df["rank_diff"] = df["home_team_rank"] - df["away_team_rank"]
    df["points_diff"] = df["home_team_points"] - df["away_team_points"]
    return df


def build_features(df: pd.DataFrame, ranking_df: pd.DataFrame | None = None,
                   n_recent: int = 5, window_goals: int = 10) -> pd.DataFrame:
    """Pipeline completo de feature engineering (vectorizado).

    df debe tener columna `result` (0=away, 1=draw, 2=home).
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["match_id"] = df.index

    # Features por equipo (racha + promedios de goles)
    long = _to_long(df)
    long = _add_team_features(long, n_recent=n_recent, window_goals=window_goals)
    df = _long_to_wide(long, df)

    # Head-to-head
    df = compute_h2h_vectorized(df)

    # Ranking FIFA
    if ranking_df is not None:
        df = merge_fifa_rank(df, ranking_df)

    # neutral como int
    if "neutral" in df.columns:
        df["neutral"] = df["neutral"].astype(int)

    df = df.drop(columns=["match_id"], errors="ignore")
    return df
