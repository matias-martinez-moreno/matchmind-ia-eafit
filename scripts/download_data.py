"""Descarga los datasets de Kaggle usando kagglehub y los coloca en data/raw/.

Ejecutar desde la raiz del proyecto:
    python scripts/download_data.py

Requiere: pip install kagglehub
"""
from pathlib import Path
import shutil

import kagglehub

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_results():
    print("Descargando International Football Results...")
    path = Path(kagglehub.dataset_download(
        "martj42/international-football-results-from-1872-to-2017"
    ))
    print(f"  Descargado en: {path}")
    for csv in path.glob("*.csv"):
        dest = RAW_DIR / csv.name
        shutil.copy(csv, dest)
        print(f"  Copiado: {csv.name} -> {dest.name}")


def download_fifa_ranking():
    print("Descargando FIFA World Rankings...")
    path = Path(kagglehub.dataset_download("cashncarry/fifaworldranking"))
    print(f"  Descargado en: {path}")
    csvs = sorted(path.glob("fifa_ranking-*.csv"))
    for csv in csvs:
        dest = RAW_DIR / csv.name
        shutil.copy(csv, dest)
        print(f"  Copiado: {csv.name} -> {dest.name}")
    # Crear alias con nombre estandar usando el mas reciente
    if csvs:
        latest = csvs[-1]
        alias = RAW_DIR / "fifa_ranking.csv"
        shutil.copy(latest, alias)
        print(f"  Alias creado: {latest.name} -> fifa_ranking.csv")


if __name__ == "__main__":
    download_results()
    download_fifa_ranking()
    print("\n[OK] Datasets disponibles en:", RAW_DIR)
    print("\nArchivos descargados:")
    for f in sorted(RAW_DIR.glob("*.csv")):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}  ({size_mb:.2f} MB)")
