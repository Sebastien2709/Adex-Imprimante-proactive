"""
scripts/gen_history_parquet.py

Convertit kpax_history_light.csv en parquet pour Supabase.
Le parquet permet une lecture filtrée par serial (pyarrow pushdown)
au lieu de scanner 2.7M lignes de CSV à chaque requête graphe.
"""
from pathlib import Path
import pandas as pd

CSV_PATH     = Path("data/processed/kpax_history_light.csv")
PARQUET_PATH = Path("data/processed/kpax_history_light.parquet")


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"[gen_history_parquet] Fichier introuvable : {CSV_PATH}")

    print(f"[gen_history_parquet] Lecture {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, low_memory=False, dtype={"serial_display": str})
    df["serial_display"] = df["serial_display"].astype(str).str.strip()

    df.to_parquet(PARQUET_PATH, index=False)
    print(f"[gen_history_parquet] ✅ {PARQUET_PATH} ({len(df)} lignes)")


if __name__ == "__main__":
    main()