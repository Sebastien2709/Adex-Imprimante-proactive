import os, pandas as pd
from src.data.utils import parse_eu_datetime, coerce_numeric, ensure_bounds

INTERIM = "data/interim"
PROC = "data/processed"
os.makedirs(PROC, exist_ok=True)

IN_FILE = os.path.join(INTERIM, "kpax_consumables.csv")
OUT_FILE = os.path.join(PROC, "kpax_consumables.parquet")

PCT_COLS = ["black_pct","cyan_pct","magenta_pct","yellow_pct"]

def clean_consumables(df: pd.DataFrame) -> pd.DataFrame:
    # dates
    for col in ["date_import", "date_update"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_eu_datetime)
    # pct numeriques
    df = coerce_numeric(df, PCT_COLS)
    for c in PCT_COLS:
        df = ensure_bounds(df, c, 0, 100)
    # flags
    if all(c in df.columns for c in PCT_COLS):
        df["all_zero_flag"] = (df[PCT_COLS].fillna(0).eq(0).all(axis=1)).astype(int)
    # base qualité
    if "serial" in df.columns:
        df["serial_missing"] = df["serial"].isna() | (df["serial"].astype(str).str.len()==0)
    return df

def main():
    if not os.path.exists(IN_FILE):
        raise SystemExit(f"Missing {IN_FILE} — run ingest.py first.")
    df = pd.read_csv(IN_FILE)
    df = clean_consumables(df)
    df.to_parquet(OUT_FILE, index=False)
    print(f"Saved {OUT_FILE} ({len(df)} rows)")

if __name__ == "__main__":
    main()
