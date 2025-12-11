import os, pandas as pd
from src.data.utils import parse_eu_datetime, coerce_numeric

INTERIM = "data/interim"
PROC = "data/processed"
os.makedirs(PROC, exist_ok=True)

IN_FILE = os.path.join(INTERIM, "meters.csv")
OUT_FILE = os.path.join(PROC, "meters.parquet")

COUNTER_COLS = [
    "start_a4_bw","end_a4_bw","start_a4_color","end_a4_color",
    "start_scan","end_scan","start_a3_bw","end_a3_bw","start_a3_color","end_a3_color"
]

def clean_meters(df: pd.DataFrame) -> pd.DataFrame:
    # parse dates (keep as date)
    for col in ["start_date","end_date","read_date"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_eu_datetime).dt.date
    # numeric counters
    df = coerce_numeric(df, COUNTER_COLS)
    # deltas
    if set(["start_a4_bw","end_a4_bw"]).issubset(df.columns):
        df["a4_bw_delta"] = df["end_a4_bw"] - df["start_a4_bw"]
    if set(["start_a4_color","end_a4_color"]).issubset(df.columns):
        df["a4_color_delta"] = df["end_a4_color"] - df["start_a4_color"]
    if set(["start_scan","end_scan"]).issubset(df.columns):
        df["scan_delta"] = df["end_scan"] - df["start_scan"]
    if set(["start_a3_bw","end_a3_bw"]).issubset(df.columns):
        df["a3_bw_delta"] = df["end_a3_bw"] - df["start_a3_bw"]
    if set(["start_a3_color","end_a3_color"]).issubset(df.columns):
        df["a3_color_delta"] = df["end_a3_color"] - df["start_a3_color"]
    return df

def main():
    if not os.path.exists(IN_FILE):
        raise SystemExit(f"Missing {IN_FILE} — run ingest.py first.")
    df = pd.read_csv(IN_FILE)
    df = clean_meters(df)
    df.to_parquet(OUT_FILE, index=False)
    print(f"Saved {OUT_FILE} ({len(df)} rows)")

if __name__ == "__main__":
    main()
