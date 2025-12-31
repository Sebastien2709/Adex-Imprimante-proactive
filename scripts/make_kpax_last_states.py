import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
KPAX_PATH = BASE_DIR / "data" / "processed" / "kpax_consumables.parquet"
OUT_CSV = BASE_DIR / "data" / "processed" / "kpax_last_states.csv"

serial_col = "$No serie$"
date_update_col = "$Date update$"
date_import_col = "$Date import$"
color_cols = {
    "black": "$% noir$",
    "cyan": "$% cyan$",
    "magenta": "$% magenta$",
    "yellow": "$% jaune$",
}

cols_to_read = [serial_col, date_update_col, date_import_col] + list(color_cols.values())
df = pd.read_parquet(KPAX_PATH, columns=cols_to_read)

def parse_date(series: pd.Series):
    raw = series.astype(str).str.strip().str.strip("$")
    return pd.to_datetime(raw, errors="coerce", dayfirst=True)

d_up = parse_date(df[date_update_col])
d_im = parse_date(df[date_import_col])
date_chosen = d_up.where(d_up.notna(), d_im)

serial_display = df[serial_col].astype(str).str.replace("$", "", regex=False).str.strip()

parts = []
for color, col in color_cols.items():
    pct = pd.to_numeric(df[col].astype(str).str.strip().str.strip("$"), errors="coerce")
    tmp = pd.DataFrame({
        "serial_display": serial_display,
        "color": color,
        "last_update": date_chosen,
        "last_pct": pct
    }).dropna(subset=["last_update"])
    parts.append(tmp)

long_df = pd.concat(parts, ignore_index=True)
idx = long_df.groupby(["serial_display", "color"])["last_update"].idxmax()
last_df = long_df.loc[idx].copy()

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
last_df.to_csv(OUT_CSV, index=False)
print("Wrote:", OUT_CSV, "rows:", len(last_df))
