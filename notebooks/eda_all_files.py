# notebooks/eda_all_files.py
import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Config ----------
INPUT_DIRS = ["data/processed", "data/interim", "data/raw"]  # ordre de priorité
OUTPUT_DIR = Path("data/processed/eda/generic")
MAX_HISTS_PER_FILE = 10           # limite d'histogrammes numériques
TOP_CATS = 20                     # top N catégories pour barplots
HIGH_CARD_LIMIT = 2000            # au-delà, on ne fait pas de barplot
DATE_COL_HINTS = {"date", "datetime", "update", "import", "releve", "ship", "doc"}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Helpers ----------

def is_date_col(col: str) -> bool:
    c = col.lower()
    return any(h in c for h in DATE_COL_HINTS)

def looks_iso_date_series(s: pd.Series, sample=200) -> bool:
    """Heuristique: si >80% des valeurs non nulles ressemblent à YYYY-MM-DD, on considère ISO."""
    patt = re.compile(r"^\d{4}-\d{2}-\d{2}")
    vals = s.dropna().astype(str).head(sample)
    if vals.empty:
        return False
    hits = sum(1 for x in vals if patt.match(x))
    return hits / len(vals) >= 0.8

def safe_read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".parquet"]:
        return pd.read_parquet(path)
    # CSV/TXT: auto-sep simple; tes bruts étant “|$”, on suppose déjà nettoyés ici
    try:
        return pd.read_csv(path)
    except Exception:
        # dernier recours : “|” (cas interim d’ingest)
        return pd.read_csv(path, sep="|")

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Résumé colonne par colonne. IMPORTANT:
    - on traite les booléens comme catégories (pas de quantiles dessus)
    - on calcule les quantiles uniquement pour les numériques non-booléens
    """
    info = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        kind = s.dtype
        na = int(s.isna().sum())
        ratio = round(100*na/max(n,1), 2)
        entry = {"column": col, "dtype": str(kind), "n_missing": na, "missing_%": ratio}

        is_bool = pd.api.types.is_bool_dtype(s)
        is_num  = pd.api.types.is_numeric_dtype(s) and not is_bool
        is_dt   = pd.api.types.is_datetime64_any_dtype(s)

        if is_num:
            entry.update({
                "min": s.min(skipna=True),
                "q25": s.quantile(0.25),
                "median": s.median(skipna=True),
                "mean": s.mean(skipna=True),
                "q75": s.quantile(0.75),
                "max": s.max(skipna=True),
            })
        elif is_dt:
            entry.update({
                "min": s.min(skipna=True),
                "max": s.max(skipna=True),
                "n_unique": s.nunique(dropna=True),
            })
        elif is_bool:
            # bool -> stats adaptées
            vc = s.value_counts(dropna=True)
            entry.update({
                "n_unique": int(vc.shape[0]),
                "true_count": int(vc.get(True, 0)),
                "false_count": int(vc.get(False, 0)),
            })
        else:
            entry.update({"n_unique": s.nunique(dropna=True)})

        info.append(entry)
    return pd.DataFrame(info)

def plot_numeric_hist(df: pd.DataFrame, base_out: Path):
    # pas d’histo sur booléens
    num_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    for i, col in enumerate(num_cols[:MAX_HISTS_PER_FILE], start=1):
        series = df[col].dropna()
        if series.empty:
            continue
        plt.figure()
        series.plot(kind="hist", bins=50, edgecolor="black")
        plt.title(f"Histogramme — {col}")
        plt.xlabel(col)
        plt.ylabel("Fréquence")
        plt.tight_layout()
        plt.savefig(base_out / f"hist_{i:02d}_{col}.png")
        plt.close()

def plot_categorical_bars(df: pd.DataFrame, base_out: Path):
    # objets + catégories + booléens (représentés en bar)
    candidate_cols = []
    for c in df.columns:
        s = df[c]
        from pandas.api.types import CategoricalDtype
        if pd.api.types.is_bool_dtype(s) or s.dtype == "object" or isinstance(s.dtype, CategoricalDtype):
            nunique = s.nunique(dropna=True)
            if 1 < nunique <= HIGH_CARD_LIMIT:  # évite cardinalité énorme
                candidate_cols.append(c)

    for col in candidate_cols:
        series = df[col].astype("object")
        series = series.fillna("NA")
        series = series.infer_objects(copy=False)
        vc = series.value_counts().head(TOP_CATS)
        if vc.empty:
            continue
        plt.figure()
        vc.plot(kind="bar", rot=45)
        plt.title(f"Top {TOP_CATS} — {col}")
        plt.xlabel(col)
        plt.ylabel("Comptes")
        plt.tight_layout()
        plt.savefig(base_out / f"bar_{col}.png")
        plt.close()

ISO_FULL = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?$")
EU_DDMM  = re.compile(r"^\d{2}/\d{2}/\d{4}")

def coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if is_date_col(c) and not pd.api.types.is_datetime64_any_dtype(df[c]):
            s = df[c].astype(str)
            sample = s.dropna().head(200)
            if len(sample) and sample.map(lambda x: bool(ISO_FULL.match(x))).mean() >= 0.8:
                df[c] = pd.to_datetime(df[c], errors="coerce", format="%Y-%m-%d", utc=False)
            elif len(sample) and sample.map(lambda x: bool(EU_DDMM.match(x))).mean() >= 0.8:
                df[c] = pd.to_datetime(df[c], errors="coerce", format="%d/%m/%Y", dayfirst=True)
            else:
                # fallback (ton code actuel) selon looks_iso_date_series
                if looks_iso_date_series(df[c]):
                    df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=False)
                else:
                    df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
    return df

def process_file(path: Path):
    print(f"[EDA] {path}")
    df = safe_read(path)
    if df.empty:
        print("  -> empty, skip")
        return

    # coercion dates sur heuristique
    df = coerce_dates(df)

    # dossier de sortie par fichier
    base = OUTPUT_DIR / path.stem
    base.mkdir(parents=True, exist_ok=True)

    # résumé colonnes
    schema_df = summarize(df)
    schema_df.to_csv(base / "summary_columns.csv", index=False, encoding="utf-8")

    # numeric hists (hors bool)
    plot_numeric_hist(df, base)

    # categorical bars (inclut bool)
    plot_categorical_bars(df, base)

    # si on a une colonne temporelle "dominante", courbe de volume dans le temps
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if date_cols:
        # choisir la colonne temporelle la plus renseignée
        dcol = sorted(date_cols, key=lambda c: df[c].notna().sum(), reverse=True)[0]
        try:
            daily = df.set_index(dcol).sort_index()
            daily_count = daily.resample("D").size()
            if daily_count.notna().any():
                plt.figure()
                daily_count.plot()
                plt.title(f"Volume par jour — index: {dcol}")
                plt.xlabel("Date")
                plt.ylabel("N lignes")
                plt.tight_layout()
                plt.savefig(base / f"ts_volume_by_day_{dcol}.png")
                plt.close()
        except Exception as e:
            print(f"  -> time series plot skipped ({e})")

def main():
    seen = set()
    for folder in INPUT_DIRS:
        p = Path(folder)
        if not p.exists():
            continue
        for ext in ("*.csv", "*.parquet"):
            for f in p.rglob(ext):
                key = str(f.resolve())
                if key in seen:
                    continue
                seen.add(key)
                process_file(f)

if __name__ == "__main__":
    main()
