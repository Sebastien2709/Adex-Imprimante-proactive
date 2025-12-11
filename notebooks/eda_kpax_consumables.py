# notebooks/eda_kpax_consumables.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROC = Path("data/processed")
OUT  = Path("data/processed/eda/kpax")
OUT.mkdir(parents=True, exist_ok=True)

PCT_COLS = ["black_pct","cyan_pct","magenta_pct","yellow_pct"]
DATE_COL = "date_update"

def load_consumables():
    path = PROC / "kpax_consumables.parquet"
    df = pd.read_parquet(path)
    if DATE_COL in df.columns and not pd.api.types.is_datetime64_any_dtype(df[DATE_COL]):
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", dayfirst=True)
    return df

def hist_pct(df):
    for c in PCT_COLS:
        if c in df.columns:
            plt.figure()
            df[c].dropna().plot(kind="hist", bins=50, edgecolor="black")
            plt.title(f"Distribution — {c}")
            plt.xlabel("%")
            plt.ylabel("Fréquence")
            plt.tight_layout()
            plt.savefig(OUT / f"hist_{c}.png")
            plt.close()

def box_pct(df):
    cols = [c for c in PCT_COLS if c in df.columns]
    if not cols:
        return
    plt.figure()
    df[cols].plot(kind="box")
    plt.title("Boxplots — niveaux toner (%)")
    plt.ylabel("%")
    plt.tight_layout()
    plt.savefig(OUT / "box_pct.png")
    plt.close()

def daily_mean(df):
    if DATE_COL not in df.columns:
        return
    # moyenne journalière par couleur (sur l'ensemble du parc)
    g = (
        df.set_index(DATE_COL)
          .sort_index()[PCT_COLS]
          .resample("D")
          .mean()
    )
    g.to_csv(OUT / "daily_mean_pct.csv", index=True, encoding="utf-8")
    for c in [x for x in PCT_COLS if x in g.columns]:
        plt.figure()
        g[c].plot()
        plt.title(f"Moyenne journalière — {c}")
        plt.xlabel("Date")
        plt.ylabel("%")
        plt.tight_layout()
        plt.savefig(OUT / f"ts_daily_mean_{c}.png")
        plt.close()

def sample_series(df, n=6):
    if DATE_COL not in df.columns or "serial" not in df.columns:
        return
    # on choisit n séries avec le plus de points
    counts = df.groupby("serial")[DATE_COL].count().sort_values(ascending=False)
    pick = counts.head(n).index.tolist()
    for s in pick:
        sub = df[df["serial"]==s].sort_values(DATE_COL)
        if sub.empty:
            continue
        plt.figure()
        for c in PCT_COLS:
            if c in sub.columns:
                plt.plot(sub[DATE_COL], sub[c], label=c)
        plt.title(f"Série — {s}")
        plt.xlabel("Date")
        plt.ylabel("% toner")
        # légende simple (optionnelle ; pas de couleurs custom)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / f"series_{s}.png")
        plt.close()

def zero_flags(df):
    # ratio de lignes all_zero_flag par société & global
    if "all_zero_flag" in df.columns:
        agg = (
            df.assign(_one=1)
              .groupby(df.get("company", pd.Series(["ALL"]*len(df))), dropna=False)
              .agg(lines=("._one","sum"),
                   zeros=("all_zero_flag","sum"))
        )
        agg["zero_ratio_%"] = (agg["zeros"] / agg["lines"]).round(4) * 100
        agg.to_csv(OUT / "all_zero_ratios_by_company.csv", encoding="utf-8")
        # barplot
        plt.figure()
        agg["zero_ratio_%"].sort_values(ascending=False).plot(kind="bar", rot=45)
        plt.title("Ratio lignes 0% par société (%)")
        plt.ylabel("% de lignes 0%")
        plt.tight_layout()
        plt.savefig(OUT / "bar_zero_ratio_by_company.png")
        plt.close()

def main():
    df = load_consumables()
    # sauvegarde statistiques colonnes
    df.describe(include="all").to_csv(OUT / "describe_consumables.csv", encoding="utf-8")
    # graphiques dédiés
    hist_pct(df)
    box_pct(df)
    daily_mean(df)
    sample_series(df, n=6)
    zero_flags(df)
    print(f"[EDA KPAX] Figures & CSV exportés dans: {OUT}")

if __name__ == "__main__":
    main()
