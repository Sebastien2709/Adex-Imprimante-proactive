from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROC = Path("data/processed")
OUT  = Path("data/processed/eda/forecasts")
OUT.mkdir(parents=True, exist_ok=True)

def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

def main():
    created = []

    fpath = PROC / "consumables_forecasts.parquet"
    if not fpath.exists():
        print(f"[EDA forecasts] Missing file: {fpath}")
        return

    f = pd.read_parquet(fpath)
    # diagnostics rapides
    print(f"[EDA forecasts] rows={len(f)}  cols={list(f.columns)}")

    # 1) Jours restants
    if "days_left_est" in f.columns:
        ser = f["days_left_est"].dropna()
        if not ser.empty:
            ser = ser.clip(-60, 180)
            plt.figure()
            ser.plot(kind="hist", bins=48, edgecolor="black")
            plt.title("Distribution des jours restants (clippés -60..180)")
            plt.xlabel("jours")
            created.append(savefig(OUT / "hist_days_left.png"))

    # 2) Stockout dates par semaine
    if "stockout_date" in f.columns:
        f["stockout_date"] = pd.to_datetime(f["stockout_date"], errors="coerce")
        wk = f.set_index("stockout_date").sort_index().resample("W").size()
        if wk.notna().any() and wk.sum() > 0:
            plt.figure()
            wk.plot()
            plt.title("Volume de stockouts prévus (par semaine)")
            plt.ylabel("nb prévisions")
            created.append(savefig(OUT / "ts_stockouts_weekly.png"))

    # 3) should_order_now par couleur
    if {"should_order_now","color"}.issubset(f.columns):
        tmp = f.assign(so=f["should_order_now"].astype(bool))
        if not tmp.empty:
            counts = tmp.groupby("color", dropna=False)["so"].agg(["sum","count"])
            if not counts.empty:
                counts["rate_%"] = (counts["sum"]/counts["count"]*100).round(1)
                counts.sort_values("sum").to_csv(OUT / "should_order_now_by_color.csv")
                plt.figure()
                counts["sum"].plot(kind="bar", rot=0)
                plt.title("Commandes à déclencher — par couleur")
                created.append(savefig(OUT / "bar_should_now_by_color.png"))

    # 4) Niveau vs jours restants (scatter)
    if {"level_now_pct","days_left_est"}.issubset(f.columns):
        sc = f[["level_now_pct","days_left_est"]].dropna()
        if not sc.empty:
            plt.figure()
            sc.plot.scatter(x="level_now_pct", y="days_left_est", alpha=0.3)
            plt.title("Niveau actuel (%) vs jours restants")
            created.append(savefig(OUT / "scatter_level_vs_days_left.png"))

    # 5) should_order_now par semaine
    if {"should_order_now","stockout_date"}.issubset(f.columns):
        g = (f.assign(so=f["should_order_now"].astype(bool))
                .set_index("stockout_date")
                .sort_index()
                .resample("W")["so"].sum())
        if g.notna().any() and g.sum() > 0:
            plt.figure()
            g.plot()
            plt.title("Commandes à déclencher (somme hebdo)")
            created.append(savefig(OUT / "ts_should_now_weekly.png"))

    # Fichier de log de contenu
    with open(OUT / "_created_files.txt", "w", encoding="utf-8") as w:
        for p in created:
            w.write(str(p) + "\n")

    print("[EDA forecasts] export ->", OUT)
    if created:
        print("  Created files:")
        for p in created:
            print("   -", p.name)
    else:
        print("  (no figures created — check columns/emptiness)")

if __name__ == "__main__":
    main()
