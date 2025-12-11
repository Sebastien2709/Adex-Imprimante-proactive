from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROC = Path("data/processed")
OUT = Path("data/processed/eda/meters")
OUT.mkdir(parents=True, exist_ok=True)

# paires Debut/Fin repérées dans tes colonnes
PAIRS = [
    ("Debut A4 NB", "Fin A4 NB"),
    ("Debut A4 CO", "Fin A4 CO"),
    ("Debut SC",    "Fin SC"),
    ("Debut A3 NB", "Fin A3 NB"),
    ("Debut A3 CO", "Fin A3 CO"),
]

def main():
    df = pd.read_parquet(PROC / "meters.parquet")

    # parse date (format vu : 31/12/22)
    # on privilégie "Date releve" (photo de fin de période). Sinon "Date fin".
    dcol = "Date releve" if "Date releve" in df.columns else ("Date fin" if "Date fin" in df.columns else None)
    if dcol is None:
        print("⚠️ Pas de colonne date ('Date releve' / 'Date fin').")
        return
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce", dayfirst=True)

    # calcule les deltas Fin - Début
    deltas = {}
    for deb, fin in PAIRS:
        if deb in df.columns and fin in df.columns:
            name = f"delta {deb.split('Debut ')[-1]}"  # ex. "delta A4 NB"
            deltas[name] = (pd.to_numeric(df[fin], errors="coerce") - pd.to_numeric(df[deb], errors="coerce")).clip(lower=0)
    delta_df = pd.DataFrame(deltas)
    df = pd.concat([df[[dcol]], delta_df], axis=1)

    # histogrammes de chaque delta
    for c in delta_df.columns:
        ser = delta_df[c].dropna()
        if ser.empty:
            continue
        ser = ser.clip(upper=ser.quantile(0.99))  # évite queue extrême
        plt.figure()
        ser.plot(kind="hist", bins=50, edgecolor="black")
        plt.title(f"Distribution — {c}")
        plt.xlabel("Pages / période")
        plt.tight_layout()
        plt.savefig(OUT / f"hist_{c.replace(' ','_')}.png")
        plt.close()

    # timeline hebdo (somme des deltas)
    weekly = (
        df.set_index(dcol)
          .sort_index()[delta_df.columns]
          .resample("W")
          .sum()
          .sum(axis=1)  # somme de tous les canaux
    )
    plt.figure()
    weekly.plot()
    plt.title("Production totale (pages / semaine)")
    plt.ylabel("Nombre de pages")
    plt.xlabel(dcol)
    plt.tight_layout()
    plt.savefig(OUT / "ts_weekly_pages.png")
    plt.close()

    print(f"[EDA Meters] OK -> {OUT} | colonnes delta = {list(delta_df.columns)}")

if __name__ == "__main__":
    main()
