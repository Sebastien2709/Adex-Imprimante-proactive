from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA = Path("data/processed")
OUT = Path("data/processed/eda/cross")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 1️⃣ Charger les datasets
# ---------------------------------------------------------
forecasts = pd.read_parquet(DATA / "consumables_forecasts.parquet")
kpax = pd.read_parquet(DATA / "kpax_consumables.parquet")
ledger = pd.read_parquet(DATA / "item_ledger.parquet")

print(f"[LOAD] forecasts={len(forecasts)} kpax={len(kpax)} ledger={len(ledger)}")

# ---------------------------------------------------------
# 2️⃣ Préparer les clés communes
# ---------------------------------------------------------
# Normaliser les noms de série
def norm_serial(s):
    if not isinstance(s, str):
        return None
    return s.strip().upper()

forecasts["serial_norm"] = forecasts["serial"].map(norm_serial)
kpax["serial_norm"] = kpax["No serie"].map(norm_serial)
ledger["serial_norm"] = ledger["No. serie"].map(norm_serial)

# ---------------------------------------------------------
# 3️⃣ Agréger KPAX (moyenne des % encre)
# ---------------------------------------------------------
color_cols = ["% noir", "% cyan", "% magenta", "% jaune"]
for c in color_cols:
    if c in kpax.columns:
        kpax[c] = pd.to_numeric(kpax[c], errors="coerce")

kpax_agg = (
    kpax.groupby("serial_norm")[color_cols]
        .mean()
        .reset_index()
        .rename(columns={c: f"avg_{c.replace('% ','')}" for c in color_cols})
)
print(f"[KPAX agg] {kpax_agg.shape}")

# ---------------------------------------------------------
# 4️⃣ Agréger Ledger (nb d’envois par série)
# ---------------------------------------------------------
ledger_agg = (
    ledger.groupby("serial_norm")
          .agg(nb_shipments=("No document", "count"))
          .reset_index()
)
print(f"[Ledger agg] {ledger_agg.shape}")

# ---------------------------------------------------------
# 5️⃣ Fusionner les trois mondes
# ---------------------------------------------------------
merged = (
    forecasts.merge(kpax_agg, on="serial_norm", how="left")
             .merge(ledger_agg, on="serial_norm", how="left")
)
merged["nb_shipments"] = merged["nb_shipments"].fillna(0)
print(f"[Merged] {merged.shape}")

# ---------------------------------------------------------
# 6️⃣ Graphes de corrélation
# ---------------------------------------------------------

# --- Scatter 1 : niveau réel vs jours restants
plt.figure()
plt.scatter(merged["avg_noir"], merged["days_left_est"], alpha=0.4)
plt.title("Corrélation niveau noir (réel KPAX) vs jours restants (prévision)")
plt.xlabel("Niveau moyen noir (%)")
plt.ylabel("Jours restants (prévision)")
plt.tight_layout()
plt.savefig(OUT / "scatter_avg_noir_vs_days_left.png")
plt.close()

# --- Scatter 2 : nb d’envois vs jours restants (plus il y a d’envois, plus les ruptures sont fréquentes)
plt.figure()
plt.scatter(merged["nb_shipments"], merged["days_left_est"], alpha=0.4)
plt.title("Nb d'envois vs jours restants (prévision)")
plt.xlabel("Nombre d'envois (Ledger)")
plt.ylabel("Jours restants (Forecast)")
plt.tight_layout()
plt.savefig(OUT / "scatter_shipments_vs_days_left.png")
plt.close()

# --- Histogramme des écarts
plt.figure()
merged["days_left_est"].clip(upper=200).hist(bins=50)
plt.title("Distribution des jours restants avant rupture")
plt.xlabel("Jours restants (forecast)")
plt.ylabel("Nombre de machines")
plt.tight_layout()
plt.savefig(OUT / "hist_days_left.png")
plt.close()

# --- Heatmap moyenne consommation vs prévision (simplifiée)
import seaborn as sns
plt.figure(figsize=(6,5))
sns.heatmap(
    merged[["avg_noir","avg_cyan","avg_magenta","avg_jaune","days_left_est"]]
          .corr(),
    annot=True, cmap="coolwarm", fmt=".2f"
)
plt.title("Corrélation entre niveaux réels et jours restants")
plt.tight_layout()
plt.savefig(OUT / "heatmap_corr_levels_vs_forecast.png")
plt.close()

print(f"[EDA Cross] Graphes exportés -> {OUT}")
