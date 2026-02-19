import pandas as pd

df = pd.read_csv(
    "data/processed/kpax_history_light.csv",
    low_memory=False,
    dtype={"serial_display": str}   # ← force string dès la lecture
)

# Nettoie les valeurs parasites
df["serial_display"] = df["serial_display"].astype(str).str.strip()

df.to_parquet("data/processed/kpax_history_light.parquet", index=False)
print(f"✅ Parquet généré : {len(df)} lignes")