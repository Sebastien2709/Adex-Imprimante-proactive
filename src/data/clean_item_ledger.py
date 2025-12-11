# src/data/clean_item_ledger.py
import os
import pandas as pd
from src.data.utils import parse_eu_datetime  # imports inutiles retirés

RAW = "data/raw"
PROC = "data/processed"
INTERIM = "data/interim"
os.makedirs(PROC, exist_ok=True)

IN_FILE = os.path.join(INTERIM, "item_ledger.csv")
OUT_FILE = os.path.join(PROC, "item_ledger.parquet")


def _norm_serial(s):
    return str(s).strip().upper() if pd.notna(s) else None


def clean_item_ledger(df: pd.DataFrame) -> pd.DataFrame:
    # --- dates ---
    if "doc_date" in df.columns:
        # parse EU en datetime (conserve doc_date d'origine si tu veux)
        df["doc_datetime"] = df["doc_date"].apply(parse_eu_datetime)

    # --- numériques ---
    for c in ("qty", "capacity_pages"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- normalisations simples ---
    if "serial" in df.columns:
        # NaN si vide puis serial_norm pour les joins
        df["serial"] = df["serial"].replace({"": pd.NA})
        df["serial_norm"] = df["serial"].apply(_norm_serial)

    # --- classification de la relation client ---
    def classify_relation(row):
        has_contract = pd.notna(row.get("contract_no")) and str(row.get("contract_no")).strip() != ""
        has_serial   = pd.notna(row.get("serial")) and str(row.get("serial")).strip() != ""
        if has_contract and has_serial:
            return "contrat_adex"            # imprimante sous contrat
        elif has_serial and not has_contract:
            return "machine_hors_contrat"    # suivie sans contrat
        else:
            return "commande_libre"          # commande libre sans suivi machine

    df["relation_type"] = df.apply(classify_relation, axis=1)

    # (facultatif) trim des colonnes texte utiles
    for col in ("article_no", "designation", "consumable_type", "company"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().where(df[col].notna())

    return df


def main():
    if not os.path.exists(IN_FILE):
        raise SystemExit(f"Missing {IN_FILE} — run ingest.py first.")

    df = pd.read_csv(IN_FILE)
    df = clean_item_ledger(df)
    df.to_parquet(OUT_FILE, index=False)

    # petit log utile
    if "relation_type" in df.columns:
        counts = df["relation_type"].value_counts(dropna=False).to_dict()
        print(f"[clean_item_ledger] relation_type counts: {counts}")

    print(f"Saved {OUT_FILE} ({len(df)} rows)")


if __name__ == "__main__":
    main()
