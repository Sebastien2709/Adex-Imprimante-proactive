from pathlib import Path
import pandas as pd


DATA_PROCESSED = Path("data/processed")


def ensure_serial_norm(df: pd.DataFrame) -> pd.DataFrame:
    """
    S'assure qu'on a une colonne 'serial_norm' en majuscules, sans espaces.
    On essaie plusieurs noms possibles (serial, $No serie$, etc.).
    """
    if "serial_norm" in df.columns:
        df["serial_norm"] = df["serial_norm"].astype(str).str.upper().str.strip()
        return df

    # colonnes candidates possibles
    candidates = [
        "serial",
        "Serial",
        "no_serie",
        "NoSerie",
        "$No serie$",
        "serial_kpax",
        "serial_bc",
    ]

    for c in candidates:
        if c in df.columns:
            df["serial_norm"] = df[c].astype(str).str.upper().str.strip()
            return df

    # fallback : on crée un serial_norm vide si vraiment rien n'existe
    df["serial_norm"] = ""
    return df


def load_replenishments_v21() -> pd.DataFrame:
    """
    Charge les recommandations V2.1 (fichier métier existant).
    """
    path = DATA_PROCESSED / "replenishments_to_create_v21.csv"
    # ATTENTION : le fichier a été écrit avec sep=";" dans predict_xgb_v21
    df = pd.read_csv(path, sep=";")
    df = ensure_serial_norm(df)
    return df



def load_contracted_devices() -> pd.DataFrame:
    """
    Charge les devices sous contrat (build_contracted_devices).
    """
    path = DATA_PROCESSED / "contracted_devices.parquet"
    df = pd.read_parquet(path)
    df = ensure_serial_norm(df)
    return df


def build_business_export():
    # 1) Charger les données
    rep = load_replenishments_v21()
    devices = load_contracted_devices()

    print(f"[enrich_v21] recommandations V2.1 : {len(rep)} lignes")
    print(f"[enrich_v21] devices sous contrat : {len(devices)} lignes")

    # 2) Merge sur serial_norm
    merged = rep.merge(
        devices,
        on="serial_norm",
        how="left",
        suffixes=("", "_dev"),
    )

    print(f"[enrich_v21] merged shape = {merged.shape}")

    # 3) Sélection des colonnes “métier” intéressantes

    # Colonnes déjà dans le CSV recommandations qu'on garde en priorité
    base_cols = [c for c in rep.columns]

    # Colonnes métier potentielles côté devices
    candidate_device_cols = [
        # client / société
        "customer_name",
        "client_name",
        "Societe",
        "societe",
        "Code client",
        "customer_no",
        "no_client",

        # contrat
        "contract_id",
        "contract_no",
        "No contrat",
        "no_contrat",
        "relation_type",

        # machine
        "device_model",
        "model",
        "Modele",
        "modele_machine",
        "brand",
        "marque",

        # localisation
        "site_name",
        "site",
        "Ville",
        "CP",
        "Code Postal",
        "Adresse",
        "Adresse site",
    ]

    device_cols = [c for c in candidate_device_cols if c in merged.columns]

    # On évite les doublons
    export_cols = []
    for c in base_cols + device_cols:
        if c not in export_cols and c in merged.columns:
            export_cols.append(c)

    export_df = merged[export_cols].copy()

    # 4) Sauvegarde
    out_path = DATA_PROCESSED / "replenishments_to_create_v21_enriched.csv"
    export_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 5) Quelques stats utiles
    missing_company = 0
    for col in ["customer_name", "client_name", "Societe", "societe"]:
        if col in export_df.columns:
            missing_company = export_df[col].isna().sum()
            break

    print(f"[enrich_v21] export sauvegardé -> {out_path}")
    print(f"[enrich_v21] lignes exportées : {len(export_df)}")
    if missing_company:
        print(f"[enrich_v21] lignes sans info client sur la colonne choisie : {missing_company}")


def main():
    build_business_export()


if __name__ == "__main__":
    main()
