from pathlib import Path
from datetime import datetime
import pandas as pd


DATA_PROCESSED = Path("data/processed")
OUTPUT_DIR = Path("data/outputs")


def choose_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Retourne le premier nom de colonne existant dans df parmi la liste."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_enriched_recos() -> pd.DataFrame:
    """
    Charge les recommandations enrichies V2.1.
    On part de replenishments_to_create_v21_enriched.csv,
    et si jamais il n'existe pas, on retombe sur replenishments_to_create_v21.csv.
    """
    enriched_path = DATA_PROCESSED / "replenishments_to_create_v21_enriched.csv"
    basic_path = DATA_PROCESSED / "replenishments_to_create_v21.csv"

    if enriched_path.exists():
        print(f"[export_recos] Chargement de {enriched_path}")
        df = pd.read_csv(enriched_path)  # ce fichier est en sep=","
    elif basic_path.exists():
        print(f"[export_recos] ATTENTION: fichier enrichi absent, fallback sur {basic_path}")
        df = pd.read_csv(basic_path, sep=";")
    else:
        raise FileNotFoundError("Aucun fichier de recommandations trouvé.")
    return df


def build_business_export() -> pd.DataFrame:
    df = load_enriched_recos()

    # ─────────────────────────────────────────────
    # 1) Détection des colonnes clés
    # ─────────────────────────────────────────────

    # Série / machine
    serial_col = choose_first_existing(df, ["serial_norm", "serial", "Serial"])
    if serial_col is None:
        df["serial_norm"] = ""
        serial_col = "serial_norm"

    # Couleur / toner
    color_col = choose_first_existing(df, ["color", "Color", "couleur"])

    # Jours restants ML
    days_left_col = choose_first_existing(df, ["days_left_v21", "days_left_ml_v21"])
    if days_left_col is None:
        raise ValueError("Impossible de trouver days_left_v21 ou days_left_ml_v21")

    # Date rupture prédite
    stockout_col = choose_first_existing(df, ["stockout_date_ml_v21", "stockout_date"])

    # ➤ Client : on ajoute "company" (important)
    client_col = choose_first_existing(
        df, ["customer_name", "client_name", "Societe", "societe", "company"]
    )

    # ➤ Code client (pas présent, mais on laisse le fallback)
    client_code_col = choose_first_existing(
        df, ["customer_no", "no_client", "Code client", "code_client"]
    )

    # ➤ Contrat : "contract_no" est maintenant reconnu
    contract_col = choose_first_existing(
        df, ["contract_no", "contract_id", "No contrat", "no_contrat", "contract"]
    )

    # Modèle / marque (pas dispo dans ton fichier, mais on garde le fallback propre)
    model_col = choose_first_existing(df, ["model", "Modele", "device_model"])
    brand_col = choose_first_existing(df, ["brand", "marque"])

    # Ville / CP (pas dispo non plus pour l’instant)
    city_col = choose_first_existing(df, ["Ville", "City", "city"])
    postal_col = choose_first_existing(df, ["CP", "Code Postal", "PostalCode"])

    # ─────────────────────────────────────────────
    # 2) Construction du dataframe final
    # ─────────────────────────────────────────────

    today = datetime.today().date()
    out = pd.DataFrame(index=df.index)
    out["date_reco"] = today.strftime("%Y-%m-%d")

    # Série
    out["serial"] = df[serial_col].astype(str)

    # Nettoyage $ de serial
    out["serial"] = out["serial"].str.replace("$", "", regex=False).str.strip()

    # Client / contrat
    out["client"] = df[client_col].astype(str) if client_col else ""
    out["contrat"] = df[contract_col].astype(str) if contract_col else ""

    # Nettoyage $ sur contrat
    out["contrat"] = out["contrat"].str.replace("$", "", regex=False).str.strip()


        # --- Type de livraison selon présence serial / contrat ---
    def infer_type_livraison(row):
        serial_ok = bool(str(row["serial"]).strip())
        contrat_ok = bool(str(row["contrat"]).strip())

        if serial_ok and contrat_ok:
            return "contrat_maintenance"
        elif serial_ok and not contrat_ok:
            return "commande_simple"
        elif not serial_ok and contrat_ok:
            return "stock_seulement"
        else:
            return "inconnu"

    out["type_livraison"] = out.apply(infer_type_livraison, axis=1)

    # Code client
    out["client_code"] = df[client_code_col].astype(str) if client_code_col else ""

    # Modèle / marque
    out["modele_machine"] = df[model_col].astype(str) if model_col else ""
    out["marque"] = df[brand_col].astype(str) if brand_col else ""

    # Couleur / toner
    out["couleur"] = df[color_col].astype(str) if color_col else ""
    out["toner"] = out["couleur"]  # pas encore de code article disponible

    # Localisation
    out["ville"] = df[city_col].astype(str) if city_col else ""
    out["code_postal"] = df[postal_col].astype(str) if postal_col else ""

    # Jours restants & rupture
    out["jours_avant_rupture"] = (
        pd.to_numeric(df[days_left_col], errors="coerce").fillna(-1).astype(int)
    )

    if stockout_col:
        out["date_rupture_estimee"] = pd.to_datetime(
            df[stockout_col], errors="coerce"
        ).dt.date.astype(str)
    else:
        out["date_rupture_estimee"] = ""

    # ─────────────────────────────────────────────
    # 3) Priorité
    # ─────────────────────────────────────────────

    def compute_priority(d):
        if d <= 3:
            return 1
        elif d <= 7:
            return 2
        elif d <= 14:
            return 3
        else:
            return 4

    out["priorite"] = out["jours_avant_rupture"].apply(compute_priority)

    # ─────────────────────────────────────────────
    # 4) Commentaire
    # ─────────────────────────────────────────────

    def comment(d):
        if d <= 3:
            return "Risque de rupture imminent"
        elif d <= 7:
            return "A planifier rapidement"
        elif d <= 14:
            return "Peut être regroupé avec d'autres envois"
        else:
            return "Surveillance standard"

    out["commentaire"] = out["jours_avant_rupture"].apply(comment)

    # ─────────────────────────────────────────────
    # 5) Tri final
    # ─────────────────────────────────────────────

    out = out.sort_values(["priorite", "client", "jours_avant_rupture"])

    return out


def save_business_export(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.today().date()

    dated_filename = OUTPUT_DIR / f"recommandations_toners_{today.strftime('%Y%m%d')}.csv"
    latest_filename = OUTPUT_DIR / "recommandations_toners_latest.csv"

    df.to_csv(dated_filename, index=False, encoding="utf-8-sig", sep=";")
    df.to_csv(latest_filename, index=False, encoding="utf-8-sig", sep=";")

    print(f"[export_recos] Fichier business (daté) -> {dated_filename}")
    print(f"[export_recos] Fichier business (latest) -> {latest_filename}")
    print(f"[export_recos] Lignes exportées : {len(df)}")


def main():
    df_out = build_business_export()
    save_business_export(df_out)


if __name__ == "__main__":
    main()
