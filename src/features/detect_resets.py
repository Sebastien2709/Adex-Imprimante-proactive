from pathlib import Path
import pandas as pd
import numpy as np


DATA_PROCESSED = Path("data/processed")
OUT_PATH = DATA_PROCESSED / "consumables_with_resets.parquet"


# --------------------
# Utilitaires
# --------------------
def norm_name(s: str) -> str:
    """Normalise un nom de colonne pour faciliter les matches."""
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    # remplacements simples
    s = (
        s.replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("%", "pct")
        .replace(" ", "_")
        .replace(".", "")
        .replace("$", "")
    )
    return s


def pick_first_existing(df, candidates):
    """Retourne le premier nom de colonne dans df parmi une liste de candidats."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# --------------------
# Détection des colonnes
# --------------------
def autodetect_columns(df: pd.DataFrame) -> dict:
    norm_map = {c: norm_name(c) for c in df.columns}

    # serial
    serial_candidates = []
    for col, nn in norm_map.items():
        if "no_serie" in nn or nn == "serial" or "serial" in nn:
            serial_candidates.append(col)
    serial_col = serial_candidates[0] if serial_candidates else None

    # company
    company_candidates = []
    for col, nn in norm_map.items():
        if "societe" in nn or "company" in nn:
            company_candidates.append(col)
    company_col = company_candidates[0] if company_candidates else None

    # ip
    ip_candidates = []
    for col, nn in norm_map.items():
        if "adresse_ip" in nn or "ip" in nn:
            ip_candidates.append(col)
    ip_col = ip_candidates[0] if ip_candidates else None

    # dates : on privilégie "date_update", sinon "date_import"
    date_update_col = None
    date_import_col = None
    for col, nn in norm_map.items():
        if "date_update" in nn or "date_mise_a_jour" in nn:
            date_update_col = col
        if "date_import" in nn:
            date_import_col = col

    # couleurs : on cherche des colonnes contenant noir / cyan / magenta / jaune
    color_cols = {"black": None, "cyan": None, "magenta": None, "yellow": None}

    for col, nn in norm_map.items():
        # on veut des colonnes qui représentent des pourcentages
        if "pct" not in nn and "%" not in col and " %" not in col:
            continue

        if any(k in nn for k in ["noir", "black", "_k_"]):
            color_cols["black"] = col if color_cols["black"] is None else color_cols["black"]
        if "cyan" in nn or "_c_" in nn:
            color_cols["cyan"] = col if color_cols["cyan"] is None else color_cols["cyan"]
        if "magenta" in nn or "_m_" in nn:
            color_cols["magenta"] = col if color_cols["magenta"] is None else color_cols["magenta"]
        if "jaune" in nn or "yellow" in nn or "_y_" in nn:
            color_cols["yellow"] = col if color_cols["yellow"] is None else color_cols["yellow"]

    # On ne garde que celles réellement trouvées
    color_cols = {k: v for k, v in color_cols.items() if v is not None}

    info = {
        "serial_col": serial_col,
        "company_col": company_col,
        "ip_col": ip_col,
        "date_update_col": date_update_col,
        "date_import_col": date_import_col,
        "color_cols": color_cols,
    }

    print("[detect_resets] autodetect_columns:", info)
    return info


# --------------------
# Passage en long & détection des resets
# --------------------
def build_long_df(cons: pd.DataFrame, info: dict) -> pd.DataFrame:
    serial_col = info["serial_col"]
    company_col = info["company_col"]
    ip_col = info["ip_col"]
    date_update_col = info["date_update_col"]
    date_import_col = info["date_import_col"]
    color_cols = info["color_cols"]

    # Sécurité minimale
    if serial_col is None:
        raise SystemExit("[detect_resets] ERREUR: impossible de trouver la colonne 'serial' / 'No serie'.")
    if date_update_col is None and date_import_col is None:
        raise SystemExit("[detect_resets] ERREUR: aucune colonne de date (Date update / Date import) trouvée.")
    if not color_cols:
        raise SystemExit(
            "[detect_resets] ERREUR: aucune colonne de pourcentage de toner (noir/cyan/magenta/jaune) détectée."
        )

    work = cons.copy()

    # --- Nettoyage des colonnes dates en texte brut ---
    for col in [date_update_col, date_import_col]:
        if col is not None and col in work.columns:
            work[col] = (
                work[col]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.strip()
            )

    # --- Nettoyage des colonnes % toner (texte → propre) ---
    for col in color_cols.values():
        if col in work.columns:
            work[col] = (
                work[col]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", ".", regex=False)
                .str.strip()
            )

    # --- Parsing des dates avec fallback Date import si Date update est pourrie ---
    def parse_dates(raw: pd.Series) -> pd.Series:
        raw = raw.astype(str)
        dt = pd.to_datetime(raw, errors="coerce", dayfirst=True)
        if dt.isna().mean() > 0.95:
            # tentative format "08/02/23 08:11"
            dt = pd.to_datetime(raw, format="%d/%m/%y %H:%M", errors="coerce")
        if dt.isna().mean() > 0.95:
            # tentative format "08/02/23"
            dt = pd.to_datetime(raw, format="%d/%m/%y", errors="coerce")
        return dt

    date_primary = None
    nat_ratio_primary = 1.0

    if date_update_col is not None and date_update_col in work.columns:
        date_primary = parse_dates(work[date_update_col])
        nat_ratio_primary = float(date_primary.isna().mean())
        print(f"[detect_resets] date_update parsed NaT ratio: {nat_ratio_primary:.3f}")

    # Si Date update est quasi vide, on tente Date import
    if (date_primary is None or nat_ratio_primary > 0.90) and date_import_col is not None and date_import_col in work.columns:
        date_import_parsed = parse_dates(work[date_import_col])
        nat_ratio_import = float(date_import_parsed.isna().mean())
        print(f"[detect_resets] date_import parsed NaT ratio: {nat_ratio_import:.3f}")

        if date_primary is None:
            date_primary = date_import_parsed
        else:
            # On complète les NaT de date_update avec date_import
            date_primary = date_primary.fillna(date_import_parsed)

    work["date_std"] = date_primary

    # Renommage des colonnes standardisées
    rename_map = {
        serial_col: "serial",
        "date_std": "date_update",
    }
    if company_col:
        rename_map[company_col] = "company"
    if ip_col:
        rename_map[ip_col] = "ip"

    work = work.rename(columns=rename_map)

    # Colonnes de base qu'on transporte
    base_cols = ["serial", "date_update"]
    if "company" in work.columns:
        base_cols.append("company")
    if "ip" in work.columns:
        base_cols.append("ip")

    # --- Passage en long ---
    long_list = []
    for color_name, col in color_cols.items():
        tmp = work[base_cols + [col]].copy()
        tmp = tmp.rename(columns={col: "pct"})
        tmp["color"] = color_name
        long_list.append(tmp)

    long_df = pd.concat(long_list, ignore_index=True)

    # Conversion du % en numérique
    long_df["pct"] = pd.to_numeric(long_df["pct"], errors="coerce")

    # Filtre : on garde où la date est valide
    before = len(long_df)
    long_df = long_df.dropna(subset=["date_update"])
    print(f"[detect_resets] long_df (après filtre date): {before} -> {len(long_df)} lignes")

    if len(long_df) == 0:
        print("[detect_resets] WARN: long_df est vide après filtre date (même avec fallback Date import).")
        return long_df

    # Filtre % raisonnable
    long_df = long_df[(long_df["pct"] >= 0) & (long_df["pct"] <= 100)]
    long_df = long_df.dropna(subset=["pct"])
    print(f"[detect_resets] long_df (après filtre pct): {len(long_df)} lignes")

    if len(long_df) == 0:
        print("[detect_resets] WARN: long_df est vide après filtre pct.")
        return long_df

    # --- Détection des resets comme avant ---
    long_df = long_df.sort_values(["serial", "color", "date_update"])

    long_df["pct_diff"] = long_df.groupby(["serial", "color"])["pct"].diff()
    long_df["is_reset"] = (long_df["pct_diff"] > 50) & (long_df["pct"] >= 60)
    long_df["cycle_id"] = (
        long_df.groupby(["serial", "color"])["is_reset"]
        .cumsum()
        .fillna(0)
        .astype(int)
    )

    return long_df



def main():
    cons_path = DATA_PROCESSED / "kpax_consumables.parquet"
    if not cons_path.exists():
        raise SystemExit(f"[detect_resets] ERREUR: fichier introuvable: {cons_path}")

    cons = pd.read_parquet(cons_path)
    print(f"[detect_resets] loaded kpax_consumables: {len(cons)} lignes, colonnes = {list(cons.columns)}")

    info = autodetect_columns(cons)
    long_df = build_long_df(cons, info)

    rows = len(long_df)
    uniq_serials = long_df["serial"].nunique() if rows > 0 else 0
    missing_serial_rows = long_df["serial"].isna().sum() if rows > 0 else 0

    print(f"[detect_resets] rows: {rows} | unique serials: {uniq_serials} | missing_serial_rows: {missing_serial_rows}")

    long_df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH} ({rows} rows)")


if __name__ == "__main__":
    main()
