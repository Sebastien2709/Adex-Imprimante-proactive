import os
import pandas as pd
from typing import Optional

PROC = "data/processed"
LEDGER = os.path.join(PROC, "item_ledger.parquet")
OUT_CONTRACTED = os.path.join(PROC, "contracted_devices.parquet")
OUT_REL = os.path.join(PROC, "serial_relations.parquet")


def norm(s: str) -> str:
    """Normalise un nom de colonne pour matcher même les trucs du style $No. serie$."""
    repl = str(s).lower().strip()

    # On enlève les $ en plus
    for ch in [" ", ".", "_", "-", "’", "'", "$"]:
        repl = repl.replace(ch, "")

    return (
        repl.replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("ç", "c")
    )


def pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """
    Essaie de trouver une colonne parmi plusieurs alias possibles.
    Exemple: ["serial","no serie","no. serie","no_serie"] doit matcher aussi "$No. serie$".
    """
    norm_map = {norm(c): c for c in df.columns}

    # 1er passage : match exact sur norm()
    for cand in candidates:
        key = norm(cand)
        if key in norm_map:
            return norm_map[key]

    # 2ème passage : match "contient" (plus flou)
    for c in df.columns:
        nc = norm(c)
        for cand in candidates:
            if norm(cand) in nc:
                return c

    return None


def norm_serial_inplace(df: pd.DataFrame, col: str):
    df[col] = df[col].astype(str).str.strip().str.upper()


def main():
    if not os.path.exists(LEDGER):
        raise SystemExit("Run clean_item_ledger first.")

    led = pd.read_parquet(LEDGER)

    serial_col = pick_col(led, ["serial", "no serie", "no. serie", "no_serie"])
    contract_col = pick_col(led, ["contract_no", "no contrat", "no. contrat", "no_contrat"])
    company_col = pick_col(led, ["company", "societe"])

    if serial_col is None:
        # 🔥 Au lieu de crasher, on crée des fichiers vides mais structurés
        print(
            f"[build_contracted_devices] WARN: Colonne 'serial' introuvable. "
            f"Colonnes: {list(led.columns)}"
        )
        empty_contracted = pd.DataFrame(columns=["serial", "company", "contract_no"])
        empty_contracted.to_parquet(OUT_CONTRACTED, index=False)

        empty_rel = pd.DataFrame(columns=["serial", "company", "relation_type"])
        empty_rel.to_parquet(OUT_REL, index=False)

        print(
            f"[build_contracted_devices] Fichiers vides créés:\n"
            f"  - {OUT_CONTRACTED}\n"
            f"  - {OUT_REL}"
        )
        return

    # sérial dispo ?
    ser_ok = led[serial_col].notna() & (~led[serial_col].astype(str).str.strip().eq(""))

    # contrat dispo ?
    if contract_col is not None:
        con_ok = led[contract_col].notna() & (~led[contract_col].astype(str).str.strip().eq(""))
    else:
        con_ok = pd.Series(False, index=led.index)

    # 1) contracted_devices
    cols = [serial_col]
    if company_col:
        cols.append(company_col)
    if contract_col:
        cols.append(contract_col)

    contracted = (
        led.loc[ser_ok & con_ok, cols]
        .drop_duplicates()
        .rename(
            columns={
                serial_col: "serial",
                **({company_col: "company"} if company_col else {}),
                **({contract_col: "contract_no"} if contract_col else {}),
            }
        )
    )

    if "company" not in contracted.columns:
        contracted["company"] = pd.NA
    if "contract_no" not in contracted.columns:
        contracted["contract_no"] = pd.NA

    norm_serial_inplace(contracted, "serial")
    contracted.to_parquet(OUT_CONTRACTED, index=False)
    print(f"Saved {OUT_CONTRACTED} ({len(contracted)} rows)")

    # 2) serial_relations
    rel_parts = []

    # a) sous contrat ADEX
    rel_parts.append(
        pd.DataFrame(
            {
                "serial": led.loc[ser_ok & con_ok, serial_col],
                "company": led.loc[ser_ok & con_ok, company_col]
                if company_col
                else pd.NA,
                "relation_type": "contrat_adex",
            }
        )
    )

    # b) machine avec série mais sans contrat
    rel_parts.append(
        pd.DataFrame(
            {
                "serial": led.loc[ser_ok & (~con_ok), serial_col],
                "company": led.loc[ser_ok & (~con_ok), company_col]
                if company_col
                else pd.NA,
                "relation_type": "machine_hors_contrat",
            }
        )
    )

    # c) lignes sans n° de série (commande libre)
    rel_parts.append(
        pd.DataFrame(
            {
                "serial": led.loc[(~ser_ok), serial_col],
                "company": led.loc[(~ser_ok), company_col] if company_col else pd.NA,
                "relation_type": "commande_libre",
            }
        )
    )

    rel_df = pd.concat(rel_parts, ignore_index=True)
    rel_df = rel_df.dropna(subset=["serial"]).copy()
    norm_serial_inplace(rel_df, "serial")

    # dédoublonnage avec priorité contrat > hors contrat > libre
    cat_rank = {"contrat_adex": 3, "machine_hors_contrat": 2, "commande_libre": 1}
    rel_df["rank"] = rel_df["relation_type"].map(cat_rank).fillna(0)

    rel_top = (
        rel_df.sort_values(["serial", "company", "rank"], ascending=[True, True, False])
        .drop_duplicates(subset=["serial", "company"], keep="first")
        .drop(columns=["rank"])
    )

    rel_top.to_parquet(OUT_REL, index=False)
    print(f"Saved {OUT_REL} ({len(rel_top)} rows)")


if __name__ == "__main__":
    main()
