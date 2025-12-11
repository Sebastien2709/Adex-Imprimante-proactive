# src/data/inspect_item_ledger.py
import os, pandas as pd

PROC = "data/processed"
LEDGER = os.path.join(PROC, "item_ledger.parquet")

def main():
    df = pd.read_parquet(LEDGER)
    print("Colonnes:", list(df.columns))

    def norm(s): 
        return (str(s).lower()
                .replace(" ", "").replace(".", "").replace("_", "")
                .replace("é","e").replace("è","e").replace("ê","e")
                .replace("à","a").replace("ç","c"))

    cols = {norm(c): c for c in df.columns}
    serial_col   = cols.get("noserie")   or cols.get("serial")
    contract_col = cols.get("nocontrat") or cols.get("contractno") or cols.get("contract")

    print("→ serial_col:", serial_col, "| contract_col:", contract_col)

    if serial_col:
        ser_notna  = df[serial_col].notna()
        ser_notemp = (~df[serial_col].astype(str).str.strip().eq(""))
        print("serial non-nuls:", ser_notna.sum())
        print("serial non vides:", (ser_notna & ser_notemp).sum())

    if contract_col:
        con_notna  = df[contract_col].notna()
        con_notemp = (~df[contract_col].astype(str).str.strip().eq(""))
        print("contract non-nuls:", con_notna.sum())
        print("contract non vides:", (con_notna & con_notemp).sum())

    if serial_col and contract_col:
        ser_ok = df[serial_col].notna() & (~df[serial_col].astype(str).str.strip().eq(""))
        con_ok = df[contract_col].notna() & (~df[contract_col].astype(str).str.strip().eq(""))
        mask = ser_ok & con_ok
        both = df.loc[mask, [serial_col, contract_col, "relation_type"]].head(5)
        print("Lignes avec serial **et** contrat:", mask.sum())
        print(both)

    # Bonus: qu'est-ce qu'on a dans relation_type ?
    if "relation_type" in df.columns:
        print("relation_type counts:")
        print(df["relation_type"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
