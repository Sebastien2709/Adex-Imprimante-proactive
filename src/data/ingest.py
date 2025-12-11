from pathlib import Path
import pandas as pd
import re

RAW = Path("data/raw")
INTERIM = Path("data/interim")
INTERIM.mkdir(parents=True, exist_ok=True)


def find_latest(pattern: str) -> Path:
    """
    Cherche le dernier fichier qui match le pattern (glob)
    dans data/raw. Exemple: 'AI_Export_ItemLedgEntries_ADEXGROUP*.txt'
    """
    candidates = sorted(RAW.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"Aucun fichier trouvé pour le pattern: {pattern}")
    # on prend le plus récent par date de modif
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"[ingest] latest for {pattern}: {latest.name}")
    return latest


def load_kpax_consumables_adex() -> pd.DataFrame:
    path = find_latest("AI*Export*Kpax*consumables*ADEXGROUP*")
    df = pd.read_csv(path, sep="|", dtype=str, encoding="utf-8", engine="python")
    return df

def load_item_ledger_adex() -> pd.DataFrame:
    path = find_latest("AI*Export*Item*Ledg*Entries*ADEXGROUP*")
    df = pd.read_csv(path, sep="|", dtype=str, encoding="utf-8", engine="python")
    return df

def load_meters_adex() -> pd.DataFrame:
    path = find_latest("AI*Export*SalesPages*Meters*ADEXGROUP*")
    df = pd.read_csv(path, sep="|", dtype=str, encoding="utf-8", engine="python")
    return df



def main():
    # 1) Item Ledger
    item_ledger = load_item_ledger_adex()
    out_item = INTERIM / "item_ledger.csv"
    item_ledger.to_csv(out_item, index=False)
    print(f"Saved: {out_item}")

    # 2) KPAX consumables
    kpax = load_kpax_consumables_adex()
    out_kpax = INTERIM / "kpax_consumables.csv"
    kpax.to_csv(out_kpax, index=False)
    print(f"Saved: {out_kpax}")

    # 3) Meters
    meters = load_meters_adex()
    out_meters = INTERIM / "meters.csv"
    meters.to_csv(out_meters, index=False)
    print(f"Saved: {out_meters}")


if __name__ == "__main__":
    main()
