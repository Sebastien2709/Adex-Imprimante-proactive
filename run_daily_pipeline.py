from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(label: str, args: list[str]) -> None:
    """
    Exécute une commande Python (module ou script) avec logs propres.
    args est la liste des arguments après 'python' (ex: ['-m', 'src.data.ingest']).
    """
    print("\n" + "=" * 80)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] ▶ {label}")
    print("=" * 80)

    cmd = [sys.executable] + args
    print(f"Commande : {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"❌ [{label}] a échoué avec code {result.returncode}")
        # Si tu veux que le pipeline s'arrête dès qu'un step plante :
        sys.exit(result.returncode)
    else:
        print(f"✅ [{label}] OK")


def main():
    # 0) Optionnel : scripts SharePoint / XRM (à activer quand le download sera calé)
    # run_step("Check env", ["check_env.py"])
    # run_step("Lister fichiers XRM", ["list_xrm_files.py"])
    # run_step("Télécharger XRM filtré", ["download_xrm_download_xrm_filtered.py"])

    # 1) Ingestion & nettoyage des données
    run_step("Ingest données brutes", ["-m", "src.data.ingest"])
    run_step("Nettoyage item_ledger", ["-m", "src.data.clean_item_ledger"])
    run_step("Nettoyage kpax_consumables", ["-m", "src.data.clean_kpax_consumables"])
    run_step("Nettoyage meters", ["-m", "src.data.clean_meters"])
    run_step("Construction contracted_devices", ["-m", "src.data.build_contracted_devices"])

    # 2) Features V1 (resets, pentes, forecasts)
    run_step("Détection resets consommables", ["-m", "src.features.detect_resets"])
    run_step("Calcul des pentes / forecasts", ["-m", "src.features.compute_slopes"])
    # Optionnel : génération recommands V1
    # run_step("Replenishments V1 (règles simples)", ["-m", "src.features.generate_replenishments"])

    # 3) Modèle V2.1 – prédictions quotidiennes
    # (Entraînement V2.1 peut être fait moins souvent, ex: hebdo)
    # run_step("Build dataset V2.1 (backtest)", ["-m", "src.model.build_dataset_v21"])
    # run_step("Train XGB V2.1", ["-m", "src.model.train_xgb_v21"])

    run_step("Prédictions offsets V2.1", ["-m", "src.model.predict_xgb_v21"])

    # 4) Enrichissement métier + export
    run_step("Enrichissement reco V2.1 (devices/contrats)", ["-m", "src.model.enrich_replenishments_v21"])
    run_step("Export fichier business final", ["-m", "src.model.export_recos_business"])

    print("\n" + "=" * 80)
    print("🎉 Pipeline quotidien terminé")
    print("➡ Fichier final : data/outputs/recommandations_toners_latest.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
