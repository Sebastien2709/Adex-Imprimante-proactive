import gzip
import io
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client
import pandas as pd

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET       = "toner-data"
CHUNK_ROWS   = 500_000   # lignes par chunk pour les gros CSV
MAX_MB       = 40        # limite conservative (Supabase free = 50 MB)

BASE_DIR = Path(__file__).resolve().parents[1]

FILES_TO_UPLOAD = [
    (BASE_DIR / "data" / "outputs"   / "recommandations_toners_latest.csv", "recommandations_toners_latest.csv"),
    (BASE_DIR / "data" / "processed" / "kpax_last_states.csv",              "kpax_last_states.csv"),
    (BASE_DIR / "data" / "processed" / "kpax_history_light.csv",            "kpax_history_light.csv"),
    (BASE_DIR / "data" / "processed" / "contract_status.parquet",           "contract_status.parquet"),
]


def _upload_raw(client, data: bytes, remote_name: str) -> bool:
    """Upload brut d'un bytes vers le bucket."""
    try:
        client.storage.from_(BUCKET).remove([remote_name])
    except Exception:
        pass
    client.storage.from_(BUCKET).upload(
        path=remote_name,
        file=data,
        file_options={"content-type": "application/octet-stream", "upsert": "true"},
    )
    return True


def upload_file(client, local_path: Path, remote_name: str) -> bool:
    if not local_path.exists():
        print(f"[SUPABASE] ⚠️  Fichier introuvable : {local_path}")
        return False

    size_mb = local_path.stat().st_size / 1_000_000
    print(f"[SUPABASE] Upload {remote_name} ({size_mb:.1f} MB)...")

    # Gros CSV → découpage en chunks gzip
    if remote_name.endswith(".csv") and size_mb > MAX_MB:
        print(f"[SUPABASE] Fichier > {MAX_MB} MB → découpage en chunks gzip...")
        df      = pd.read_csv(local_path)
        total   = len(df)
        n_chunks = (total // CHUNK_ROWS) + (1 if total % CHUNK_ROWS else 0)

        # Supprimer les anciens chunks
        old_chunks = [f"{remote_name}.chunk{i}.gz" for i in range(20)]
        try:
            client.storage.from_(BUCKET).remove(old_chunks)
        except Exception:
            pass

        for i in range(n_chunks):
            chunk_df = df.iloc[i * CHUNK_ROWS : (i + 1) * CHUNK_ROWS]

            # CSV → bytes
            csv_buf = io.StringIO()
            chunk_df.to_csv(csv_buf, index=False)
            csv_bytes = csv_buf.getvalue().encode("utf-8")

            # gzip en mémoire
            gz_buf = io.BytesIO()
            with gzip.GzipFile(fileobj=gz_buf, mode="wb") as gz:
                gz.write(csv_bytes)
            gz_data    = gz_buf.getvalue()
            chunk_name = f"{remote_name}.chunk{i}.gz"
            chunk_mb   = len(gz_data) / 1_000_000
            print(f"[SUPABASE]   chunk {i+1}/{n_chunks} → {chunk_name} ({chunk_mb:.1f} MB)")
            _upload_raw(client, gz_data, chunk_name)

        # Métadonnées
        meta = json.dumps({"chunks": n_chunks, "rows": total}).encode()
        _upload_raw(client, meta, f"{remote_name}.meta.json")
        print(f"[SUPABASE] ✅ {remote_name} uploadé en {n_chunks} chunks")
        return True

    # Fichier normal (<40 MB)
    with open(local_path, "rb") as f:
        data = f.read()
    _upload_raw(client, data, remote_name)
    print(f"[SUPABASE] ✅ {remote_name} uploadé")
    return True


def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise SystemExit(
            "[SUPABASE] ERREUR: SUPABASE_URL et SUPABASE_KEY manquants dans .env"
        )

    client  = create_client(SUPABASE_URL, SUPABASE_KEY)
    print(f"[SUPABASE] Connecté à {SUPABASE_URL}")
    print(f"[SUPABASE] Bucket : {BUCKET}")

    success = 0
    for local_path, remote_name in FILES_TO_UPLOAD:
        if upload_file(client, local_path, remote_name):
            success += 1

    print(f"\n[SUPABASE] {success}/{len(FILES_TO_UPLOAD)} fichiers uploadés ✅")


if __name__ == "__main__":
    main()