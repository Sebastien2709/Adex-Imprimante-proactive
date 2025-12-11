import os
from pathlib import Path
from urllib.parse import urlparse

import msal
import requests
from dotenv import load_dotenv

load_dotenv()

TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

SITE_URL = os.getenv("SHAREPOINT_SITE_URL")
LIBRARY_NAME = os.getenv("SHAREPOINT_LIBRARY_NAME")
FOLDER_PATH = os.getenv("SHAREPOINT_FOLDER_PATH")

LOCAL_DATA_DIR = Path(os.getenv("LOCAL_DATA_DIR", "./data/raw/xrm"))

AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = ["https://graph.microsoft.com/.default"]
GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"

# --- PATTERNS RECHERCHÉS ---
TARGET_PATTERNS = [
    "AI Export_Kpax_consumables_ADEXGROUP",
    "AI_Export_ItemLedgEntries_ADEXGROUP",
    "AI_Export_SalesPagesMeters_ADEXGROUP"
]


def get_access_token():
    app = msal.ConfidentialClientApplication(
        CLIENT_ID,
        authority=AUTHORITY,
        client_credential=CLIENT_SECRET,
    )
    result = app.acquire_token_silent(SCOPES, account=None)
    if not result:
        result = app.acquire_token_for_client(scopes=SCOPES)

    if "access_token" not in result:
        raise RuntimeError(f"Impossible d'obtenir un token: {result}")
    return result["access_token"]


def graph_get(url, token, params=None):
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers, params=params)
    print(f"GET {url} → {r.status_code}")
    r.raise_for_status()
    return r.json()


def parse_site_host_and_path(site_url):
    parsed = urlparse(site_url)
    host = parsed.netloc
    parts = [p for p in parsed.path.split("/") if p]

    if len(parts) >= 2 and parts[0].lower() == "sites":
        site_path = parts[1]
    else:
        site_path = parts[-1]

    return host, site_path


def get_site_id(token, site_url):
    host, site_path = parse_site_host_and_path(site_url)
    url = f"{GRAPH_BASE_URL}/sites/{host}:/sites/{site_path}"
    data = graph_get(url, token)
    return data["id"]


def get_drive_id(token, site_id, library_name):
    url = f"{GRAPH_BASE_URL}/sites/{site_id}/drives"
    data = graph_get(url, token)
    for drive in data.get("value", []):
        if drive["name"].lower() == library_name.lower():
            return drive["id"]
    raise RuntimeError(f"Drive '{library_name}' introuvable.")


def list_files_in_folder(token, drive_id, folder_path):
    folder_path = folder_path.lstrip("/")
    url = f"{GRAPH_BASE_URL}/drives/{drive_id}/root:/{folder_path}:/children"
    data = graph_get(url, token)
    return [item for item in data.get("value", []) if "file" in item]


def download_file(token, drive_id, item_id, dest_path):
    url = f"{GRAPH_BASE_URL}/drives/{drive_id}/items/{item_id}/content"
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers, stream=True)
    r.raise_for_status()

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


def filename_matches(name):
    """Retourne True si le fichier correspond à un des patterns."""
    return any(pattern in name for pattern in TARGET_PATTERNS)


def main():
    print("🔌 Connexion...")
    token = get_access_token()
    print("✅ Token OK")

    site_id = get_site_id(token, SITE_URL)
    print(f"🌐 Site ID = {site_id}")

    drive_id = get_drive_id(token, site_id, LIBRARY_NAME)
    print(f"📁 Drive ID = {drive_id}")

    remote_files = list_files_in_folder(token, drive_id, FOLDER_PATH)
    print(f"📄 {len(remote_files)} fichiers trouvés sur SharePoint")

    local_files = {f.name for f in LOCAL_DATA_DIR.glob("*.txt")}

    print("\n🎯 Fichiers à traiter :")
    for item in remote_files:
        name = item["name"]

        # Filtrer les .txt
        if not name.lower().endswith(".txt"):
            continue

        # Filtrer selon les patterns
        if not filename_matches(name):
            continue

        # Filtrer ceux déjà présents en local
        if name in local_files:
            print(f"   ⏭️  Déjà présent : {name}")
            continue

        # Téléchargement
        print(f"⬇️  Téléchargement : {name}")
        dest = LOCAL_DATA_DIR / name
        download_file(token, drive_id, item["id"], dest)

    print("\n✅ Terminé.")


if __name__ == "__main__":
    main()
