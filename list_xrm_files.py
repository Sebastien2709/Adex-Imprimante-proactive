import os
from urllib.parse import urlparse

import msal
import requests
from dotenv import load_dotenv

load_dotenv()

TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

SITE_URL = os.getenv("SHAREPOINT_SITE_URL")
LIBRARY_NAME = os.getenv("SHAREPOINT_LIBRARY_NAME", "Documents partagés")
FOLDER_PATH = os.getenv("SHAREPOINT_FOLDER_PATH", "/")

AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = ["https://graph.microsoft.com/.default"]
GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"


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
        print("❌ Impossible d'obtenir un token")
        print(result)
        raise SystemExit(1)

    return result["access_token"]


def graph_get(url, token, params=None):
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers, params=params)
    print(f"Graph GET {url} → {r.status_code}")
    r.raise_for_status()
    return r.json()


def parse_site_host_and_path(site_url: str):
    parsed = urlparse(site_url)
    host = parsed.netloc  # ex: adexgroupfr.sharepoint.com
    parts = [p for p in parsed.path.split("/") if p]
    # ex: ["sites", "XRM"]
    if len(parts) >= 2 and parts[0].lower() == "sites":
        site_path = parts[1]  # "XRM"
    else:
        site_path = parts[-1]
    return host, site_path


def get_site_id(token: str, site_url: str) -> str:
    host, site_path = parse_site_host_and_path(site_url)
    url = f"{GRAPH_BASE_URL}/sites/{host}:/sites/{site_path}"
    data = graph_get(url, token)
    return data["id"]


def get_drive_id(token: str, site_id: str, library_name: str) -> str:
    url = f"{GRAPH_BASE_URL}/sites/{site_id}/drives"
    data = graph_get(url, token)
    for drive in data.get("value", []):
        print(f"Drive trouvé : {drive['name']}")
        if drive["name"].lower() == library_name.lower():
            return drive["id"]
    raise RuntimeError(f"Drive '{library_name}' non trouvé sur le site {site_id}")


def list_files_in_folder(token: str, drive_id: str, folder_path: str):
    # Ex : root:/Adex_Import_Directory/IA_bdd:/children
    # On enlève les éventuels / de début
    folder_path = folder_path.lstrip("/")
    url = f"{GRAPH_BASE_URL}/drives/{drive_id}/root:/{folder_path}:/children"
    data = graph_get(url, token)
    return [item for item in data.get("value", []) if "file" in item]


def main():
    token = get_access_token()
    print("✅ Token OK")

    site_id = get_site_id(token, SITE_URL)
    print(f"🌐 Site ID = {site_id}")

    drive_id = get_drive_id(token, site_id, LIBRARY_NAME)
    print(f"📁 Drive ID = {drive_id}")

    files = list_files_in_folder(token, drive_id, FOLDER_PATH)
    print(f"\n📄 Fichiers trouvés dans {FOLDER_PATH} :")
    if not files:
        print("  (aucun fichier trouvé)")
    else:
        for f in files:
            print(f"  - {f['name']}")

if __name__ == "__main__":
    main()
