import os
import requests
from dotenv import load_dotenv
import msal

# Charger le .env
load_dotenv()

TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

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

    print("✅ Token obtenu avec succès")
    return result["access_token"]

    
def graph_get(url, token, params=None):
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers, params=params)
    print(f"Requête Graph → {r.status_code}")
    if r.status_code != 200:
        print("Réponse d'erreur :")
        print(r.text)
        raise SystemExit(1)
    return r.json()


def main():
    token = get_access_token()

    # 1er test : appeler /me/sites ou /sites
    url = f"{GRAPH_BASE_URL}/sites?top=5"
    data = graph_get(url, token)

    print("✅ Appel Graph réussi, sites trouvés :")
    for site in data.get("value", []):
        print(f"- {site.get('name')} | {site.get('webUrl')}")


if __name__ == "__main__":
    main()
