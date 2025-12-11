import os
from dotenv import load_dotenv

load_dotenv()

print("SHAREPOINT_SITE_URL =", os.getenv("SHAREPOINT_SITE_URL"))
print("SHAREPOINT_LIBRARY_NAME =", os.getenv("SHAREPOINT_LIBRARY_NAME"))
print("SHAREPOINT_FOLDER_PATH =", os.getenv("SHAREPOINT_FOLDER_PATH"))
