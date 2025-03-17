from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SEARCH_SERVICE_NAME = os.getenv("AZURE_SEARCH_SERVICE_NAME")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Initialize SearchIndexClient
index_client = SearchIndexClient(
    endpoint=f"https://{SEARCH_SERVICE_NAME}.search.windows.net",
    credential=AzureKeyCredential(SEARCH_API_KEY)
)

# Retrieve and print index details
index_info = index_client.get_index(INDEX_NAME)
print(index_info)


# Print available fields
print("📄 Index Fields:")
for field in index_info.fields:
    print(f"- {field.name} ({field.type})")