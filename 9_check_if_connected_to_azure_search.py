from azure.search.documents import SearchClient
import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential

# Load environment variables
load_dotenv()
SEARCH_SERVICE_NAME = os.getenv("AZURE_SEARCH_SERVICE_NAME")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Create a search client
search_client = SearchClient(
    endpoint=f"https://{SEARCH_SERVICE_NAME}.search.windows.net",
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_API_KEY)
)

# Count the number of indexed chunks (pages)
count = search_client.get_document_count()
print(f"📄 Total indexed chunks (sections of the document): {count}")
