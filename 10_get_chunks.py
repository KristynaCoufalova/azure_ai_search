from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
import json
import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential

# Load environment variables
load_dotenv()
SEARCH_SERVICE_NAME = os.getenv("AZURE_SEARCH_SERVICE_NAME")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Step 1: Connect to Azure AI Search
index_client = SearchIndexClient(
    endpoint=f"https://{SEARCH_SERVICE_NAME}.search.windows.net",
    credential=AzureKeyCredential(SEARCH_API_KEY)
)

search_client = SearchClient(
    endpoint=f"https://{SEARCH_SERVICE_NAME}.search.windows.net",
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_API_KEY)
)

# Step 2: Fetch Available Index Fields
index_info = index_client.get_index(INDEX_NAME)
available_fields = {field.name for field in index_info.fields}

# Step 3: Define Fields for Selection (Only Select Existing Ones)
desired_fields = ["id", "chunk_id", "title", "chunk", "page_number"]
valid_fields = [field for field in desired_fields if field in available_fields]

if not valid_fields:
    raise ValueError("No valid fields found in the index. Check your schema.")

print(f" Available Fields in Index: {valid_fields}")

# Step 4: Fetch Data from Azure Search
results = search_client.search("*", select=valid_fields, top=1000)

# Step 5: Convert Results to List
chunks = [doc for doc in results]

# Step 6: Save to JSON File
output_file = "indexed_chunks.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=4, ensure_ascii=False)

print(f"Exported {len(chunks)} chunks to '{output_file}'!")
