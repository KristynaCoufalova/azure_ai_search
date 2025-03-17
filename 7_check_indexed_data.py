import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SEARCH_SERVICE_NAME = os.getenv("AZURE_SEARCH_SERVICE_NAME")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Define the endpoint URL for querying the index
url = f"https://{SEARCH_SERVICE_NAME}.search.windows.net/indexes/{INDEX_NAME}/docs/search?api-version=2024-02-01"

# Define the headers
headers = {
    "Content-Type": "application/json",
    "api-key": SEARCH_API_KEY
}

# Define the search request body to retrieve all indexed documents
body = {
    "search": "*",  # Wildcard search to fetch all documents
    "select": "chunk_id,parent_id,chunk,title",
    "top": 5  # Fetch up to 5 documents
}

# Send the search request
response = requests.post(url, headers=headers, json=body)

# Check if request was successful
if response.status_code == 200:
    results = response.json().get("value", [])
    if results:
        print(f"Found {len(results)} indexed documents:\n")
        for i, result in enumerate(results, 1):
            print(f"Document {i}:")
            print(f"ID nadřazeného dokumentu: {result.get('parent_id', 'N/A')}")
            print(f"ID úryvku: {result.get('chunk_id', 'N/A')}")
            print(f"Název dokumentu: {result.get('title', 'N/A')}")
            print(f"Obsah: {result.get('chunk', 'N/A')[:200]}..." if len(result.get('chunk', '')) > 200 else result.get('chunk', 'N/A'))
            print("-" * 80)
    else:
        print("No documents found. Your index might be empty.")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
