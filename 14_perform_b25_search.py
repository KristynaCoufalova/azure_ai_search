import os
import requests
import json
from dotenv import load_dotenv

# 🔹 Load environment variables
load_dotenv()
SEARCH_SERVICE_NAME = os.getenv("AZURE_SEARCH_SERVICE_NAME")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Define query text
dotaz = "Jaké legislativní předpisy Evropské unie se vztahují na veřejnou podporu v rámci OP JAK?"
pocet_vysledku = 3  # Number of top results

# Define the search request body for BM25
bm25_body = {
    "search": dotaz,
    "queryType": "simple",
    "searchFields": "chunk",
    "select": "chunk_id,parent_id,chunk,title",
    "top": pocet_vysledku
}

# Define the endpoint URL
url = f"https://{SEARCH_SERVICE_NAME}.search.windows.net/indexes/{INDEX_NAME}/docs/search?api-version=2023-11-01"

# Define the headers
headers = {
    "Content-Type": "application/json",
    "api-key": SEARCH_API_KEY
}

# Send the BM25 search request
response = requests.post(url, headers=headers, json=bm25_body)

# Check if request was successful
if response.status_code == 200:
    results = response.json().get("value", [])
    print(f"Found {len(results)} BM25 results:\n")

    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"ID nadřazeného dokumentu: {result.get('parent_id', 'N/A')}")
        print(f"ID úryvku: {result.get('chunk_id', 'N/A')}")
        print(f"Název dokumentu: {result.get('title', 'N/A')}")
        print(f"Skóre shody: {result.get('@search.score', 'N/A')}")
        print(f"Obsah: {result.get('chunk', 'N/A')[:200]}..." if len(result.get('chunk', '')) > 200 else result.get('chunk', 'N/A'))
        print("-" * 80)
else:
    print(f"Error: {response.status_code}")
    print(response.text)
