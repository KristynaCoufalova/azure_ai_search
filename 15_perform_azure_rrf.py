import os
import requests
import json
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()
SEARCH_SERVICE_NAME = os.getenv("AZURE_SEARCH_SERVICE_NAME")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = "text-embedding-3-large"

# Define query text
dotaz = "Jaké legislativní předpisy Evropské unie se vztahují na veřejnou podporu v rámci OP JAK?"
pocet_vysledku = 10  # Number of top-ranked documents

# Generate an embedding for the query using OpenAI
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-01"
)
response = openai_client.embeddings.create(
    model=AZURE_OPENAI_DEPLOYMENT,
    input=dotaz
)
embedding_vector = response.data[0].embedding
print(f" Generated OpenAI embedding vector with {len(embedding_vector)} dimensions")

# 🔹 Define Azure AI Search Hybrid Search request (BM25 + Vector Search)
body = {
    "search": dotaz,  # BM25 query
    "queryType": "simple",
    "searchFields": "chunk",
    "select": "chunk_id,parent_id,chunk,title",  #  Removed `@search.score` and `@search.rerankerScore`
    "top": pocet_vysledku,
    "vectorQueries": [
        {
            "kind": "vector",
            "vector": embedding_vector,
            "fields": "vector",
            "k": 50  # Number of neighbors considered in vector search
        }
    ]
}

#  Define the endpoint URL (Azure AI Search Hybrid Search API version `2024-07-01`)
url = f"https://{SEARCH_SERVICE_NAME}.search.windows.net/indexes/{INDEX_NAME}/docs/search?api-version=2024-07-01"

# Define the headers
headers = {
    "Content-Type": "application/json",
    "api-key": SEARCH_API_KEY
}

# Send the Hybrid Search request (BM25 + Vector + RRF)
response = requests.post(url, headers=headers, json=body)

# 🔍 Check if request was successful
if response.status_code == 200:
    results = response.json().get("value", [])
    print(f"✅ Found {len(results)} Hybrid (BM25 + Vector + RRF) results:\n")

    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"ID nadřazeného dokumentu: {result.get('parent_id', 'N/A')}")
        print(f"ID úryvku: {result.get('chunk_id', 'N/A')}")
        print(f"Název dokumentu: {result.get('title', 'N/A')}")
        print(f"Skóre shody (BM25 + Vector): {result.get('@search.score', 'N/A')}")  # ✅ Now retrieved from metadata
        print(f"RRF Skóre (rerankerScore): {result.get('@search.rerankerScore', 'N/A')}")  # ✅ Now retrieved from metadata
        print(f"Obsah: {result.get('chunk', 'N/A')[:200]}..." if len(result.get('chunk', '')) > 200 else result.get('chunk', 'N/A'))
        print("-" * 80)
else:
    print(f" Error: {response.status_code}")
    print(response.text)

# NAs are because semantic seach is not allowed (paywall)
