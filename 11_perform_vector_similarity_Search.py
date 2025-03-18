#11_perform_vector_similarity_Search.py
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

# Create Azure OpenAI client
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-01"  # Latest Azure OpenAI API version
)

# Define query text
dotaz = "Jaké legislativní předpisy Evropské unie se vztahují na veřejnou podporu v rámci OP JAK?"
#dotaz = "Jakým způsobem postupovat u veřejných vysokých škol a v.v.i. při určování kategorie malého/středního/velkého podniku?"
pocet_vysledku = 3  # Number of nearest neighbors to retrieve

# Generate an embedding for the query
response = openai_client.embeddings.create(
    model=AZURE_OPENAI_DEPLOYMENT,
    input=dotaz
)

# Extract the embedding vector
embedding_vector = response.data[0].embedding

# Define the search request body
body = {
    "select": "chunk_id,parent_id,chunk,title",
    "top": pocet_vysledku,
    "vectorQueries": [
        {
            "kind": "vector", 
            "vector": embedding_vector,
            "fields": "vector", 
            "k": pocet_vysledku
        }
    ]
}

# Define the endpoint URL
url = f"https://{SEARCH_SERVICE_NAME}.search.windows.net/indexes/{INDEX_NAME}/docs/search?api-version=2023-11-01"

# Define the headers
headers = {
    "Content-Type": "application/json",
    "api-key": SEARCH_API_KEY
}

# Send the search request
response = requests.post(url, headers=headers, json=body)

# Check if request was successful
if response.status_code == 200:
    results = response.json().get("value", [])
    print(f"Found {len(results)} results:\n")
    
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