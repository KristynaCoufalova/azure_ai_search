#13_perform_cohere_reranking.py
import os
import requests
import json
from dotenv import load_dotenv
import cohere
from openai import AzureOpenAI

# Load environment variables
load_dotenv()
SEARCH_SERVICE_NAME = os.getenv("AZURE_SEARCH_SERVICE_NAME")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = "text-embedding-3-large"

# Create clients
cohere_client = cohere.Client(COHERE_API_KEY)
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-01"  # Latest Azure OpenAI API version
)

# Define query (Czech)
dotaz = "Jaké legislativní předpisy Evropské unie se vztahují na veřejnou podporu v rámci OP JAK?"
pocet_vysledku = 3  # Number of results

# Generate vector embedding for query using OpenAI
response = openai_client.embeddings.create(
    model=AZURE_OPENAI_DEPLOYMENT,
    input=dotaz
)
embedding_vector = response.data[0].embedding
print(f"Generated OpenAI embedding vector with {len(embedding_vector)} dimensions")

# 🔹 Perform Vector Search Only
url = f"https://{SEARCH_SERVICE_NAME}.search.windows.net/indexes/{INDEX_NAME}/docs/search?api-version=2023-11-01"
headers = {
    "Content-Type": "application/json",
    "api-key": SEARCH_API_KEY
}

vector_body = {
    "select": "chunk_id,parent_id,chunk,title",
    "top": pocet_vysledku,
    "vectorQueries": [
        {
            "kind": "vector",
            "vector": embedding_vector,
            "fields": "vector",
            "k": 50
        }
    ]
}
vector_response = requests.post(url, headers=headers, json=vector_body)
vector_results = vector_response.json().get("value", []) if vector_response.status_code == 200 else []
print(f"Vector Search returned {len(vector_results)} results")

# Step 2: Apply Cohere Re-Ranking
# Prepare documents for Cohere Reranker
doc_objects = [doc for doc in vector_results if doc.get("chunk")]
documents = [doc["chunk"] for doc in doc_objects]

# Use Cohere Multilingual Reranker
rerank_response = cohere_client.rerank(
    model="rerank-multilingual-v2.0",  # Supports Czech
    query=dotaz,
    documents=documents
)

# Extract reranked results with their IDs
reranked_results_with_ids = []
for result in rerank_response.results:
    index = result.index
    if 0 <= index < len(doc_objects):
        reranked_results_with_ids.append(doc_objects[index])

# Step 3: Display Final Results
print(f"\n🔎 Query: {dotaz}")
print(f" Final Re-ranked Results (Cohere Reranker - Multilingual):\n")
for i, result in enumerate(reranked_results_with_ids, 1):
    chunk_id = result.get("chunk_id", "N/A")
    title = result.get("title", "N/A")
    text = result.get("chunk", "")
    print(f"{i}. [{chunk_id}] - {title}")
    print(f"   {text[:500]}..." if len(text) > 500 else text)
    print()
