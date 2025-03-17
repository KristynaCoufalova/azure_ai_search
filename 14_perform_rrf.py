# 13_rrf.py
#reciprocical rank fusion
import os
import requests
import json
from collections import defaultdict
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

# Create OpenAI client
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-01"  # Latest Azure OpenAI API version
)

# Define query (Czech)
dotaz = "Jaké legislativní předpisy Evropské unie se vztahují na veřejnou podporu v rámci OP JAK?"
pocet_vysledku = 20  # Number of results

# Generate vector embedding for query using OpenAI
response = openai_client.embeddings.create(
    model=AZURE_OPENAI_DEPLOYMENT,
    input=dotaz
)
embedding_vector = response.data[0].embedding
print(f" Generated OpenAI embedding vector with {len(embedding_vector)} dimensions")

#  Define Azure AI Search endpoint
url = f"https://{SEARCH_SERVICE_NAME}.search.windows.net/indexes/{INDEX_NAME}/docs/search?api-version=2023-11-01"
headers = {
    "Content-Type": "application/json",
    "api-key": SEARCH_API_KEY
}

#  Step 1: Perform BM25 Search (Only in "chunk" field)
bm25_body = {
    "search": dotaz,
    "queryType": "simple",
    "searchFields": "chunk",
    "select": "chunk_id,parent_id,chunk,title",
    "top": 20
}
bm25_response = requests.post(url, headers=headers, json=bm25_body)
bm25_results = bm25_response.json().get("value", []) if bm25_response.status_code == 200 else []

#  Step 2: Perform Vector Search (Use "vector" field)
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
print(f" Vector Search returned {len(vector_results)} results")

#  Debug: Print Raw BM25 and Vector Search Results Before Fusion
print("\n Debug: Raw BM25 Results (Before RRF)")
for i, doc in enumerate(bm25_results[:5], 1):
    print(f"{i}. {doc.get('chunk_id', 'N/A')} - {doc.get('title', 'N/A')}")
    print(f"   {doc.get('chunk', '⚠️ No chunk found')[:250]}...\n")

print("\n Debug: Raw Vector Search Results (Before RRF)")
for i, doc in enumerate(vector_results[:5], 1):
    print(f"{i}. {doc.get('chunk_id', 'N/A')} - {doc.get('title', 'N/A')}")
    print(f"   {doc.get('chunk', '⚠️ No chunk found')[:250]}...\n")

#  Step 3: Apply Reciprocal Rank Fusion (RRF)
def reciprocal_rank_fusion(bm25_results, vector_results, c=60):
    """Merges two ranked result lists using Reciprocal Rank Fusion (RRF)."""
    ranks = defaultdict(lambda: [None, None, None, None])  # {doc_id: [bm25_rank, vector_rank, text, title]}

    # Assign ranks for BM25
    for rank, doc in enumerate(bm25_results, start=1):
        ranks[doc["chunk_id"]][0] = rank
        ranks[doc["chunk_id"]][2] = doc.get("chunk", "")
        ranks[doc["chunk_id"]][3] = doc.get("title", "")

    # Assign ranks for Vector Search
    for rank, doc in enumerate(vector_results, start=1):
        ranks[doc["chunk_id"]][1] = rank
        ranks[doc["chunk_id"]][2] = doc.get("chunk", "")
        ranks[doc["chunk_id"]][3] = doc.get("title", "")

    # Compute RRF scores
    rrf_scores = {}
    for doc_id, (bm25_rank, vector_rank, text, title) in ranks.items():
        score = 0
        if bm25_rank is not None:
            score += 1 / (bm25_rank + c)
        if vector_rank is not None:
            score += 1 / (vector_rank + c)

        rrf_scores[doc_id] = {
            "score": score,
            "text": text,
            "title": title
        }

    # Sort by RRF score
    reranked_docs = sorted(rrf_scores.items(), key=lambda x: x[1]["score"], reverse=True)

    return [
        {"doc_id": doc_id, "score": result["score"], "text": result["text"], "title": result["title"]}
        for doc_id, result in reranked_docs
    ]

#  Filter out empty or unhelpful results BEFORE RRF
bm25_filtered = [doc for doc in bm25_results if doc.get("chunk") and len(doc["chunk"]) > 100]
vector_filtered = [doc for doc in vector_results if doc.get("chunk") and len(doc["chunk"]) > 100]
print(f"\n After filtering: {len(bm25_filtered)} BM25 results, {len(vector_filtered)} vector results")

fused_results = reciprocal_rank_fusion(bm25_filtered, vector_filtered)
print(f" Fused results count: {len(fused_results)}")

# 🔹 Step 4: Display Final Results
print(f"\n Query: {dotaz}")
print(f" Final Re-ranked Results (RRF only):\n")
for i, result in enumerate(fused_results, 1):
    chunk_id = result.get("doc_id", "N/A")
    title = result.get("title", "N/A")
    text = result.get("text", "")
    
    # Display chunk_id and title along with the text
    print(f"{i}. [{chunk_id}] - {title}")
    print(f"   {text[:500]}..." if len(text) > 500 else text)
    print()  # Add a blank line between results
