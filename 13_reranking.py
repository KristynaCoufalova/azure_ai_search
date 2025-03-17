import os
import requests
import json
from collections import defaultdict
from dotenv import load_dotenv
import cohere
from openai import AzureOpenAI
from sklearn.metrics import ndcg_score

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
pocet_vysledku = 20  # Number of results

# Generate vector embedding for query using OpenAI
response = openai_client.embeddings.create(
    model=AZURE_OPENAI_DEPLOYMENT,
    input=dotaz
)
embedding_vector = response.data[0].embedding
print(f"Generated OpenAI embedding vector with {len(embedding_vector)} dimensions")

# 🔹 Define Azure AI Search endpoint
url = f"https://{SEARCH_SERVICE_NAME}.search.windows.net/indexes/{INDEX_NAME}/docs/search?api-version=2023-11-01"
headers = {
    "Content-Type": "application/json",
    "api-key": SEARCH_API_KEY
}

# Step 1: Perform BM25 Search (Only in "chunk" field)
bm25_body = {
    "search": dotaz,
    "queryType": "simple",
    "searchFields": "chunk",
    "select": "chunk_id,parent_id,chunk,title",
    "top": 20
}
bm25_response = requests.post(url, headers=headers, json=bm25_body)
bm25_results = bm25_response.json().get("value", []) if bm25_response.status_code == 200 else []

# Step 2: Perform Vector Search (Use "vector" field)
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

# Debug: Print Raw BM25 and Vector Search Results Before Fusion
print("\n Debug: Raw BM25 Results (Before RRF)")
for i, doc in enumerate(bm25_results[:5], 1):
    print(f"{i}. {doc.get('chunk_id', 'N/A')} - {doc.get('title', 'N/A')}")
    print(f"   {doc.get('chunk', '⚠️ No chunk found')[:250]}...\n")

print("\n Debug: Raw Vector Search Results (Before RRF)")
for i, doc in enumerate(vector_results[:5], 1):
    print(f"{i}. {doc.get('chunk_id', 'N/A')} - {doc.get('title', 'N/A')}")
    print(f"   {doc.get('chunk', '⚠️ No chunk found')[:250]}...\n")

# Step 3: Apply Reciprocal Rank Fusion (RRF)
def reciprocal_rank_fusion(results_bm25, results_vector, k=60):
    """Merges two ranked result lists using Reciprocal Rank Fusion (RRF)."""
    fused_scores = defaultdict(float)
    
    # Assign RRF scores from BM25
    for rank, doc in enumerate(results_bm25, start=1):
        fused_scores[doc["chunk_id"]] += 1 / (k + rank)

    # Assign RRF scores from Vector Search
    for rank, doc in enumerate(results_vector, start=1):
        fused_scores[doc["chunk_id"]] += 1 / (k + rank)

    # Sort by RRF score (higher is better)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

# Filter out empty or unhelpful results BEFORE RRF
bm25_filtered = [doc for doc in bm25_results if doc.get("chunk") and len(doc["chunk"]) > 100]
vector_filtered = [doc for doc in vector_results if doc.get("chunk") and len(doc["chunk"]) > 100]
print(f"\n After filtering: {len(bm25_filtered)} BM25 results, {len(vector_filtered)} vector results")

fused_results = reciprocal_rank_fusion(bm25_filtered, vector_filtered)
print(f" Fused results count: {len(fused_results)}")

# Step 4: Re-Rank with Cohere Multilingual Model
# Prepare documents for Cohere Reranker with their IDs
doc_objects = [
    next((doc for doc in bm25_filtered + vector_filtered if doc["chunk_id"] == doc_id), None)
    for doc_id, _ in fused_results
]
doc_objects = [doc for doc in doc_objects if doc and doc.get("chunk")]
print(f" Documents for reranking: {len(doc_objects)}")

# Create list of just text content for Cohere reranker
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
        # Add the original document object with its chunk_id to the results
        reranked_results_with_ids.append(doc_objects[index])

# Step 5: Evaluate Using NDCG (Handles Missing File)
def evaluate_results(reranked_results_with_ids, query):
    """Evaluate results using NDCG, handle missing files gracefully."""
    if not os.path.exists("evaluation_set.json"):
        print("\n⚠️ Skipping evaluation: 'evaluation_set.json' not found.")
        return
    
    try:
        with open("evaluation_set.json", "r") as f:
            eval_set = json.load(f)

        if query in eval_set:
            # Extract chunk_ids from the updated reranked_results_with_ids format
            y_pred = [[result.get("chunk_id") for result in reranked_results_with_ids]]
            y_true = [eval_set[query]]

            ndcg = ndcg_score(y_true, y_pred)
            print(f"\n📊 NDCG Score: {ndcg:.4f}")
        else:
            print("\n⚠️ No evaluation data available for this query.")
    except Exception as e:
        print(f"\n⚠️ Error during evaluation: {str(e)}")

# Step 6: Display Final Results
print(f"\n🔎 Query: {dotaz}")
print(f" Final Re-ranked Results (Cohere Reranker - Multilingual):\n")
for i, result in enumerate(reranked_results_with_ids, 1):
    chunk_id = result.get("chunk_id", "N/A")
    title = result.get("title", "N/A")
    text = result.get("chunk", "")
    
    # Display chunk_id and title along with the text
    print(f"{i}. [{chunk_id}] - {title}")
    print(f"   {text[:500]}..." if len(text) > 500 else text)
    print()  # Add a blank line between results

# Run Evaluation
evaluate_results(reranked_results_with_ids, dotaz)