import os
import requests
import time
import re
import json
import pandas as pd
from dotenv import load_dotenv
import cohere
from openai import AzureOpenAI
from collections import defaultdict
import random

# Load environment variables
load_dotenv()

SEARCH_SERVICE_NAME = os.getenv("AZURE_SEARCH_SERVICE_NAME")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = "text-embedding-3-large"

# Initialize clients
cohere_client = cohere.Client(COHERE_API_KEY)
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-01"
)

# Read the dataframe from CSV file
csv_file = "opjak_eval_pzp_250.csv"
print(f"📊 Loading test data from {csv_file}...")
try:
    df_test = pd.read_csv(csv_file, sep=';')
    print(f"✅ Successfully loaded {len(df_test)} rows from CSV")
except Exception as e:
    print(f"❌ Error loading CSV file: {e}")
    exit(1)

# Display sample of the loaded data
print("\n📝 Sample of loaded data:")
print(df_test.head())

# Save partial results to handle interruptions
RESULTS_FILE = "partial_retrieval_results.csv"

def get_embedding(query, max_retries=3):
    """Generate OpenAI embedding for the query with retry logic."""
    for attempt in range(max_retries):
        try:
            response = openai_client.embeddings.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                input=query
            )
            return response.data[0].embedding
        except Exception as e:
            wait_time = (2 ** attempt) + random.random()
            print(f" Error generating embedding for '{query}': {e}")
            print(f" Retrying in {wait_time:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
            time.sleep(wait_time)
    
    print(f" Failed to generate embedding after {max_retries} attempts")
    return None


def perform_search(query, embedding_vector, top_k, max_retries=3):
    """Perform BM25 and Vector Search with retry logic."""
    url = f"https://{SEARCH_SERVICE_NAME}.search.windows.net/indexes/{INDEX_NAME}/docs/search?api-version=2023-11-01"
    headers = {"Content-Type": "application/json", "api-key": SEARCH_API_KEY}

    # BM25 Search with retries
    bm25_results = []
    for attempt in range(max_retries):
        try:
            bm25_body = {
                "search": query,
                "queryType": "simple",
                "searchFields": "chunk",
                "select": "chunk_id,parent_id,chunk,title",
                "top": top_k
            }
            bm25_response = requests.post(url, headers=headers, json=bm25_body, timeout=30)
            if bm25_response.status_code == 200:
                bm25_results = bm25_response.json().get("value", [])
                break
            else:
                print(f" Azure BM25 Search Error: {bm25_response.status_code}")
                time.sleep((2 ** attempt) + random.random())
        except Exception as e:
            wait_time = (2 ** attempt) + random.random()
            print(f" Error in BM25 search: {e}")
            print(f" Retrying in {wait_time:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
            time.sleep(wait_time)

    # Vector Search with retries
    vector_results = []
    for attempt in range(max_retries):
        try:
            vector_body = {
                "select": "chunk_id,parent_id,chunk,title",
                "top": top_k,
                "vectorQueries": [
                    {
                        "kind": "vector",
                        "vector": embedding_vector,
                        "fields": "vector",
                        "k": top_k
                    }
                ]
            }
            vector_response = requests.post(url, headers=headers, json=vector_body, timeout=30)
            if vector_response.status_code == 200:
                vector_results = vector_response.json().get("value", [])
                break
            else:
                print(f" Azure Vector Search Error: {vector_response.status_code}")
                time.sleep((2 ** attempt) + random.random())
        except Exception as e:
            wait_time = (2 ** attempt) + random.random()
            print(f" Error in vector search: {e}")
            print(f" Retrying in {wait_time:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
            time.sleep(wait_time)

    # Add rate limiting with some randomness to avoid synchronized requests
    time.sleep(6 + random.random())

    return bm25_results, vector_results

def rerank_with_cohere(query, documents, top_k=10, max_retries=3):
    """Rerank documents using Cohere Reranker with retry logic."""
    for attempt in range(max_retries):
        try:
            print(f" Sending request to Cohere Reranker API...")
            
            # Use Cohere Multilingual Reranker (supports Czech)
            rerank_response = cohere_client.rerank(
                model="rerank-multilingual-v2.0",
                query=query,
                documents=documents,
                top_n=top_k
            )
            
            print(f" Cohere reranking successful - received {len(rerank_response.results)} results")
            # Add rate limiting with some randomness to avoid synchronized requests
            time.sleep(6 + random.random())
            return rerank_response.results
        except Exception as e:
            wait_time = (2 ** attempt) + random.random() * 2  # More randomness for connection issues
            print(f" Error in Cohere reranking: {e}")
            print(f" Retrying in {wait_time:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
            time.sleep(wait_time)
    
    print(f" Failed to rerank with Cohere after {max_retries} attempts")
    return []

def reciprocal_rank_fusion(results_bm25, results_vector, k=60):
    """Merges BM25 and Vector Search results using Reciprocal Rank Fusion (RRF)."""
    fused_scores = defaultdict(float)
    
    # Assign RRF scores from BM25
    for rank, doc in enumerate(results_bm25, start=1):
        fused_scores[doc["chunk_id"]] += 1 / (k + rank)

    # Assign RRF scores from Vector Search
    for rank, doc in enumerate(results_vector, start=1):
        fused_scores[doc["chunk_id"]] += 1 / (k + rank)

    # Sort by RRF score (higher is better)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

def extract_page_number(chunk_id):
    """Extract the last number from chunk_id and add 1."""
    match = re.search(r"pages_(\d+)", chunk_id)
    return str(int(match.group(1)) + 1) if match else None

def load_existing_results():
    """Load existing partial results if available."""
    try:
        if os.path.exists(RESULTS_FILE):
            df = pd.read_csv(RESULTS_FILE)
            print(f"✅ Loaded {len(df)} existing results from {RESULTS_FILE}")
            return df
    except Exception as e:
        print(f"❌ Error loading existing results: {e}")
    return pd.DataFrame()

def evaluate_results(df):
    """Evaluate reranked retrieval performance with API rate limiting and error handling."""
    
    # Load existing results to resume from interruptions
    existing_results_df = load_existing_results()
    existing_queries = set()
    if not existing_results_df.empty:
        existing_queries = set(existing_results_df["Query"].tolist())
    
    results = existing_results_df.to_dict('records') if not existing_results_df.empty else []
    
    # Start or resume processing
    for index, row in df.iterrows():
        query = row["Question"]
        
        # Skip if we've already processed this query
        if query in existing_queries:
            print(f"\n🔄 Skipping already processed query: {query}")
            continue
            
        # Convert Pages to string and handle both float and string formats
        if pd.isna(row["Pages"]):
            continue  # Skip rows with missing Pages
            
        # Convert to string and handle both individual values and comma-separated values
        page_str = str(row["Pages"]).strip()
        if "," in page_str:
            expected_pages = [p.strip() for p in page_str.split(",")]
        else:
            # Handle single numerical value (potentially with decimal point)
            expected_pages = [str(int(float(page_str)))]  # Convert to int then back to str

        print(f"\n🔹 Query #{index+1}: {query}")
        print(f"📄 Expected Pages: {expected_pages}")

        # Generate embeddings with retry
        embedding_vector = get_embedding(query)
        if embedding_vector is None:
            continue

        # Retrieve documents with retry
        bm25_results, vector_results = perform_search(query, embedding_vector, 50)

        # Fuse results using RRF
        fused_results = reciprocal_rank_fusion(bm25_results, vector_results)

        # Retrieve full document details
        doc_objects = [
            next((doc for doc in bm25_results + vector_results if doc["chunk_id"] == doc_id), None)
            for doc_id, _ in fused_results
        ]
        doc_objects = [doc for doc in doc_objects if doc and doc.get("chunk")]

        # Cohere Reranking
        if not doc_objects:
            print(f" No valid documents found for query: {query}")
            continue
        
        # Prepare documents for reranking
        documents = [doc["chunk"] for doc in doc_objects]
        
        # Rerank with Cohere with retry
        reranked_results = rerank_with_cohere(query, documents, top_k=25)
        
        # If reranking failed, use the original results
        if not reranked_results:
            print(" Using original fused results without reranking.")
            # Just use the first 25 fused results
            reranked_docs = doc_objects[:25]
        else:
            # Extract final reranked documents
            reranked_docs = [doc_objects[result.index] for result in reranked_results 
                            if 0 <= result.index < len(doc_objects)]

        # Extract retrieved pages
        retrieved_pages_all = [extract_page_number(doc["chunk_id"]) for doc in reranked_docs if doc.get("chunk_id")]
        retrieved_pages_all = [page for page in retrieved_pages_all if page is not None]
        
        # Get different slices for evaluation
        retrieved_pages_5 = retrieved_pages_all[:5]
        retrieved_pages_10 = retrieved_pages_all[:10]
        retrieved_pages_25 = retrieved_pages_all[:25]

        print(f"🔍 Retrieved Pages (Top 5): {retrieved_pages_5}")
        print(f"🔍 Retrieved Pages (Top 10): {retrieved_pages_10}")
        print(f"🔍 Retrieved Pages (Top 25): {retrieved_pages_25}")

        # Compute match accuracy
        match_5 = any(page in retrieved_pages_5 for page in expected_pages)
        match_10 = any(page in retrieved_pages_10 for page in expected_pages)
        match_25 = any(page in retrieved_pages_25 for page in expected_pages)

        # Store results
        result_entry = {
            "Query": query,
            "Expected Pages": expected_pages,
            "Retrieved Pages (Top 5)": retrieved_pages_5,
            "Retrieved Pages (Top 10)": retrieved_pages_10,
            "Retrieved Pages (Top 25)": retrieved_pages_25,
            "Match Found (Top 5)": match_5,
            "Match Found (Top 10)": match_10,
            "Match Found (Top 25)": match_25
        }
        results.append(result_entry)

        # Save partial results after each query to handle interruptions
        partial_df = pd.DataFrame(results)
        partial_df.to_csv(RESULTS_FILE, index=False)
        print(f" Saved partial results ({len(results)} queries processed so far)")

    return pd.DataFrame(results)

# Run evaluation with error handling and resumption
try:
    print("\n🚀 Starting evaluation with resilient error handling...")
    evaluation_results_multiple_top_k = evaluate_results(df_test)

    # Save final results
    evaluation_results_multiple_top_k.to_csv("reranked_retrieval_evaluation.csv", index=False)
    print("\n✅ Reranking evaluation complete. Results saved to reranked_retrieval_evaluation.csv.")

    # Compute accuracy
    def compute_accuracy(df, top_k_values=[5, 10, 25]):
        """Compute accuracy for each top_k value."""
        accuracy_results = {}

        for top_k in top_k_values:
            match_column = f"Match Found (Top {top_k})"
            accuracy = df[match_column].mean()  # Mean gives proportion of True values
            accuracy_results[f"Accuracy (Top {top_k})"] = accuracy

        return accuracy_results

    # Compute accuracy for each top_k
    accuracy_scores = compute_accuracy(evaluation_results_multiple_top_k)

    # Convert to DataFrame for display and saving
    df_accuracy = pd.DataFrame([accuracy_scores])

    # Save accuracy results to CSV
    df_accuracy.to_csv("hybrid_reranking_accuracy_scores.csv", index=False)

    # Print accuracy results
    print("\n✅ Accuracy Results:")
    print(df_accuracy.to_string(index=False))

except KeyboardInterrupt:
    print("\n⚠️ Evaluation interrupted by user. Partial results have been saved.")
except Exception as e:
    print(f"\n❌ Error during evaluation: {e}")
    print("⚠️ Partial results have been saved and can be resumed later.")