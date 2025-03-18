import os
import requests
import time
import re
import pandas as pd
import cohere
from openai import AzureOpenAI
from dotenv import load_dotenv

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

def get_embedding(query):
    """Generate OpenAI embedding for the query."""
    try:
        print(f" Generating embedding for: '{query}'")
        response = openai_client.embeddings.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            input=query
        )
        time.sleep(6)  # Rate limiting - ensure 10 API calls per minute
        return response.data[0].embedding
    except Exception as e:
        print(f" Error generating embedding for '{query}': {e}")
        return None

def perform_vector_search(embedding_vector, top_k=50):
    """Perform Azure AI vector search to get initial results."""
    url = f"https://{SEARCH_SERVICE_NAME}.search.windows.net/indexes/{INDEX_NAME}/docs/search?api-version=2023-11-01"
    headers = {"Content-Type": "application/json", "api-key": SEARCH_API_KEY}
    
    search_body = {
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
    
    try:
        print(f" Performing vector search with top_k={top_k}")
        response = requests.post(url, headers=headers, json=search_body)
        if response.status_code != 200:
            print(f" Azure Search Error: {response.status_code} - {response.text}")
            return []
        results = response.json().get("value", [])
        print(f" Vector search returned {len(results)} results")
        time.sleep(6)  # Rate limiting
        return results
    except Exception as e:
        print(f" Error in vector search: {e}")
        return []

def rerank_with_cohere(query, documents, top_k=10):
    """Rerank documents using Cohere Reranker."""
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
        time.sleep(6)  # Rate limiting - ensure 10 API calls per minute
        return rerank_response.results
    except Exception as e:
        print(f" Error in Cohere reranking: {e}")
        return []

def extract_page_number(chunk_id):
    """Extract the last number from chunk_id and add 1."""
    match = re.search(r"pages_(\d+)", chunk_id)
    return str(int(match.group(1)) + 1) if match else None

def evaluate_results_with_multiple_top_k(df, top_k_values=[5, 10, 25]):
    """Evaluate retrieval performance for different top_k values and save results to CSV."""
    results = []
    initial_vector_k = 50  # Get more results initially for reranking

    for index, row in df.iterrows():
        query = row["Question"]
        
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

        print(f"\n Query #{index+1}: {query}")
        print(f" Expected Pages: {expected_pages}")

        # Get embedding for vector search
        embedding_vector = get_embedding(query)
        if embedding_vector is None:
            continue

        # Get initial results with vector search
        initial_results = perform_vector_search(embedding_vector, top_k=initial_vector_k)
        
        # Filter out documents with empty chunks
        doc_objects = [doc for doc in initial_results if doc.get("chunk")]
        
        # Prepare documents for Cohere reranking
        if not doc_objects:
            print(f" No valid documents found for query: {query}")
            continue
            
        documents = [doc["chunk"] for doc in doc_objects]
        
        # Store results for different top_k values
        retrieved_pages_dict = {}
        match_results = {}

        for top_k in top_k_values:
            # Rerank with Cohere
            reranked_results = rerank_with_cohere(query, documents, top_k=top_k)
            
            # Extract page numbers from reranked results
            reranked_docs = []
            for result in reranked_results:
                if 0 <= result.index < len(doc_objects):
                    reranked_docs.append(doc_objects[result.index])
            
            retrieved_pages = [extract_page_number(doc["chunk_id"]) for doc in reranked_docs if doc.get("chunk_id")]
            # Filter out None values
            retrieved_pages = [page for page in retrieved_pages if page is not None]
            
            retrieved_pages_dict[f"Retrieved Pages (Top {top_k})"] = retrieved_pages
            match_results[f"Match Found (Top {top_k})"] = any(page in retrieved_pages for page in expected_pages)
            
            print(f" Retrieved Pages (Top {top_k}): {retrieved_pages}")
            print(f" Match Found (Top {top_k}): {match_results[f'Match Found (Top {top_k})']}")

        # Store results
        result_entry = {
            "Query": query,
            "Expected Pages": expected_pages
        }
        result_entry.update(retrieved_pages_dict)
        result_entry.update(match_results)
        results.append(result_entry)

    # Convert to DataFrame and save to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv("cohere_reranking_evaluation_multiple_top_k.csv", index=False)

    print("\n📁 Results saved to 'cohere_reranking_evaluation_multiple_top_k.csv'")
    return df_results

# Run evaluation with different top_k values
evaluation_results_multiple_top_k = evaluate_results_with_multiple_top_k(df_test)

# Print results
print("\n Evaluation Results:")
print(evaluation_results_multiple_top_k.to_string(index=False))

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
df_accuracy.to_csv("cohere_reranking_accuracy_scores.csv", index=False)

# Print accuracy results
print("\n✅ Accuracy Results:")
print(df_accuracy.to_string(index=False))