import os
import pandas as pd
import requests
import time
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SEARCH_SERVICE_NAME = os.getenv("AZURE_SEARCH_SERVICE_NAME")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = "text-embedding-3-large"

# Initialize OpenAI client
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-01"
)

# Load test dataset
df = pd.read_csv("test_dataset.csv")

def get_embedding(query):
    """Generate OpenAI embedding for the query and handle errors."""
    try:
        response = openai_client.embeddings.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            input=query
        )
        embedding = response.data[0].embedding
        print(f" Embedding generated for query: '{query}' (first 5 dims: {embedding[:5]})")
        return embedding
    except Exception as e:
        print(f" Error generating embedding for '{query}': {e}")
        return None

def perform_vector_search(embedding_vector, top_k=10):
    """Perform Azure AI vector search and check for errors."""
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
                "k": 50
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=search_body)
        
        if response.status_code != 200:
            print(f" Azure Search Error: {response.status_code} - {response.text}")
            return []

        results = response.json().get("value", [])
        print(f" Retrieved {len(results)} results from Azure Search.")
        return results

    except Exception as e:
        print(f" Error in vector search: {e}")
        return []

def extract_page_number(chunk_id):
    """Extract the last two characters from chunk_id as page number."""
    try:
        return chunk_id[-2:]  # Get last two characters
    except Exception:
        return "XX"  # Fallback for errors

def fuzzy_match(expected_answer, retrieved_texts, top_k):
    """Perform a case-insensitive substring match within top_k results."""
    return any(expected_answer.lower() in text.lower() for text in retrieved_texts[:top_k])

def evaluate_results(df):
    """Evaluate retrieval performance and print debug output with corrected page extraction."""
    results = []
    
    for index, row in df.iterrows():
        query = row["Question"]
        expected_answer = str(row["Expected Answer"]).strip()
        expected_page = str(row["Pages"]).strip()
        
        print("\n--------------------------------------")
        print(f" Query: {query}")
        print(f" Expected Answer: {expected_answer}")
        print(f" Expected Page: {expected_page}")
        
        embedding_vector = get_embedding(query)
        if embedding_vector is None:
            print("⚠️ Skipping this query due to embedding error.")
            continue
        
        search_results = perform_vector_search(embedding_vector)
        
        # Extract correct page numbers from chunk_id
        retrieved_pages = [extract_page_number(str(doc["chunk_id"]).strip()) for doc in search_results]
        retrieved_texts = [str(doc["chunk"]).strip() for doc in search_results]
        
        # Print retrieved pages with corrected page numbers
        print(f"🔍 Retrieved Pages (Fixed): {retrieved_pages[:5]}")
        print(f"📖 Top Retrieved Texts:")
        for i, text in enumerate(retrieved_texts[:3]):
            print(f"  ✅ Page {retrieved_pages[i]}: {text[:200]}...")  # Show only first 200 chars

        # Compute match scores using fuzzy matching
        top_5_match = fuzzy_match(expected_answer, retrieved_texts, 5)
        top_10_match = fuzzy_match(expected_answer, retrieved_texts, 10)
        top_25_match = fuzzy_match(expected_answer, retrieved_texts, 25)

        results.append({
            "Query": query,
            "Expected Answer": expected_answer,
            "Expected Page": expected_page,
            "Retrieved Pages": retrieved_pages[:10],  # Store top 10 retrieved pages (shortened)
            "Top 5 Match": top_5_match,
            "Top 10 Match": top_10_match,
            "Top 25 Match": top_25_match
        })

        # Add a delay to avoid API rate limits
        time.sleep(0.5)
    
    return pd.DataFrame(results)

# Run evaluation
evaluation_results = evaluate_results(df)
evaluation_results.to_csv("retrieval_evaluation.csv", index=False)
print("\n✅ Evaluation complete. Results saved to retrieval_evaluation.csv.")

