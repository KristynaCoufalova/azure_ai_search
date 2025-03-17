import os
import requests
import time
import re
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SEARCH_SERVICE_NAME = os.getenv("AZURE_SEARCH_SERVICE_NAME")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Manually created test dataset
data = [
    (0, "Jaké přílohy musí žadatel doložit s žádostí o podporu", "56, 57, 97, 102"),
    (4, "Jaké přílohy musí žadatel doložit s žádostí o podporu - Doklad o obratu", "260"),
    (5, "Jaké dokumenty musí žadatel doložit před vydáním právního aktu o poskytnutí podpory (PA)", "113"),
    (38, "Jakým způsobem postupovat u veřejných vysokých škol a v.v.i. při určování kategorie malého/středního/velkého podniku?", "29"),
    (57, "Jak dlouho může probíhat realizace projektu?", "121"),
]

df_test = pd.DataFrame(data, columns=["Nr.", "Question", "Pages"])

def perform_bm25_search(query, top_k):
    """Perform Azure AI BM25 search."""
    url = f"https://{SEARCH_SERVICE_NAME}.search.windows.net/indexes/{INDEX_NAME}/docs/search?api-version=2023-11-01"
    headers = {"Content-Type": "application/json", "api-key": SEARCH_API_KEY}
    
    search_body = {
        "search": query,
        "queryType": "simple",
        "searchFields": "chunk",
        "select": "chunk_id,parent_id,chunk,title",
        "top": top_k
    }
    
    try:
        response = requests.post(url, headers=headers, json=search_body)
        if response.status_code != 200:
            print(f"❌ Azure Search Error: {response.status_code} - {response.text}")
            return []
        return response.json().get("value", [])
    except Exception as e:
        print(f"❌ Error in BM25 search: {e}")
        return []

def extract_page_number(chunk_id):
    """Extract the last number from chunk_id and add 1."""
    match = re.search(r"pages_(\d+)", chunk_id)
    return str(int(match.group(1)) + 1) if match else None

def evaluate_results(df):
    """Evaluate BM25 retrieval performance and compute accuracy."""
    results = []
    total_queries = len(df)
    correct_5, correct_10, correct_25 = 0, 0, 0  # Track correct matches

    for index, row in df.iterrows():
        query = row["Question"]
        expected_pages = row["Pages"].split(", ")

        print(f"\n🔹 Query: {query}")
        print(f"📄 Expected Pages: {expected_pages}")

        # Perform BM25 searches for different top_k values
        search_results_5 = perform_bm25_search(query, 5)
        search_results_10 = perform_bm25_search(query, 10)
        search_results_25 = perform_bm25_search(query, 25)

        # Extract retrieved pages
        retrieved_pages_5 = [extract_page_number(doc["chunk_id"]) for doc in search_results_5 if doc.get("chunk_id")]
        retrieved_pages_10 = [extract_page_number(doc["chunk_id"]) for doc in search_results_10 if doc.get("chunk_id")]
        retrieved_pages_25 = [extract_page_number(doc["chunk_id"]) for doc in search_results_25 if doc.get("chunk_id")]

        print(f"🔍 Retrieved Pages (Top 5): {retrieved_pages_5}")
        print(f"🔍 Retrieved Pages (Top 10): {retrieved_pages_10}")
        print(f"🔍 Retrieved Pages (Top 25): {retrieved_pages_25}")

        # Check if expected pages match retrieved pages
        match_5 = any(page in retrieved_pages_5 for page in expected_pages)
        match_10 = any(page in retrieved_pages_10 for page in expected_pages)
        match_25 = any(page in retrieved_pages_25 for page in expected_pages)

        # Update accuracy counters
        correct_5 += int(match_5)
        correct_10 += int(match_10)
        correct_25 += int(match_25)

        results.append({
            "Query": query,
            "Expected Pages": expected_pages,
            "Retrieved Pages (Top 5)": retrieved_pages_5,
            "Retrieved Pages (Top 10)": retrieved_pages_10,
            "Retrieved Pages (Top 25)": retrieved_pages_25,
            "Match Found (Top 5)": match_5,
            "Match Found (Top 10)": match_10,
            "Match Found (Top 25)": match_25
        })

        time.sleep(0.5)  # Prevent rate limiting

    # Compute accuracy
    accuracy_5 = round((correct_5 / total_queries) * 100, 2)
    accuracy_10 = round((correct_10 / total_queries) * 100, 2)
    accuracy_25 = round((correct_25 / total_queries) * 100, 2)

    print(f"\n📊 Accuracy Scores:")
    print(f"✅ Accuracy (Top 5): {accuracy_5}%")
    print(f"✅ Accuracy (Top 10): {accuracy_10}%")
    print(f"✅ Accuracy (Top 25): {accuracy_25}%")

    return pd.DataFrame(results), accuracy_5, accuracy_10, accuracy_25

# Run evaluation
bm25_evaluation_results, acc_5, acc_10, acc_25 = evaluate_results(df_test)

# Save results to CSV
bm25_evaluation_results.to_csv("bm25_retrieval_evaluation.csv", index=False)

# Save accuracy scores
accuracy_data = pd.DataFrame({
    "Top K": [5, 10, 25],
    "Accuracy (%)": [acc_5, acc_10, acc_25]
})
accuracy_data.to_csv("bm25_accuracy_scores.csv", index=False)

print("\n✅ BM25 evaluation complete. Results saved to bm25_retrieval_evaluation.csv and bm25_accuracy_scores.csv.")
