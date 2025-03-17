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

# Creating the dataframe with test queries
# Creating the dataframe with all the given data
data = [
    (0, "Jaké přílohy musí žadatel doložit s žádostí o podporu - Prohlášení o přijatelnosti žadatele/partnera", "56"),
    (1, "Jaké přílohy musí žadatel doložit s žádostí o podporu - Prohlášení o přijatelnosti žadatele/partnera", "57"),
    (2, "Jaké přílohy musí žadatel doložit s žádostí o podporu - Prohlášení o přijatelnosti žadatele/partnera", "97"),
    (3, "Jaké přílohy musí žadatel doložit s žádostí o podporu - Prohlášení o přijatelnosti žadatele/partnera", "102"),
    (4, "Jaké přílohy musí žadatel doložit s žádostí o podporu - Doklad o obratu", "260"),
    (5, "Jaké dokumenty musí žadatel doložit před vydáním právního aktu o poskytnutí podpory (PA)", "113"),
    (6, "Jaké dokumenty musí žadatel doložit před vydáním právního aktu o poskytnutí podpory (PA)", "113"),
    (7, "Jaké dokumenty musí žadatel doložit před vydáním právního aktu o poskytnutí podpory (PA)", "113"),
    (8, "Jaké dokumenty musí žadatel doložit před vydáním právního aktu o poskytnutí podpory (PA)", "113"),
    (9, "Jaké dokumenty musí žadatel doložit před vydáním právního aktu o poskytnutí podpory (PA)", "113"),
    (10, "Jaké dokumenty musí žadatel doložit před vydáním právního aktu o poskytnutí podpory (PA)", "113"),
    (11, "Jaké dokumenty musí žadatel doložit před vydáním právního aktu o poskytnutí podpory (PA)", "113"),
    (12, "Jaké dokumenty musí žadatel doložit před vydáním právního aktu o poskytnutí podpory (PA)", "113"),
    (13, "Jaké dokumenty musí žadatel doložit před vydáním právního aktu o poskytnutí podpory (PA)", "113"),
    (14, "Jaké dokumenty musí žadatel doložit před vydáním právního aktu o poskytnutí podpory (PA)", "113"),
    (15, "Jaké dokumenty musí žadatel doložit před vydáním právního aktu o poskytnutí podpory (PA)", "113"),
    (16, "Jaké dokumenty musí žadatel doložit před vydáním právního aktu o poskytnutí podpory (PA)", "113"),
    (17, "Jaké dokumenty musí žadatel doložit před vydáním právního aktu o poskytnutí podpory (PA)", "113"),
    (18, "Jaké přílohy musí žadatel doložit s žádostí o podporu - Doklad o typu a právní formě příjemce", "57"),
    (19, "Jaké přílohy musí žadatel doložit s žádostí o podporu - Prokázání vlastnické struktury", "57"),
    (20, "Jaké přílohy musí žadatel doložit s žádostí o podporu - Nepovinné přílohy dle výzvy", "260"),
    (21, "Jaké dokumenty musí žadatel doložit před vydáním právního aktu o poskytnutí podpory (PA)", "113"),
    (22, "Jaké dokumenty musí žadatel doložit před vydáním právního aktu o poskytnutí podpory (PA)", "113"),
    (23, "Jaké datum je pro splnění indikátoru nejdůležitější?", "168"),
    (24, "Jaká je lhůta pro předložení zprávy o realizaci?", "125"),
    (25, "Může žadatel zahájit realizaci projektu před vydáním PA?", "126"),
    (26, "Jak se dokládají paušální náklady?", "229"),
    (27, "Musejí se u paušálních nákladů dokládat účetní doklady?", "229"),
    (28, "V jaké fázi musí být stavební záměr před vydáním PA?", "97"),
    (29, "Existuje vzor smlouvy o partnerství?", "114"),
    (30, "Může příjemce fakturovat partnerovi služby realizované pro projekt?", "63"),
    (31, "Jak je v projektu financován partner?", "62"),
    (32, "Jak dlouho může probíhat realizace projektu?", "121"),
    (33, "Je možné prodloužit realizaci projektu?", "124"),
    (34, "Může příjemce nesouhlasit s řídicím orgánem ohledně administrativního ověření ŽoP?", "247"),
    (35, "Jakou povinnou publicitu musí realizovat příjemce projektů?", "162"),
    (36, "Je příjemce správcem nebo zpracovatelem osobních údajů v projektu?", "134"),
    (37, "Jaké podmínky musí výzkumná organizace plnit při využití dotovaného vybavení?", "153"),
    (38, "Jak určovat kategorii podniku pro veřejné vysoké školy?", "29"),
    (39, "Kdy se nevyplňuje list Skupina podniků v příloze č. 6 PpŽP?", "59"),
    (40, "Jaká je lhůta pro odeslání dokumentů požadovaných k vydání rozhodnutí?", "113"),
    (41, "Co se stane pokud žadatel nedodá požadované podklady?", "113"),
    (42, "Jakým způsobem musí být požadované dokumenty odeslány?", "113"),
    (43, "Kdy je možné kombinovat ex-ante a ex-post platby na úrovni projektu?", "90"),
    (44, "Na základě čeho je určen způsob financování projektů?", "90"),
    (45, "Kdy se kontroluje střet zájmů?", "61"),
    (46, "Co je datum dosažení indikátoru?", "169"),
    (47, "Kdy se začínají předkládat ZoR?", "125"),
    (48, "Jakou povinnou publicitu musí realizovat příjemce projektů?", "162"),
    (49, "Jaké podmínky musí být plněny při využití vybavení podpořeného z dotace?", "153"),
    (50, "Jakými způsoby je zajištěno financování projektů?", "93"),
    (51, "Výčet podstatných změn zakládajících změnu právního aktu.", "142"),
    (52, "Využívá příslušný OP některou z metod zjednodušeného vykazování výdajů?", "225"),
    (53, "Jak se vypočte maximální počet jednotek vykázaných na zaměstnance?", "88"),
    (54, "Existuje vzor partnerské smlouvy?", "63"),
    (55, "Jaká je povinná publicita pro projekty nad 500 000 EUR?", "162"),
    (56, "V jaké dokumentaci jsou pravidla pro zadávání veřejných zakázek?", "149"),
    (57, "Kdy příjemce nemusí dodržovat postupy veřejných zakázek?", "164"),
    (58, "Co je to jeden podnik?", "156"),
    (59, "Kdy se začíná předkládat zpráva o realizaci projektu?", "125"),
    (60, "Jak dlouho může probíhat realizace projektu?", "121")
]

df_test = pd.DataFrame(data, columns=["Nr.", "Question", "Pages"])


def get_embedding(query):
    """Generate OpenAI embedding for the query."""
    try:
        response = openai_client.embeddings.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            input=query
        )
        return response.data[0].embedding
    except Exception as e:
        print(f" Error generating embedding for '{query}': {e}")
        return None


def perform_search(query, embedding_vector, top_k):
    """Perform BM25 and Vector Search with rate limiting."""
    url = f"https://{SEARCH_SERVICE_NAME}.search.windows.net/indexes/{INDEX_NAME}/docs/search?api-version=2023-11-01"
    headers = {"Content-Type": "application/json", "api-key": SEARCH_API_KEY}

    # BM25 Search
    bm25_body = {
        "search": query,
        "queryType": "simple",
        "searchFields": "chunk",
        "select": "chunk_id,parent_id,chunk,title",
        "top": top_k
    }
    bm25_response = requests.post(url, headers=headers, json=bm25_body)
    bm25_results = bm25_response.json().get("value", []) if bm25_response.status_code == 200 else []

 

    # Vector Search
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
    vector_response = requests.post(url, headers=headers, json=vector_body)
    vector_results = vector_response.json().get("value", []) if vector_response.status_code == 200 else []

    time.sleep(6)  # Ensure rate limiting

    return bm25_results, vector_results

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

def evaluate_results(df):
    """Evaluate reranked retrieval performance with API rate limiting."""
    results = []
    for index, row in df.iterrows():
        query = row["Question"]
        expected_pages = row["Pages"].split(", ")

        print(f"\n🔹 Query: {query}")
        print(f"📄 Expected Pages: {expected_pages}")

        # Generate embeddings
        embedding_vector = get_embedding(query)
        if embedding_vector is None:
            continue

        # Retrieve documents
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
        rerank_response = cohere_client.rerank(
            model="rerank-multilingual-v2.0",
            query=query,
            documents=[doc["chunk"] for doc in doc_objects]
        )

        time.sleep(6)  # Ensure 10 API calls per minute

        # Extract final reranked documents
        reranked_results = [doc_objects[result.index] for result in rerank_response.results]

        # Extract retrieved pages
        retrieved_pages_5 = [extract_page_number(doc["chunk_id"]) for doc in reranked_results[:5]]
        retrieved_pages_10 = [extract_page_number(doc["chunk_id"]) for doc in reranked_results[:10]]
        retrieved_pages_25 = [extract_page_number(doc["chunk_id"]) for doc in reranked_results[:25]]

        print(f"🔍 Retrieved Pages (Top 5): {retrieved_pages_5}")
        print(f"🔍 Retrieved Pages (Top 10): {retrieved_pages_10}")
        print(f"🔍 Retrieved Pages (Top 25): {retrieved_pages_25}")

        # Compute match accuracy
        match_5 = any(page in retrieved_pages_5 for page in expected_pages)
        match_10 = any(page in retrieved_pages_10 for page in expected_pages)
        match_25 = any(page in retrieved_pages_25 for page in expected_pages)

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

    return pd.DataFrame(results)

# Run evaluation with rate limiting
evaluation_results_multiple_top_k = evaluate_results(df_test)

# Save results
evaluation_results_multiple_top_k.to_csv("reranked_retrieval_evaluation.csv", index=False)

print("\n✅ Reranking evaluation complete. Results saved to reranked_retrieval_evaluation.csv.")

