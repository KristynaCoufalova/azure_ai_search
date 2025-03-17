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

def perform_bm25_search(query, top_k=10):
    """Perform BM25 search using Azure AI Search."""
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
            print(f" Azure Search Error: {response.status_code} - {response.text}")
            return []
        return response.json().get("value", [])
    except Exception as e:
        print(f" Error in BM25 search: {e}")
        return []

def extract_page_number(chunk_id):
    """Extract the last number from chunk_id and add 1."""
    match = re.search(r"pages_(\d+)", chunk_id)
    return str(int(match.group(1)) + 1) if match else None

def evaluate_results_with_multiple_top_k(df, top_k_values=[5, 10, 25]):
    """Evaluate retrieval performance for different top_k values and save results to CSV."""
    results = []

    for index, row in df.iterrows():
        query = row["Question"]
        expected_pages = row["Pages"].split(", ")

        print(f"\n Query: {query}")
        print(f" Expected Pages: {expected_pages}")

        # Perform searches for different top_k values
        retrieved_pages_dict = {}
        match_results = {}

        for top_k in top_k_values:
            search_results = perform_bm25_search(query, top_k=top_k)
            retrieved_pages = [extract_page_number(doc["chunk_id"]) for doc in search_results if doc.get("chunk_id")]

            retrieved_pages_dict[f"Retrieved Pages (Top {top_k})"] = retrieved_pages
            match_results[f"Match Found (Top {top_k})"] = any(page in retrieved_pages for page in expected_pages)

        # Store results
        result_entry = {
            "Query": query,
            "Expected Pages": expected_pages
        }
        result_entry.update(retrieved_pages_dict)
        result_entry.update(match_results)
        results.append(result_entry)

        time.sleep(0.5)  # Prevent rate limiting

    # Convert to DataFrame and save to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv("bm25_retrieval_evaluation_multiple_top_k.csv", index=False)

    print("\n📁 Results saved to 'bm25_retrieval_evaluation_multiple_top_k.csv'")
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
df_accuracy.to_csv("bm25_accuracy_scores.csv", index=False)

# Print accuracy results
print("\n✅ Accuracy Results:")
print(df_accuracy.to_string(index=False))