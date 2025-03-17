#1_create_index.py
#AZURE_SEARCH_INDEX_NAME="test5-index"
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    SearchIndex
)
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
SEARCH_SERVICE_NAME = os.getenv("AZURE_SEARCH_SERVICE_NAME")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "text-embedding-3-large")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_EMBEDDING_DIMENSIONS = int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "3072"))

# Connect to Azure AI Search
index_client = SearchIndexClient(
    endpoint=f"https://{SEARCH_SERVICE_NAME}.search.windows.net",
    credential=AzureKeyCredential(SEARCH_API_KEY)
)

# Delete existing index if needed
try:
    index_client.delete_index(INDEX_NAME)
    print(f"Deleted existing index '{INDEX_NAME}'")
except Exception:
    print(f"Index '{INDEX_NAME}' does not exist yet")

# Define fields with vector search
fields = [
    SearchField(name="title", type=SearchFieldDataType.String),
    SearchField(name="chunk_id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True, analyzer_name="keyword"),
    SearchField(name="chunk", type=SearchFieldDataType.String),
    SearchField(name="parent_id", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
    SearchField(
        name="vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        vector_search_dimensions=AZURE_OPENAI_EMBEDDING_DIMENSIONS,
        vector_search_profile_name="myHnswProfile"  # Ensure this is linked to the vectorizer
    )
]

# Define the vectorizer
vectorizer = AzureOpenAIVectorizer(
    vectorizer_name="myOpenAI",
    kind="azureOpenAI",
    parameters=AzureOpenAIVectorizerParameters(
        resource_url=AZURE_OPENAI_ENDPOINT,
        deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        model_name=AZURE_OPENAI_MODEL
    )
)

# Configure vector search profile
vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="myHnsw",
            parameters={"m": 4, "efConstruction": 400, "efSearch": 500, "metric": "cosine"}
        )
    ],
    profiles=[
        VectorSearchProfile(
            name="myHnswProfile",
            algorithm_configuration_name="myHnsw",
            vectorizer_name="myOpenAI"  # Ensure this profile is linked to vectorizer
        )
    ],
    vectorizers=[vectorizer]  # Ensure vectorizer is included
)

# Create the index
index = SearchIndex(
    name=INDEX_NAME,
    fields=fields,
    vector_search=vector_search
)

result = index_client.create_or_update_index(index)
print(f"✅ Index '{result.name}' created successfully!")
