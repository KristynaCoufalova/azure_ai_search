#4_create_indexer.py
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndexer,
    IndexingParameters,
    IndexingParametersConfiguration,
    BlobIndexerImageAction
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SEARCH_SERVICE_NAME = os.getenv("AZURE_SEARCH_SERVICE_NAME")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
USE_OCR = os.getenv("USE_OCR", "false").lower() == "true"

# Define the skillset name
skillset_name = f"{INDEX_NAME}-skillset"

# Create the Search Indexer Client
credential = AzureKeyCredential(SEARCH_API_KEY)
indexer_client = SearchIndexerClient(
    endpoint=f"https://{SEARCH_SERVICE_NAME}.search.windows.net",
    credential=credential
)

# Create an indexer
indexer_name = f"{INDEX_NAME}-indexer"
indexer_parameters = None
if USE_OCR:
    indexer_parameters = IndexingParameters(
        configuration=IndexingParametersConfiguration(
            image_action=BlobIndexerImageAction.GENERATE_NORMALIZED_IMAGE_PER_PAGE,
            query_timeout=None))

indexer = SearchIndexer(
    name=indexer_name,
    description="Indexer to index documents and generate embeddings",
    skillset_name=skillset_name,
    target_index_name=INDEX_NAME,
    data_source_name="test1-data-source",
    parameters=indexer_parameters
)

indexer_result = indexer_client.create_or_update_indexer(indexer)
indexer_client.run_indexer(indexer_name)
print(f'✅ {indexer_name} is created and running. If queries return no results, please wait a bit and try again.')
