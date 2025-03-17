#2_create_data_source.py
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SEARCH_SERVICE_NAME = os.getenv("AZURE_SEARCH_SERVICE_NAME")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "test"

# Create an indexer client
indexer_client = SearchIndexerClient(
    endpoint=f"https://{SEARCH_SERVICE_NAME}.search.windows.net",
    credential=AzureKeyCredential(SEARCH_API_KEY)
)

# Define the container
container = SearchIndexerDataContainer(name=CONTAINER_NAME)

# Create a data source connection
data_source_connection = SearchIndexerDataSourceConnection(
    name="test-data-source", # can be changed
    type="azureblob",
    connection_string=STORAGE_CONNECTION_STRING,
    container=container
)

# Create or update the data source
data_source = indexer_client.create_or_update_data_source_connection(data_source_connection)
print(f"Data source '{data_source.name}' created or updated")

