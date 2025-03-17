from azure.storage.blob import BlobServiceClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "test"  # Change to your actual container name

# Connect to Azure Storage
blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

# List all files in the container
blobs = list(container_client.list_blobs())

if blobs:
    print(f"Found {len(blobs)} files in Azure Storage:")
    for blob in blobs:
        print(f"📂 {blob.name}")
else:
    print("No files found in Azure Storage. Upload PDFs and retry indexing.")
