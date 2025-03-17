import os
import logging
import time
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexerClient
from azure.core.credentials import AzureKeyCredential

# Load environment variables
load_dotenv()
SEARCH_SERVICE_NAME = os.getenv("AZURE_SEARCH_SERVICE_NAME")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Configure logging
logging.basicConfig(
    filename="indexer_status.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Initialize Azure Search Client
indexer_client = SearchIndexerClient(
    endpoint=f"https://{SEARCH_SERVICE_NAME}.search.windows.net",
    credential=AzureKeyCredential(SEARCH_API_KEY),
)

def check_indexer_status():
    """Check and log indexer status every 60 seconds."""
    while True:
        try:
            indexer_status = indexer_client.get_indexer_status(f"{INDEX_NAME}-indexer")

            # Extract Execution History
            execution_history = indexer_status.execution_history

            # Log General Indexer Info
            logging.info(f"Indexer Name: {INDEX_NAME}-indexer")
            logging.info(f"Current Status: {indexer_status.status}")
            logging.info(f"Last Execution Result: {indexer_status.last_result.status}")
            logging.info(f"Execution History Count: {len(execution_history)}")

            # Log Detailed Execution Info
            for i, execution in enumerate(execution_history[:5]):  # Show last 5 executions
                logging.info(f"\nExecution {i+1}:")
                logging.info(f"  Start Time: {execution.start_time}")
                logging.info(f"  End Time: {execution.end_time}")
                logging.info(f"  Status: {execution.status}")
                logging.info(f"  Items Processed: {execution.item_count}")
                logging.info(f"  Items Failed: {execution.failed_item_count}")

                if execution.error_message:
                    logging.error(f"  Error Message: {execution.error_message}")

                if execution.warnings:
                    logging.warning(f"  Warnings: {execution.warnings}")

            # Print Summary in Terminal
            print(f"Checked at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Indexer Status: {indexer_status.status}")
            print(f"Last Execution Result: {indexer_status.last_result.status}")
            print(f"Check 'indexer_status.log' for detailed logs.\n")

        except Exception as e:
            logging.error(f"Error checking indexer status: {str(e)}")
            print(f"Error: {str(e)}")

        # Wait 1 minute before checking again
        time.sleep(60)

# Start monitoring the indexer
check_indexer_status()
