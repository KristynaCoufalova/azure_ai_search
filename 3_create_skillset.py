#3_create_skillset.py

import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import (
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    SearchIndexerSkillset
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SEARCH_SERVICE_NAME = os.getenv("AZURE_SEARCH_SERVICE_NAME")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "text-embedding-3-large")  # Ensure model is set
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
PDF_SPLIT_FUNCTION_URL = os.getenv("PDF_SPLIT_FUNCTION_URL", "https://pdf-split-function-kc.azurewebsites.net/api/split_pdf")
PDF_SPLIT_FUNCTION_KEY = os.getenv("PDF_SPLIT_FUNCTION_KEY", "")  # Optional: Add your function key if needed

# Print loaded values for debugging
print(f"AZURE_OPENAI_ENDPOINT: {AZURE_OPENAI_ENDPOINT}")
print(f"AZURE_OPENAI_EMBEDDING_DEPLOYMENT: {AZURE_OPENAI_EMBEDDING_DEPLOYMENT}")
print(f"PDF_SPLIT_FUNCTION_URL: {PDF_SPLIT_FUNCTION_URL}")

# Define the skillset name
skillset_name = f"{INDEX_NAME}-skillset"

# Create the Search Indexer Client
credential = AzureKeyCredential(SEARCH_API_KEY)
indexer_client = SearchIndexerClient(
    endpoint=f"https://{SEARCH_SERVICE_NAME}.search.windows.net",
    credential=credential
)

def create_skillset():
    """Creates a skillset for Azure AI Search using custom PDF splitting function."""
    skills = []

    # Custom PDF splitting Azure Function (instead of built-in Split Skill)
    custom_split_skill = {
        "@odata.type": "#Microsoft.Skills.Custom.WebApiSkill",
        "description": "Custom PDF page splitting using Azure Function",
        "uri": PDF_SPLIT_FUNCTION_URL,
        "context": "/document",
        "batchSize": 1,  # Process one document at a time
        "inputs": [
            {
                "name": "content",
                "source": "/document/metadata_storage_path"
            }
        ],
        "outputs": [
            {
                "name": "pages",
                "targetName": "pages"
            }
        ]
    }
    
    # Add function key if provided
    if PDF_SPLIT_FUNCTION_KEY:
        custom_split_skill["httpHeaders"] = {
            "x-functions-key": PDF_SPLIT_FUNCTION_KEY
        }
    
    skills.append(custom_split_skill)

    # Check if required values are available
    if not AZURE_OPENAI_ENDPOINT:
        raise ValueError("AZURE_OPENAI_ENDPOINT is not set in environment variables")
    if not AZURE_OPENAI_EMBEDDING_DEPLOYMENT:
        raise ValueError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT is not set in environment variables")
    if not AZURE_OPENAI_KEY:
        raise ValueError("AZURE_OPENAI_KEY is not set in environment variables")

    # Embedding Skill
    embedding_skill = {
        "@odata.type": "#Microsoft.Skills.Text.AzureOpenAIEmbeddingSkill",
        "description": "Skill to generate embeddings via Azure OpenAI",
        "context": "/document/pages/*",
        "resourceUri": AZURE_OPENAI_ENDPOINT,
        "deploymentId": AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        "apiKey": AZURE_OPENAI_KEY,
        "modelName": AZURE_OPENAI_MODEL,  # Explicitly specify model name
        "inputs": [
            {"name": "text", "source": "/document/pages/*"}
        ],
        "outputs": [
            {"name": "embedding", "targetName": "vector"}
        ]
    }

    # Print the embedding skill for debugging
    print("Embedding skill configuration:")
    print(f"  resourceUri: {embedding_skill['resourceUri']}")
    print(f"  deploymentId: {embedding_skill['deploymentId']}")
    print(f"  API Key present: {'Yes' if embedding_skill['apiKey'] else 'No'}")

    skills.append(embedding_skill)

    # Index Projection - Updated to match the actual output format of the function
    index_projection = {
        "selectors": [
            {
                "targetIndexName": INDEX_NAME,
                "parentKeyFieldName": "parent_id",
                "sourceContext": "/document/pages/*",
                "mappings": [
                    {"name": "chunk", "source": "/document/pages/*"},
                    {"name": "vector", "source": "/document/pages/*/vector"},
                    {"name": "title", "source": "/document/metadata_storage_name"}
                ]
            }
        ],
        "parameters": {
            "projectionMode": "skipIndexingParentDocuments"
        }
    }

    # Create and return the skillset
    skillset = SearchIndexerSkillset(
        name=skillset_name,
        description="Skillset using custom Azure Function for PDF page splitting and embedding documents",
        skills=skills
    )

    # Set the index_projection property dynamically
    setattr(skillset, "index_projection", index_projection)

    return skillset

# Create and deploy the skillset
try:
    skillset = create_skillset()
    indexer_client.create_or_update_skillset(skillset)
    print(f"✅ Skillset '{skillset.name}' created successfully!")
except ValueError as e:
    print(f"Error: {str(e)}")
except Exception as e:
    print(f"Unexpected error: {str(e)}")