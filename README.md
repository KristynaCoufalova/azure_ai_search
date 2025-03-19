Azure AI Search RAG Implementation

This project implements a Retrieval-Augmented Generation (RAG) system using Azure AI Search with vector capabilities, Azure OpenAI embeddings, and hybrid search methods.

Project Overview

The project creates a complete RAG pipeline from document indexing to retrieval evaluation, providing various search methods and comparing their performance. The system utilizes:

Azure AI Search

Azure OpenAI embeddings

Cohere reranking

Hybrid search algorithms

Features

Azure AI Search index with vector search capabilities

PDF document processing and chunking

Text embedding generation via Azure OpenAI

Multiple retrieval methods:

BM25 (lexical search)

Vector similarity search

Cohere reranking

Reciprocal Rank Fusion (RRF)

Comprehensive evaluation framework

Installation

Clone this repository

Install the required dependencies:

pip install -r Tutorial-rag-requirements.txt

Set up environment variables by creating a .env file with the following variables:

AZURE_SEARCH_SERVICE_NAME=your-search-service-name
AZURE_SEARCH_API_KEY=your-search-api-key
AZURE_SEARCH_INDEX_NAME=your-index-name
AZURE_STORAGE_CONNECTION_STRING=your-storage-connection-string
AZURE_OPENAI_ENDPOINT=your-openai-endpoint
AZURE_OPENAI_KEY=your-openai-key
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your-embedding-deployment-name
AZURE_OPENAI_MODEL=text-embedding-3-large
AZURE_OPENAI_EMBEDDING_DIMENSIONS=3072
PDF_SPLIT_FUNCTION_URL=your-function-url
PDF_SPLIT_FUNCTION_KEY=your-function-key
COHERE_API_KEY=your-cohere-api-key

Usage

1. Create and Set Up Azure Resources

Run the following scripts in sequence to set up the infrastructure:

python 1_create_index.py
python 2_create_data_source.py
python 3_create_skillset.py
python 4_create_indexer.py

2. Check Status and Monitor Resources

python 5_check_index_properties.py
python 6_check_if_indexer_running.py
python 7_check_indexed_data.py
python 8_check_pdfs_in_azure_storage.py
python 9_check_if_connected_to_azure_search.py
python 10_get_chunks.py

3. Perform Searches with Different Methods

# BM25 (Keyword-based) Search
python 12_perform_b25_search.py

# Vector Similarity Search
python 11_perform_vector_similarity_Search.py

# Cohere Reranking
python 13_perform_cohere_reranking.py

# Reciprocal Rank Fusion (Hybrid Search)
python 14_perform_rrf.py

4. Run Evaluation Tests

# Evaluate Vector Search
python 15_test_vector_search.py

# Evaluate BM25 Search
python 16_test_bm25_search.py

# Evaluate Cohere Reranking
python 17_test_cohere_reranking_search.py

# Evaluate RRF with Reranking
python 18_test_rrf.py

Project Structure

File

Description

1_create_index.py

Creates the Azure AI Search index with vector search capabilities

2_create_data_source.py

Sets up the data source connection to Azure Blob Storage

3_create_skillset.py

Creates a skillset for PDF processing and embedding generation

4_create_indexer.py

Sets up and runs the indexer to process documents

5_check_index_properties.py

Verifies the index configuration

6_check_if_indexer_running.py

Monitors indexer status

7_check_indexed_data.py

Checks indexed documents

8_check_pdfs_in_azure_storage.py

Lists PDFs in Azure Storage

9_check_if_connected_to_azure_search.py

Verifies connection to Azure Search

10_get_chunks.py

Exports indexed chunks to a JSON file

11_perform_vector_similarity_Search.py

Performs vector similarity search

12_perform_b25_search.py

Performs BM25 keyword search

13_perform_cohere_reranking.py

Implements Cohere reranking

14_perform_rrf.py

Implements Reciprocal Rank Fusion

15_test_vector_search.py

Evaluates vector search performance

16_test_bm25_search.py

Evaluates BM25 search performance

17_test_cohere_reranking_search.py

Evaluates Cohere reranking

18_test_rrf.py

Evaluates RRF with reranking performance

Performance Results

The project includes a comprehensive evaluation of different retrieval methods:

Method

Top 5 Accuracy

Top 10 Accuracy

Top 25 Accuracy

BM25

80.08%

84.46%

96.02%

Vector Search

81.27%

87.65%

94.82%

RRF with Reranking

94.82%

97.21%

99.60%

The hybrid approach combining BM25, vector search, and reranking (RRF) significantly outperforms individual methods, achieving nearly perfect accuracy with the top 25 results.

Key Components

  Azure AI Search: Provides both keyword-based and vector search capabilities
  
  Azure OpenAI: Generates text embeddings for semantic search
  
  Azure Functions: Custom function for PDF splitting
  
  Cohere Reranking: Improves result relevance with multilingual reranking
  
  Reciprocal Rank Fusion: Combines results from multiple search methods

Requirements

  Python 3.8+
  
  Azure subscription with:
      
      Azure AI Search service
      
      Azure OpenAI service
      
      Azure Blob Storage


Cohere API key
