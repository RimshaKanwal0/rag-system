## Retrieval-Augmented Generation (RAG) System
This project implements a Retrieval-Augmented Generation (RAG) pipeline from scratch. It demonstrates how to enhance a large language model's (LLM) responses by first retrieving relevant information from a custom knowledge base (a text document) and then using that context to generate a grounded and accurate answer.

The system is built using LangChain for document processing, ChromaDB as a vector store for efficient similarity search, and Hugging Face Transformers (GPT-2 and Sentence Transformers) for embedding and text generation.

##  Features
Document Loading & Text Splitting: Loads a text document and splits it into manageable, overlapping chunks for optimal context retrieval.
Vector Embeddings: Generates dense vector representations of text chunks using the BAAI/bge-small-en-v1.5 model for high-quality semantic search.
Vector Database: Stores and indexes embeddings in ChromaDB for fast and efficient similarity queries.
Contextual Retrieval: For a given user query, retrieves the most relevant text chunks from the knowledge base.
Answer Generation: Leverages a GPT-2 model to synthesize a coherent answer based on the retrieved context and the original query.

```bash
rag-system/
├── RAG system.py          # Main Python script containing the full RAG pipeline
├── sample1.txt           # Example text file used as the knowledge base
└── README.md             # This file
```
## Installation & Setup
**Prerequisites**
Python 3.7+
pip (Python package manager)

## Install Dependencies
Run the following commands to install all required Python libraries:
```bash
# Install core LangChain components
pip install -U langchain-community langchain

# Install vector database
pip install chromadb

# Install embedding and transformer models
pip install sentence-transformers transformers

# (Optional) If you encounter any issues, you may need:
pip install torch
```
## Usage
**1. Prepare Your Knowledge Base**
Place the text you want to use as your knowledge source in a file (e.g., sample1.txt) in the same directory as your script.

**2. Run the RAG Pipeline**
The main script (RAG system.py) performs the following steps sequentially:

Load Document: Uses TextLoader to read the text file.
Split Text: Uses RecursiveCharacterTextSplitter to create chunks of text with a specified size and overlap.
Create Embeddings: Initializes the HuggingFace embedding model to convert text chunks into vectors.
Initialize ChromaDB: Creates a persistent ChromaDB client and collection to store the vectors and their metadata.
Populate Database: Adds the generated embeddings and their corresponding text chunks (metadata) to the database.
Query the System: Takes a user query (e.g., "What is RAG?"), retrieves the most relevant chunks from the database, and constructs a prompt.
Generate Response: Feeds the prompt to the GPT-2 model to generate a contextual answer.

**To run the entire pipeline, simply execute the script:**

```bash
python "RAG system.py"
```
## Example Query
The script is hardcoded to query: "**What is RAG?**"
The system will retrieve context from sample1.txt related to this question and generate an answer based on that context.

## Code Overview / Key Components
Text Loader & Splitter: TextLoader, RecursiveCharacterTextSplitter
Embedding Model: HuggingFaceEmbeddings with BAAI/bge-small-en-v1.5
Vector Database: chromadb.Client
Language Model: GPT2LMHeadModel from Hugging Face Transformers
Retrieval & Generation Logic: The core RAG logic is implemented by querying ChromaDB and formatting the prompt for the GPT-2 model.

## Customization
Knowledge Source: Change the file_path in the TextLoader to point to your own .txt file.
Chunking: Adjust chunk_size and chunk_overlap in the RecursiveCharacterTextSplitter to tune context retrieval.
Embeddings: Swap the model_name in HuggingFaceEmbeddings to use a different embedding model (e.g., all-MiniLM-L6-v2).
Query: Modify the query_text variable to ask different questions of your document.
LLM: You can replace the GPT-2 model with a larger one (gpt2-medium, gpt2-large) or a different model entirely from Hugging Face.

## Important Notes

>This implementation uses GPT-2, which is a relatively small model. For more sophisticated and accurate generation, consider using a larger model like GPT-3.5/GPT-4 (via API), Llama 2, or Mistral.

>The current setup is designed for simplicity and runs in a single script. For production use, you would want to modularize the code and create a proper API.

>The ChromaDB persistence directory is set to ./chroma_storage. Delete this folder if you change your source document to force a re-creation of the vector store.

## License
This project is provided for educational and demonstration purposes.



