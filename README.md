# Retrieval-Augmented Generation (RAG) System

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline built using `LangChain`, `ChromaDB`, and `GPT-2`. The system retrieves relevant context from a document, embeds it, and uses a language model to generate answers to user queries.

## Features

- **Text Splitting:** Splits input text into smaller, overlapping chunks for better context retrieval.
- **Embeddings:** Creates embeddings using `HuggingFaceEmbeddings` for efficient similarity search.
- **Document Retrieval:** Stores and retrieves relevant document chunks using `ChromaDB`.
- **Query Response Generation:** Generates responses based on retrieved context using GPT-2.

---

## Installation

### Prerequisites

- Python 3.13
- Google Colab (if running in the cloud)

### Install Required Libraries

Run the following commands to install the necessary libraries:

```bash
pip install -U langchain-community
pip install -U langchain
pip install chromadb
pip install sentence-transformers
