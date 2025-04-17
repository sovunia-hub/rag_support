# RAG System for Contact Center
This project is a **RAG (Retrieval-Augmented Generation)** system designed to automate customer interactions in a contact center. The system leverages a combination of modern technologies to improve query response quality and reduce latency.

## Technologies Used

- **YandexGPT-5** integrated via HuggingFace: A powerful text generation model for creating responses to customer queries.
- **FAISS**: A library for efficient similarity search, used to quickly retrieve relevant information from large datasets.
- **Sentence-Transformers**: Models for generating sentence embeddings, helping to accurately find similar queries in the database and improving the search process.

## Key Features

1. **Real-time Query Responses**: The system generates responses to user queries using YandexGPT-5, augmented with information retrieved via FAISS.
2. **Database Search**: FAISS enables fast retrieval of the most relevant documents to respond to queries, based on semantic search.
3. **Conversation Support**: Sentence-Transformers helps the system process queries and provide highly accurate answers.

## How It Works

1. When a user submits a query, the system analyzes it using Sentence-Transformers to extract embeddings.
2. FAISS uses these embeddings to search for the most relevant documents in the database.
3. The found documents are used to generate an accurate response through YandexGPT-5.

## Installation

To install and run the project on your machine, follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/sovunia-hub/rag_support.git
cd rag_support
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the application
```bash
python desktop_app.py
```
