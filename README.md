# 📄 PDF RAG Question Answering System

## 🚀 Overview
This project implements a Retrieval-Augmented Generation (RAG) system that allows users to perform question-answering over PDF documents.  
It extracts text from PDFs,split it into meaningful chunks, converts them into embeddings, and stores them in a FAISS Index for efficient retrieval.  
Relevant document chunks are fetched based on user queries and passed to an LLM to generate accurate, context-aware answers.  
This approach ensures responses are grounded in the actual document content rather than relying solely on model knowledge.

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```
### 2️⃣ Create a virtual environment
```bash
python3 -m venv venv
```
### 3️⃣ Activate the virtual environment

On Linux / Mac:
```bash
source venv/bin/activate
```
On Windows:
```bash
venv\Scripts\activate
```
### 4️⃣ Install dependencies

```bash
pip install -r requirements.txt
```
### 5️⃣ Setup environment variables

Create a .env file in the root directory and add:

```bash
GEMINI_API_KEY=<your LLM generated api key>
```
### 6️⃣ Run the application

```bash
python3 pdf_rag.py
```

## 📂 Project Structure
```bash
.
├── pdf_rag.py        # Main script for RAG pipeline
├── requirements.txt  # Project dependencies
├── .env              # Environment variables
└── README.md         # Documentation
```

## 💡 Features
- 📄 PDF text extraction using PyPDF
- 🧩 Smart chunking based on question-answer patterns
- 🔍 Hybrid search (Semantic + Keyword)
  - FAISS for dense vector similarity
  - BM25 for keyword-based retrieval
- 🔁 Multi-query generation using LLM for better recall
- 🧠 Cross-encoder reranking for high-precision results
- 🎯 Context-aware answer generation using Gemini LLM
- 📌 Displays most relevant chunk used for answering
- 💬 Interactive UI built with Streamlit

## 🛠️ Tech Stack
- Python
- Streamlit (UI)
- FAISS (Vector similarity search)
- BM25 (Keyword search - rank_bm25)
- Sentence Transformers (Embeddings)
- Cross Encoder (Reranking)
- Gemini API (LLM for QA + query expansion)
- NumPy
- PyPDF (PDF parsing)

## 📌 Future Improvements
- Add persistent storage for embeddings (avoid reprocessing PDFs)
- Support multiple PDFs and document collections
- Add chat history / conversational memory
- Improve chunking using semantic or token-based splitting
- Add API layer (FastAPI/Django) for production use
- Deploy as a full-stack app (frontend + backend)
- Add evaluation metrics (precision, recall, retrieval accuracy)
- Optimize hybrid scoring (weighted fusion of FAISS + BM25)

## 🧠 How It Works
1. PDF is uploaded and text is extracted
2. Text is split into meaningful chunks
3. Embeddings are generated using Sentence Transformers
4. Two retrieval systems are built:
   - FAISS (semantic search)
   - BM25 (keyword search)
5. User query is expanded into multiple queries using LLM
6. Hybrid retrieval fetches relevant chunks
7. Cross-encoder reranks results for accuracy
8. Top chunks are passed to Gemini LLM for answer generation
9. Most relevant chunk is displayed for transparency
