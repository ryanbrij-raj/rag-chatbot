# Local RAG Chatbot

A fully local Retrieval-Augmented Generation (RAG) chatbot for document Q&A — no API keys, no cost, runs entirely offline on your own machine.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange)

## Overview

Upload any document — PDF, text, or markdown — and ask questions about it. The app retrieves the most relevant passages using semantic search, then generates a grounded answer with a local open-source LLM, citing exactly which passages it used.

```
Document → Chunk → Embed (sentence-transformers) → FAISS Index
                                                         │
Question → Embed → Cosine Similarity Search → Top-k Chunks → Ollama LLM → Answer
```

## Features

- **100% local** — embeddings, vector search, and LLM inference all run on your machine
- **No API keys, no cost** — uses open-source models via [Ollama](https://ollama.com)
- **Source attribution** — every answer shows exactly which document chunks were used, with similarity scores
- **Configurable retrieval** — adjust chunk size, overlap, and number of sources per query
- **Multi-format support** — `.txt`, `.md`, `.pdf`

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector search | FAISS (exact cosine similarity) |
| LLM inference | Ollama (Llama 3.2 / Mistral / Phi-3) |
| UI | Streamlit |
| PDF parsing | pypdf |

## Setup

### 1. Install Ollama
Download from [ollama.com](https://ollama.com), then pull a model:
```bash
ollama pull llama3.2
```

### 2. Set up the Python environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

## Usage

1. Select your Ollama model in the sidebar
2. Upload a document
3. Click **Process Document**
4. Ask questions — answers come with expandable source citations

## Project Structure

```
.
├── app.py              # Streamlit UI
├── rag_engine.py        # Chunking, embedding, FAISS retrieval, Ollama generation
├── requirements.txt
└── README.md
```

## How It Works

1. **Chunking** — the document is split into overlapping word windows to preserve context across boundaries
2. **Embedding** — each chunk is converted into a 384-dimensional vector using a sentence-transformer model
3. **Indexing** — vectors are stored in a FAISS flat index for exact cosine similarity search
4. **Retrieval** — at query time, the question is embedded and compared against all chunk vectors to find the top-k most relevant
5. **Generation** — retrieved chunks are passed as context to a local LLM (via Ollama), which is instructed to answer only from the provided sources

## Possible Improvements

- Swap the flat FAISS index for `IndexIVFFlat` to scale to large document collections
- Add a cross-encoder reranker for higher-precision retrieval
- Persist the FAISS index to disk so it survives app restarts
- Support multiple documents simultaneously with per-chunk source tagging

## License

MIT
