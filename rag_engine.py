"""
rag_engine.py — 100% local RAG pipeline. No API keys. No cost.

Stack:
  Embeddings : sentence-transformers  (runs on your CPU, free)
  Vector DB  : FAISS                  (in-memory, free)
  LLM        : Ollama                 (local open-source models, free)

Prereqs:
  1. Install Ollama from https://ollama.com
  2. Run: ollama pull llama3.2        (or mistral, gemma2, phi3, etc.)
  3. pip install -r requirements.txt
"""

import re
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


OLLAMA_URL = "http://localhost:11434/api/chat"   # Ollama's default local endpoint


class RAGEngine:
    def __init__(
        self,
        embed_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama3.2",             # any model you've pulled with `ollama pull`
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ):
        """
        Args:
            embed_model:   Sentence-transformers model. Runs locally, ~80 MB download once.
            llm_model:     Ollama model name. Must be pulled first: `ollama pull <name>`
                           Good options: llama3.2, mistral, gemma2, phi3, llama3.2:1b (tiny)
            chunk_size:    Target words per chunk.
            chunk_overlap: Words shared between consecutive chunks.
        """
        print(f"Loading embedding model '{embed_model}'…")
        self.embed_model = SentenceTransformer(embed_model)
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.chunks: list[str] = []
        self.index: faiss.Index | None = None

        self._check_ollama()

    # startup checks

    def _check_ollama(self):
        """Warn early if Ollama isn't running."""
        try:
            r = requests.get("http://localhost:11434", timeout=3)
            r.raise_for_status()
        except Exception:
            raise RuntimeError(
                "Ollama is not running.\n"
                "  1. Install from https://ollama.com\n"
                "  2. Start it (it runs as a background app)\n"
                f"  3. Run: ollama pull {self.llm_model}"
            )

    # document loadign

    def load_text(self, text: str) -> int:
        """Chunk text, embed all chunks, build FAISS index. Returns chunk count."""
        self.chunks = self._chunk(text)
        if not self.chunks:
            raise ValueError("Document produced no chunks — is it empty?")

        print(f"Embedding {len(self.chunks)} chunks…")
        embeddings = self.embed_model.encode(
            self.chunks, show_progress_bar=True, convert_to_numpy=True
        ).astype(np.float32)

        faiss.normalize_L2(embeddings)          # normalise → inner product = cosine sim

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)     # exact cosine search
        self.index.add(embeddings)

        print(f"✓ Indexed {len(self.chunks)} chunks (dim={dim})")
        return len(self.chunks)

    def load_pdf(self, path: str) -> int:
        """Extract text from a PDF and load it."""
        try:
            import pypdf
        except ImportError:
            raise ImportError("Run: pip install pypdf")
        reader = pypdf.PdfReader(path)
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        return self.load_text(text)

    # retrieval

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """
        Find the top-k most relevant chunks for `query` using cosine similarity.

        Returns:
            [{"chunk": str, "index": int, "score": float}, ...]
        """
        if self.index is None:
            raise RuntimeError("No document loaded. Call load_text() first.")

        q_emb = self.embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_emb)

        scores, indices = self.index.search(q_emb, k)

        return [
            {"chunk": self.chunks[idx], "index": int(idx), "score": float(score)}
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

    # gen via ollama

    def answer(self, query: str, k: int = 3) -> tuple[str, list[dict]]:
        """
        Full RAG pipeline: retrieve → build prompt → generate with local LLM.

        Returns:
            (answer_text, retrieved_chunks)
        """
        hits = self.retrieve(query, k)

        context = (
            "\n\n---\n\n".join(
                f"[Source {i + 1}]\n{h['chunk']}" for i, h in enumerate(hits)
            )
            if hits
            else "No relevant document passages found."
        )

        prompt = (
            f"You are a helpful assistant. Answer the user's question using ONLY "
            f"the provided document sources. Cite sources as [Source N]. "
            f"If the answer is not in the sources, say so — do not guess.\n\n"
            f"Document sources:\n\n{context}\n\n"
            f"---\n\nQuestion: {query}"
        )

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,    # local models can be slow on CPU — give them time
        )
        response.raise_for_status()
        answer_text = response.json()["message"]["content"]

        return answer_text, hits

    # internals

    def _chunk(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        words = text.split()
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            if i + self.chunk_size >= len(words):
                break
        return chunks