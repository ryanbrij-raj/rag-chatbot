"""
app.py — Streamlit UI for the fully local RAG chatbot.

Run with:
    streamlit run app.py
"""

import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="Local RAG Chatbot", page_icon="🦙", layout="wide")

# session state

if "engine" not in st.session_state:
    st.session_state.engine = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# sidebar

with st.sidebar:
    st.title("🦙 Local RAG Chatbot")
    st.caption("100% local · no API keys · no cost")
    st.divider()

    st.header("Model")
    llm_model = st.selectbox(
        "Ollama model",
        ["llama3.2", "llama3.2:1b", "mistral", "gemma2", "phi3", "llama3.1"],
        help="Must be pulled first: `ollama pull <model>`",
    )
    st.caption(
        "First time? Run in your terminal:\n"
        f"```\nollama pull {llm_model}\n```"
    )

    st.header("Document")
    uploaded = st.file_uploader("Upload a file", type=["txt", "pdf", "md"])

    st.subheader("Settings")
    chunk_size = st.slider("Chunk size (words)", 100, 500, 300, 50)
    chunk_overlap = st.slider("Chunk overlap (words)", 0, 100, 50, 10)
    top_k = st.slider("Sources per query", 1, 6, 3)

    if st.button("Process Document", type="primary", disabled=uploaded is None, use_container_width=True):
        with st.spinner("Processing…"):
            if uploaded.type == "application/pdf":
                try:
                    import pypdf
                except ImportError:
                    st.error("Run: pip install pypdf")
                    st.stop()
                reader = pypdf.PdfReader(uploaded)
                text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
            else:
                text = uploaded.read().decode("utf-8", errors="replace")

            if not text.strip():
                st.error("No text found in the file.")
                st.stop()

            try:
                engine = RAGEngine(
                    llm_model=llm_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                n = engine.load_text(text)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

            st.session_state.engine = engine
            st.session_state.doc_name = uploaded.name
            st.session_state.messages = []

        st.success(f"Indexed **{n}** chunks from {uploaded.name}")

    if st.session_state.engine:
        st.divider()
        eng = st.session_state.engine
        st.markdown(f"**File:** {st.session_state.doc_name}")
        st.markdown(f"**Chunks:** {len(eng.chunks)}")
        st.markdown(f"**LLM:** {eng.llm_model} (local)")
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# chat

st.header("Chat with your document")

if st.session_state.engine is None:
    st.info(
        "👈 Upload a document and click **Process Document** to get started.\n\n"
        "Make sure Ollama is installed and running — see the README."
    )
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("hits"):
            with st.expander(f"📎 {len(msg['hits'])} sources retrieved"):
                for i, h in enumerate(msg["hits"]):
                    col1, col2 = st.columns([3, 1])
                    col1.markdown(f"**Source {i+1}** · chunk #{h['index']+1}")
                    col2.markdown(
                        f"<div style='text-align:right;color:#10B981;font-weight:600'>"
                        f"sim {h['score']:.2f}</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(h["chunk"][:400] + ("…" if len(h["chunk"]) > 400 else ""))
                    if i < len(msg["hits"]) - 1:
                        st.divider()

if query := st.chat_input("Ask a question about your document…"):
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking with {st.session_state.engine.llm_model}…"):
            try:
                answer, hits = st.session_state.engine.answer(query, k=top_k)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        st.write(answer)

        if hits:
            with st.expander(f"📎 {len(hits)} sources retrieved"):
                for i, h in enumerate(hits):
                    col1, col2 = st.columns([3, 1])
                    col1.markdown(f"**Source {i+1}** · chunk #{h['index']+1}")
                    col2.markdown(
                        f"<div style='text-align:right;color:#10B981;font-weight:600'>"
                        f"sim {h['score']:.2f}</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(h["chunk"][:400] + ("…" if len(h["chunk"]) > 400 else ""))
                    if i < len(hits) - 1:
                        st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer, "hits": hits})