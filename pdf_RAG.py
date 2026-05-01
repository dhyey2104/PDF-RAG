import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
import os
import re
import ast
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
import chromadb

# ==============================
# ENV SETUP
# ==============================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ==============================
# PDF READING
# ==============================
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

# ==============================
# CHUNKING
# ==============================
def chunk_text(text):
    pattern = r'(\d+\.\s.*?\?)'
    parts = re.split(pattern, text)

    chunks = []
    for i in range(1, len(parts), 2):
        q = parts[i]
        a = parts[i+1] if i+1 < len(parts) else ""
        chunk = (q + " " + a).strip()
        if chunk:
            chunks.append(" ".join(chunk.split()))

    return list(set(chunks))

# ==============================
# CHROMA DB
# ==============================
def create_chroma_db(chunks):
    client = chromadb.Client(
        settings=chromadb.Settings(
            persist_directory="./chroma_db"
        )
    )

    collection = client.get_or_create_collection(name="pdf_rag")

    # 🔥 CLEAR OLD DATA (IMPORTANT)
    if collection.count() > 0:
        collection.delete(where={})

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).tolist()

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks
    )

    return collection
def chroma_search(query, collection, k=5):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )

    return results["documents"][0]

# ==============================
# BM25
# ==============================
def create_bm25_index(chunks):
    tokenized = [c.lower().split() for c in chunks]
    return BM25Okapi(tokenized)

def bm25_search(query, bm25, chunks, k=5):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in ranked]

# ==============================
# MULTI QUERY
# ==============================
def generate_queries(question):
    prompt = f"""
    Generate 3 rephrased versions of the question.

    Question: {question}

    Return ONLY a Python list like:
    ["q1", "q2", "q3"]
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"temperature": 0.7}
    )

    try:
        return ast.literal_eval(response.text.strip())
    except:
        return [question]


def rewrite_query_with_history(question, history):
    if not history:
        return question

    history_text = ""
    for q, a in history[-3:]:  # last 3 interactions
        history_text += f"User: {q}\nAssistant: {a}\n"

    prompt = f"""
    You are a query rewriting assistant.

    Convert the follow-up question into a standalone question.

    Chat History:
    {history_text}

    Follow-up Question:
    {question}

    Standalone Question:
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"temperature": 0.2}
    )

    return response.text.strip()


# ==============================
# RERANKING
# ==============================
def rerank_chunks(query, chunks):
    chunks = [c for c in chunks if isinstance(c, str) and c.strip()]

    pairs = [(query, c) for c in chunks]
    scores = reranker.predict(pairs)

    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    print("\n[DEBUG] Reranker Scores:")
    for c, s in scored:
        print(f"{s:.4f} -> {c[:80]}")

    return [c[0] for c in scored[:3]]

# ==============================
# HYBRID RETRIEVAL
# ==============================
def hybrid_retrieve(question, collection, bm25, chunks):
    queries = generate_queries(question)

    print("\n[Generated Queries]:", queries)

    all_chunks = set()

    for q in queries:
        # 🔵 CHROMA SEARCH
        chroma_results = chroma_search(q, collection, k=5)

        print(f"\n[CHROMA for: {q}]")
        for c in chroma_results:
            print(c[:80])
            all_chunks.add(c)

        # 🟢 BM25 SEARCH
        bm25_results = bm25_search(q, bm25, chunks, k=5)

        print(f"\n[BM25 for: {q}]")
        for c in bm25_results:
            print(c[:80])
            all_chunks.add(c)

    if not all_chunks:
        return []

    return rerank_chunks(question, list(all_chunks))

# ==============================
# LLM
# ==============================
def ask_llm(context, question):
    prompt = f"""
    You are a helpful assistant.

    Use ONLY the given context.

    If answer requires counting or listing, infer carefully.

    Context:
    {context}

    Question:
    {question}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"temperature": 0.2}
    )

    return response.text.strip()

# ==============================
# EXPLAINABILITY
# ==============================
def find_most_relevant_chunk(answer, chunks):
    if not chunks:
        return None, 0

    answer_emb = embedding_model.encode(
        [answer],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    chunk_embs = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = np.dot(chunk_embs, answer_emb.T).flatten()
    best_idx = np.argmax(scores)

    return chunks[best_idx], scores[best_idx]

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="Hybrid RAG Chat", layout="wide")

st.title("📄 Chat with your PDF (Hybrid RAG + ChromaDB)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

st.subheader("💬 Chat")

for q, a in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Assistant:** {a}")

if uploaded_file:
    if "collection" not in st.session_state:
        with st.spinner("Processing PDF..."):
            text = read_pdf(uploaded_file)

            chunks = chunk_text(text)
            collection = create_chroma_db(chunks)
            bm25 = create_bm25_index(chunks)

            st.session_state.collection = collection
            st.session_state.chunks = chunks
            st.session_state.bm25 = bm25

        st.success("PDF processed!")

    query = st.text_input("Ask a question")

    if query:
        # 🔥 NEW: rewrite query using history
        rewritten_query = rewrite_query_with_history(
            query,
            st.session_state.chat_history
        )

        st.caption(f"🔍 Rewritten Query: {rewritten_query}")
        st.info(f"📡 Retrieval Query Used → {rewritten_query}")

        relevant_chunks = hybrid_retrieve(
            rewritten_query,
            st.session_state.collection,
            st.session_state.bm25,
            st.session_state.chunks
        )

        if relevant_chunks:
            st.success("📦 Context retrieved from PDF (RAG used)")
        else:
            st.error("⚠️ No context retrieved → Answer may be guessed")
        if not relevant_chunks:
            st.warning("I don't know")
        else:
            context = "\n\n".join(relevant_chunks)
            answer = ask_llm(context, query)

            st.subheader("Answer")
            st.write(answer)
            # 🔥 SAVE HISTORY
            st.session_state.chat_history.append((query, answer))

            best_chunk, score = find_most_relevant_chunk(answer, relevant_chunks)

            st.subheader("📌 Source of Answer")

            st.markdown("**Most Relevant Chunk Used:**")
            st.write(best_chunk)

            st.caption(f"Relevance Score: {score:.4f}")

            if score > 0.5:
                st.caption("✔ This chunk strongly influenced the answer")
            else:
                st.caption("⚠ Weak match → answer may include assumptions")
            with st.expander("All Retrieved Chunks"):
                for c in relevant_chunks:
                    st.write(c)
