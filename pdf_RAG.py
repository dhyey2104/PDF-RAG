import os
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
from pypdf import *
import ast
import re
import streamlit as st
from rank_bm25 import BM25Okapi

# ==============================
# 1. LOAD ENV VARIABLES
# ==============================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# ==============================
# 2. INIT GEMINI CLIENT
# ==============================
client = genai.Client(api_key=api_key)

# ==============================
# 3. LOAD EMBEDDING MODEL (LOCAL - FREE)
# ==============================
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# RERANKER MODEL
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def read_pdf(filepath):
    print(filepath)
    reader = PdfReader(filepath)
    text = ""

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"

    return text

def chunk_text(text):
    # split by numbered questions OR headings
    pattern = r'(\d+\.\s.*?\?)'

    parts = re.split(pattern, text)

    chunks = []
    for i in range(1, len(parts), 2):
        question = parts[i]
        answer = parts[i+1] if i+1 < len(parts) else ""

        chunk = (question + " " + answer).strip()

        if chunk:
            chunks.append(chunk)

    # clean + dedupe
    chunks = list(set([" ".join(c.split()) for c in chunks]))

    return chunks
def create_embeddings(chunks):
    return embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )


def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

# ==============================
# BM25 (KEYWORD SEARCH)
# ==============================
def create_bm25_index(chunks):
    tokenized = [c.lower().split() for c in chunks]
    return BM25Okapi(tokenized)


def bm25_search(query, bm25, chunks, k=5):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in ranked]


def retrieve(query,index,chunks, k=3, threshold=0.3):
    query_embeddings = embedding_model.encode( [query],convert_to_numpy=True,normalize_embeddings=True)

    distance, indices = index.search(query_embeddings,k)

    result = []
    for i,score in zip(indices[0],distance[0]):
        if score > threshold:
            result.append(chunks[i])

    return result


# ==============================
# MULTI QUERY GENERATION
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

    text = response.text.strip()

    try:
        queries = ast.literal_eval(text)  # convert string → list
        return [q.strip() for q in queries if isinstance(q, str)]
    except:
        # fallback (if parsing fails)
        return [question]


# ==============================
# RERANK FUNCTION
# ==============================
def rerank_chunks(query, chunks):
    # ✅ CLEAN INPUTS
    clean_chunks = [
        str(chunk) for chunk in chunks
        if isinstance(chunk, str) and chunk.strip()
    ]

    if not clean_chunks:
        return []

    pairs = [(query, chunk) for chunk in clean_chunks]

    scores = reranker.predict(pairs)

    scored_chunks = list(zip(clean_chunks, scores))

    # sort by reranker score
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    print("\n[DEBUG] Reranker Scores:")
    for chunk, score in scored_chunks:
        print(f"{score:.4f} -> {chunk[:80]}")

    return [c[0] for c in scored_chunks]

# ==============================
# MULTI QUERY RETRIEVAL
# ==============================
def multi_query_retrieve(question, index, chunks, k=3):
    queries = generate_queries(question)

    print("\n[Generated Queries]:")
    for q in queries:
        print("-", q)

    all_chunks = []
    seen = set()

    for q in queries:
        query_embedding = embedding_model.encode(
            [q],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        distances, indices = index.search(query_embedding, k)

        print(f"\n[DEBUG SCORES for query: {q}]")

        for i, score in zip(indices[0], distances[0]):
            print(f"Score: {score:.4f} -> {chunks[i][:80]}")

            if score > 0.45:
                chunk = chunks[i]
                if chunk not in seen:
                    all_chunks.append((chunk))
                    seen.add(chunk)
     #APPLY RERANKING HERE
    if not all_chunks:
        return []

    reranked_chunks = rerank_chunks(question, all_chunks)

    #TAKE TOP 3 AFTER RERANK
    return reranked_chunks[:3]


# ==============================
# HYBRID RETRIEVAL
# ==============================
def hybrid_retrieve(question, index, bm25, chunks):
    queries = generate_queries(question)

    print("\n[Generated Queries]:", queries)

    all_chunks = set()

    for q in queries:
        # FAISS SEARCH
        q_emb = embedding_model.encode(
            [q],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        distances, indices = index.search(q_emb, 5)

        print(f"\n[FAISS for: {q}]")
        for i, score in zip(indices[0], distances[0]):
            print(f"{score:.4f} -> {chunks[i][:80]}")
            if score > 0.3:
                all_chunks.add(chunks[i])

        # BM25 SEARCH
        bm25_results = bm25_search(q, bm25, chunks, k=5)

        print(f"\n[BM25 for: {q}]")
        for c in bm25_results:
            print(c[:80])
            all_chunks.add(c)

    if not all_chunks:
        return []

    # RERANK FINAL
    return rerank_chunks(question, list(all_chunks))



def ask_llm(context, question):
    prompt = f"""
    You are a helpful assistant.

    Use ONLY the given context.

    If the answer is present indirectly (like count or list),
    infer carefully from the context.

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

    return response.text

def find_most_relevant_chunk(answer, chunks):
    if not chunks:
        return None

    # embed answer
    answer_emb = embedding_model.encode(
        [answer],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # embed chunks
    chunk_embs = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # cosine similarity
    scores = np.dot(chunk_embs, answer_emb.T).flatten()

    best_idx = np.argmax(scores)

    return chunks[best_idx], scores[best_idx]

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="PDF RAG Chat", layout="wide")

st.title("📄 Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    if "index" not in st.session_state:
        with st.spinner("Processing PDF..."):
            text = read_pdf(uploaded_file)

            chunks = chunk_text(text)
            embeddings = create_embeddings(chunks)
            index = create_faiss_index(embeddings)

            bm25 = create_bm25_index(chunks)

            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.bm25 = bm25

        st.success("PDF processed!")

    query = st.text_input("Ask a question")

    if query:
        relevant_chunks = hybrid_retrieve(
            query,
            st.session_state.index,
            st.session_state.bm25,
            st.session_state.chunks
        )
        if not relevant_chunks:
            st.warning("I don't know")
        else:
            context = "\n\n".join(relevant_chunks)
            answer = ask_llm(context, query)

            st.subheader("Answer")
            st.write(answer)

            best_chunk, score = find_most_relevant_chunk(answer, relevant_chunks)

            st.subheader("📌 Most Relevant Chunk (Used for Answer)")
            st.write(best_chunk)
            st.caption(f"Relevance Score: {score:.4f}")

            with st.expander("See Context"):
                for c in relevant_chunks:
                    st.write(c)
