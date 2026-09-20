import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Hybrid RAG (FAISS + Neo4j)", layout="wide")
st.title("📚 Hybrid RAG — FAISS + Neo4j AuraDB + Guardrails")

with st.sidebar:
    st.header("1. Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file and st.button("Process & Index"):
        with st.spinner("Extracting, chunking, embedding, and building the knowledge graph..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=600)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(
                        f"Indexed **{data['doc_name']}** — "
                        f"{data['num_pages']} pages, {data['num_chunks']} chunks, "
                        f"{data['num_entities']} entities extracted."
                    )
                else:
                    st.error(f"Upload failed: {resp.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach backend at {BACKEND_URL}: {e}")

    st.divider()
    st.caption(f"Backend: {BACKEND_URL}")

st.header("2. Ask a question")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Your question about the uploaded document(s)")
ask = st.button("Ask")

if ask and question.strip():
    with st.spinner("Retrieving (vector + graph) and generating a grounded answer..."):
        try:
            resp = requests.post(f"{BACKEND_URL}/query", json={"question": question}, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.history.append((question, data))
            else:
                st.error(f"Query failed: {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach backend at {BACKEND_URL}: {e}")

for q, data in reversed(st.session_state.history):
    st.markdown(f"**Q: {q}**")
    if data.get("blocked"):
        st.warning(f"🚫 Blocked by guardrails: {data.get('block_reason')}")
    else:
        st.markdown(data["answer"])
        with st.expander(f"Sources ({len(data['sources'])})"):
            for s in data["sources"]:
                st.markdown(
                    f"- `{s['retrieval_type']}` **{s['doc_name']}**, page {s['page']}\n\n  {s['text']}..."
                )
    st.divider()
