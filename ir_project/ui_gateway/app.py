"""
Simple Streamlit UI to test the IR pipeline.
Run after starting the backend services.
"""

import requests
import streamlit as st

from shared.config import SERVICE_URLS

st.set_page_config(page_title="IR Search System", layout="wide")
st.title("Information Retrieval System")
st.caption("SOA demo interface for preprocessing, indexing, and retrieval.")

query = st.text_input("Search query")
model = st.selectbox(
    "Retrieval model",
    ["tf_idf", "bm25", "embedding", "hybrid_serial", "hybrid_parallel"],
)
top_k = st.slider("Top K", min_value=1, max_value=20, value=10)

col1, col2 = st.columns(2)
with col1:
    bm25_k1 = st.number_input("BM25 k1", min_value=0.1, max_value=3.0, value=1.5, step=0.1)
with col2:
    bm25_b = st.number_input("BM25 b", min_value=0.0, max_value=1.0, value=0.75, step=0.05)

if st.button("Search", type="primary"):
    if not query.strip():
        st.warning("Enter a query first.")
    else:
        payload = {
            "query": query,
            "model": model,
            "top_k": top_k,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
        }
        try:
            response = requests.post(
                f"{SERVICE_URLS['retrieval']}/search",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            st.error(f"Retrieval service is unavailable: {exc}")
        else:
            st.success(f"Returned {len(data['results'])} results in {data['elapsed_ms']} ms")
            for item in data["results"]:
                st.markdown(f"**#{item['rank']} | {item['doc_id']} | score={item['score']}**")
                if item.get("snippet"):
                    st.write(item["snippet"])

with st.expander("Service health"):
    for service_name, base_url in SERVICE_URLS.items():
        try:
            health = requests.get(f"{base_url}/health", timeout=5).json()
            st.write(f"{service_name}: {health.get('status', 'unknown')}")
        except requests.RequestException:
            st.write(f"{service_name}: offline")
