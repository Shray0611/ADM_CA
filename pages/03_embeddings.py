import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from modules.embedding import get_embeddings

def make_unique(names):
    count = defaultdict(int)
    result = []
    for name in names:
        count[name] += 1
        if count[name] == 1:
            result.append(name)
        else:
            result.append(f"{name}_{count[name]}")
    return result

st.set_page_config(page_title="Embeddings Explorer", page_icon="🧠", layout="wide")

if "corpus" not in st.session_state:
    from modules.state_manager import load_state
    st.session_state.corpus = load_state()

st.title("🧠 Embeddings Explorer")

if not st.session_state.corpus.get("processed", False):
    st.warning("Please upload and process documents first on the Upload page.")
    st.stop()

documents = st.session_state.corpus["documents"]
embeddings = st.session_state.corpus["embeddings"]

filenames = [doc['filename'] for doc in documents]
unique_names = make_unique(filenames)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Embeddings Table (First 10 Dimensions)")
    # Show first 10 dimensions only
    target_dims = min(embeddings.shape[1], 10)
    df_emb = pd.DataFrame(embeddings[:, :target_dims], columns=[f"Dim_{i}" for i in range(target_dims)])
    df_emb.insert(0, "Filename", filenames)
    st.dataframe(df_emb, use_container_width=True)

with col2:
    st.subheader("2. Similarity Matrix")
    if len(documents) > 1:
        sim_matrix = cosine_similarity(embeddings)
        # Limit visual to first 15x15 if too large, simply for speed/rendering constraints
        limit = min(sim_matrix.shape[0], 25)
        
        sim_df = pd.DataFrame(sim_matrix[:limit, :limit], 
                              index=unique_names[:limit], 
                              columns=unique_names[:limit])
                              
        fig = px.imshow(sim_df, 
                        zmin=0, zmax=1.0, 
                        color_continuous_scale='Viridis',
                        title="Cosine Similarity Matrix (Truncated to first 25 docs)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need multiple documents to generate similarity matrix.")

st.divider()

st.subheader("3. Semantic Query Explorer")
query = st.text_input("Enter a topic or phrase to find similar documents:")

if query:
    query_emb = get_embeddings([query])
    # Compute similarity between query and all documents
    sims = cosine_similarity(query_emb, embeddings).flatten()
    
    # Get top 5 matches
    top_indices = sims.argsort()[-5:][::-1]
    
    st.markdown("**Top Matching Documents:**")
    for idx in top_indices:
        score = sims[idx]
        if score > 0.1: # Trivial threshold
            st.info(f"**{filenames[idx]}** — Similarity: {score:.3f}")
            with st.expander("Preview Document"):
                st.write(documents[idx]['processed_text'][:500] + "...")
