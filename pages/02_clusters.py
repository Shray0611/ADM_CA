import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

st.set_page_config(page_title="Cluster Visualization", page_icon="🗺️", layout="wide")

if "corpus" not in st.session_state:
    from modules.state_manager import load_state
    st.session_state.corpus = load_state()

st.title("🗺️ Cluster Visualization")

if not st.session_state.corpus.get("processed", False):
    st.warning("Please upload and process documents first on the Upload page.")
    st.stop()

documents = st.session_state.corpus["documents"]
embeddings = st.session_state.corpus["embeddings"]
labels = st.session_state.corpus["labels"]
cluster_labels = st.session_state.corpus["cluster_labels"]

if len(documents) < 2:
    st.warning("Need at least 2 documents to visualize PCA.")
    st.stop()

# Perform PCA (2D only)
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Prepare DataFrame for Plotly
df = pd.DataFrame({
    'x': reduced_embeddings[:, 0],
    'y': reduced_embeddings[:, 1],
    'cluster_id': labels,
    'filename': [doc['filename'] for doc in documents],
    'preview': [doc['processed_text'][:100] + "..." for doc in documents]
})

# Apply String mapping for standard legend grouping
df['Cluster'] = df['cluster_id'].apply(lambda x: cluster_labels.get(x, f"Cluster {x}") if x != -1 else "Misc / Noise")

# Color mapping to handle Noise as grey
unique_clusters = df['Cluster'].unique()
color_discrete_map = {"Misc / Noise": "grey"}

# Plot
fig = px.scatter(
    df, x='x', y='y', color='Cluster',
    hover_data={'filename': True, 'Cluster': True, 'preview': True, 'x': False, 'y': False},
    title="2D PCA Projection of Document Embeddings",
    color_discrete_map=color_discrete_map,
    template="plotly_dark",
    width=900, height=600
)

# Enforce explicit styling
fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
fig.update_layout(legend_title_text='Clusters')

st.plotly_chart(fig, use_container_width=True)

with st.expander("How this works"):
    st.write("We use PCA (Principal Component Analysis) to reduce the 384-dimensional sentence embeddings down to 2 dimensions for simple visualization. Grey dots represent noise points identified by HDBSCAN.")
