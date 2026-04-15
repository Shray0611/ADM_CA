import streamlit as st
from dotenv import load_dotenv

# Load env variables globally
load_dotenv()

st.set_page_config(
    page_title="Document Clustering & Summarization",
    page_icon="🚀",
    layout="wide"
)

if "corpus" not in st.session_state:
    st.session_state.corpus = {
        "documents": [],
        "embeddings": None,
        "labels": [],
        "cluster_labels": {},
        "summaries": {},
        "processed": False
    }

st.title("🚀 Fast Document Clustering & Summarization")

st.markdown("""
Welcome to the Semantic Document Clustering App. 

This system uses:
- **`SentenceTransformers`** (`all-MiniLM-L6-v2`) for fast batch embeddings.
- **`HDBSCAN`** for robust density-based clustering.
- **`TF-IDF`** for lightweight, accurate cluster labeling.
- **`Groq LLaMA-3.3-70B`** for intelligent cluster summarization.

### Navigation 
Please use the sidebar to navigate through the steps:
1. **Upload**: Ingest and process documents.
2. **Clusters**: Visualize clusters in 2D space.
3. **Embeddings Explorer**: Inspect embeddings and semantics.
4. **Summaries**: Read LLM generative insights for each cluster.
5. **Documents**: Browse assigned documents by cluster.
""")

# Display status of current process
if st.session_state.corpus["processed"]:
    num_docs = len(st.session_state.corpus["documents"])
    num_clusters = len([c for c in set(st.session_state.corpus["labels"]) if c != -1])
    
    st.success(f"✅ System is currently maintaining {num_docs} processed documents across {num_clusters} clusters.")
else:
    st.info("System is waiting for documents. Please proceed to the **Upload** page.")
