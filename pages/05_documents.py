import streamlit as st

st.set_page_config(page_title="Documents", page_icon="📄", layout="wide")

st.title("📄 Document Browser")

if not st.session_state.corpus.get("processed", False):
    st.warning("Please upload and process documents first on the Upload page.")
    st.stop()

documents = st.session_state.corpus["documents"]
labels = st.session_state.corpus["labels"]
cluster_labels = st.session_state.corpus["cluster_labels"]

unique_clusters = list(set(labels))
cluster_options = {cid: cluster_labels.get(cid, f"Cluster {cid}") for cid in unique_clusters}

selected_cid = st.selectbox(
    "Select a Cluster:",
    options=list(cluster_options.keys()),
    format_func=lambda x: f"{cluster_options[x]} (ID: {x})"
)

st.markdown("---")

count = 0
for idx, doc in enumerate(documents):
    if labels[idx] == selected_cid:
        count += 1
        with st.expander(f"📄 {doc['filename']}"):
            st.markdown("**Processed Content Preview:**")
            preview = doc['processed_text'][:1000]
            st.write(preview + ("..." if len(doc['processed_text']) > 1000 else ""))
            
            st.markdown("**Full Raw Text:**")
            st.text(doc['raw_text'])

if count == 0:
    st.info("No documents found for this cluster.")
