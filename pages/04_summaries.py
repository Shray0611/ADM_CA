import streamlit as st

st.set_page_config(page_title="Cluster Summaries", page_icon="📋", layout="wide")

st.title("📋 Cluster Summaries")

if not st.session_state.corpus.get("processed", False):
    st.warning("Please upload and process documents first on the Upload page.")
    st.stop()

documents = st.session_state.corpus["documents"]
labels = st.session_state.corpus["labels"]
cluster_labels = st.session_state.corpus["cluster_labels"]
summaries = st.session_state.corpus["summaries"]

# Get unique clusters, remove noise for main display natively but handle -1 if present
unique_clusters = list(set(labels))

if not summaries:
    st.warning("Summaries have not been generated. Please restart the pipeline or ensure Groq API key is valid.")
    st.stop()

for cid in unique_clusters:
    # Get count and docs
    count = labels.count(cid)
    doc_names = [d["filename"] for i, d in enumerate(documents) if labels[i] == cid]
    
    label_text = cluster_labels.get(cid, "Unnamed")
    
    # JSON keys from LLM may be strings
    summary_data = summaries.get(str(cid), summaries.get(cid, {}))
    
    title = summary_data.get("title", label_text)
    summary_text = summary_data.get("summary", "No summary available.")
    insights = summary_data.get("insights", [])
    
    with st.container():
        st.subheader(f"Cluster {cid}: {title}")
        
        # Badges & Metadata
        col1, col2 = st.columns([3, 2])
        with col1:
            words = label_text.split(" · ")
            badges_html = "".join([f"<span style='background-color:#4A4A4A; padding:4px 8px; border-radius:4px; margin-right:5px; font-size: 0.9em'>{w}</span>" for w in words])
            st.markdown(badges_html, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"**Documents ({count}):** " + ", ".join([f"`{n}`" for n in doc_names]))
            
        st.markdown("**Summary:**")
        st.write(summary_text)
        
        if insights:
            st.markdown("**Insights:**")
            for insight in insights:
                st.markdown(f"- {insight}")
                
        st.divider()
