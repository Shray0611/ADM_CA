import streamlit as st
import time
from modules.ingestion import process_uploaded_files
from modules.embedding import get_embeddings
from modules.clustering import cluster_embeddings
from modules.labeling import extract_cluster_labels
from modules.summarizer import get_cluster_summaries

st.set_page_config(page_title="Upload & Process", page_icon="📤", layout="wide")

st.title("📤 Upload Documents")
st.markdown("Upload multiple `TXT`, `PDF`, or `DOCX` files. The system will process everything in one run.")

uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=['txt', 'pdf', 'docx'])

if uploaded_files:
    if st.button("Run Pipeline", type="primary", use_container_width=True):
        
        start_time = time.time()
        
        # UI Status indicators
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            # 1. Parsing
            status_text.text("📂 Step 1/5: Parsing & Light Preprocessing...")
            documents = process_uploaded_files(uploaded_files)
            if not documents:
                st.error("No valid text extracted from the files.")
                st.stop()
            progress_bar.progress(20)
            
            # 2. Embedding
            status_text.text("🧠 Step 2/5: Batch Embedding (SentenceTransformer)...")
            texts = [doc["processed_text"] for doc in documents]
            embeddings = get_embeddings(texts)
            progress_bar.progress(40)
            
            # 3. Clustering
            status_text.text("🗺️ Step 3/5: Clustering (HDBSCAN)...")
            labels = cluster_embeddings(embeddings)
            progress_bar.progress(60)
            
            # 4. Labeling
            status_text.text("🏷️ Step 4/5: Extracting TF-IDF Labels...")
            cluster_labels = extract_cluster_labels(documents, labels)
            progress_bar.progress(80)
            
            # 5. Summarization
            status_text.text("📝 Step 5/5: Generating LLaMA Summaries (Groq)...")
            # Create mapping
            cluster_docs = {}
            for doc_dict, lbl in zip(documents, labels):
                if lbl not in cluster_docs:
                    cluster_docs[lbl] = []
                cluster_docs[lbl].append({"filename": doc_dict['filename'], "text": doc_dict['processed_text']})
                
            summaries = get_cluster_summaries(cluster_docs, cluster_labels)
            progress_bar.progress(100)
            
            # Save to Session State
            st.session_state.corpus = {
                "documents": documents,
                "embeddings": embeddings,
                "labels": labels.tolist(),
                "cluster_labels": cluster_labels,
                "summaries": summaries,
                "processed": True
            }
            
            elapsed = time.time() - start_time
            status_text.text(f"✅ Pipeline completed in {elapsed:.2f} seconds!")
            st.success("Processing complete! You can now explore the results in the sidebar.")
            
        except Exception as e:
            status_text.text("❌ Error occurred during processing.")
            st.error(f"Details: {str(e)}")
            progress_bar.empty()
