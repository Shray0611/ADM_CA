import streamlit as st
import time
import numpy as np
from modules.ingestion import process_uploaded_files
from modules.embedding import get_embeddings
from modules.clustering import cluster_embeddings, assign_to_existing_clusters, update_centroids
from modules.labeling import extract_cluster_labels
from modules.summarizer import get_cluster_summaries
from modules.state_manager import save_state

st.set_page_config(page_title="Upload & Process", page_icon="📤", layout="wide")

if "corpus" not in st.session_state:
    from modules.state_manager import load_state
    st.session_state.corpus = load_state()

st.title("📤 Upload Documents")
st.markdown("Upload multiple `TXT`, `PDF`, or `DOCX` files. The system implements incremental memory and will dynamically group newly related clusters.")

uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=['txt', 'pdf', 'docx'])

if uploaded_files:
    if st.button("Run Pipeline", type="primary", use_container_width=True):
        start_time = time.time()
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            # 1. Parsing
            status_text.text("📂 Step 1/6: Parsing & Light Preprocessing...")
            new_documents = process_uploaded_files(uploaded_files)
            if not new_documents:
                st.error("No valid text extracted from the files.")
                st.stop()
            progress_bar.progress(20)
            
            # 2. Embedding newly uploaded docs
            status_text.text("🧠 Step 2/6: Batch Embedding incoming documents...")
            new_texts = [doc["processed_text"] for doc in new_documents]
            new_embeddings = get_embeddings(new_texts)
            progress_bar.progress(40)
            
            # 3. Incremental Assignment
            status_text.text("⚖️ Step 3/6: Comparing against historic cluster memory...")
            state = st.session_state.corpus
            
            # If database is completely empty:
            if not state.get("processed") or len(state.get("embeddings", [])) == 0:
                labels = cluster_embeddings(new_embeddings)
                state["documents"] = new_documents
                state["embeddings"] = new_embeddings
                state["labels"] = labels.tolist()
                modified_clusters = set(labels)
            else:
                # Compare against current centroids
                assigned_labels = assign_to_existing_clusters(new_embeddings, state.get("centroids", {}))
                
                # Append incrementally
                state["documents"].extend(new_documents)
                if state["embeddings"] is None or len(state["embeddings"]) == 0:
                    state["embeddings"] = new_embeddings
                else:
                    state["embeddings"] = np.vstack([state["embeddings"], new_embeddings])
                state["labels"].extend(assigned_labels)
                modified_clusters = set(assigned_labels)
                
            progress_bar.progress(60)
                
            # 4. Handle Noise Pool
            noise_indices = [i for i, lbl in enumerate(state["labels"]) if lbl == -1]
            if len(noise_indices) >= 2:
                status_text.text("🔄 Step 4/6: Analyzing growing Noise Pool...")
                noise_embs = state["embeddings"][noise_indices]
                noise_labels = cluster_embeddings(noise_embs)
                
                # Identify next valid global numeric ID
                valid_ids = [lbl for lbl in state["labels"] if lbl != -1]
                max_existing_id = max(valid_ids) if valid_ids else -1
                next_new_id = max_existing_id + 1
                
                unique_noise_clusters = set(noise_labels)
                noise_map = {}
                new_cluster_ids = []
                for nl in unique_noise_clusters:
                    if nl != -1: 
                        noise_map[nl] = int(next_new_id)
                        new_cluster_ids.append(int(next_new_id))
                        next_new_id += 1
                        
                # Reassign newly discovered patterns back into global namespace
                for i, relative_l in zip(noise_indices, noise_labels):
                    if relative_l != -1:
                        new_global_id = noise_map[relative_l]
                        state["labels"][i] = new_global_id
                        modified_clusters.add(new_global_id)

            progress_bar.progress(70)

            # 5. Update Centroids
            status_text.text("📊 Step 5/6: Updating Centroid Memory Vectors...")
            state["centroids"] = update_centroids(state["embeddings"], state["labels"])
            progress_bar.progress(80)
            
            # 6. Labels & Groq Summarization (only touching modified)
            status_text.text("📝 Step 6/6: Updating Dynamic Labels & LLM Summaries...")
            
            # Prepare fresh mapping for active execution
            cluster_docs_map = {}
            for doc_dict, lbl in zip(state["documents"], state["labels"]):
                if lbl not in cluster_docs_map:
                    cluster_docs_map[lbl] = []
                cluster_docs_map[lbl].append({"filename": doc_dict['filename'], "text": doc_dict['processed_text']})
                
            if "cluster_labels" not in state:
                state["cluster_labels"] = {}
                
            # Regenerate TF-IDF selectively logic (Global array context needed for Sklearn matrix)
            full_labels_mapping = extract_cluster_labels(state["documents"], state["labels"])
            
            # ALWAYS update state["cluster_labels"] for modified OR newly generated
            for cid in modified_clusters:
                if cid in full_labels_mapping:
                    state["cluster_labels"][cid] = full_labels_mapping[cid]
                    
            if 'new_cluster_ids' in locals():
                for cid in new_cluster_ids:
                    if cid in full_labels_mapping:
                        state["cluster_labels"][cid] = full_labels_mapping[cid]

            # Trigger Summarizer ONLY for affected clusters
            modified_cluster_docs = {cid: cluster_docs_map[cid] for cid in modified_clusters if cid in cluster_docs_map}
            
            if modified_cluster_docs:
                updated_summaries = get_cluster_summaries(modified_cluster_docs, state["cluster_labels"])
                if "summaries" not in state:
                    state["summaries"] = {}
                for cid, s_dict in updated_summaries.items():
                    state["summaries"][str(cid)] = s_dict
                    
            state["processed"] = True
            
            # Flush changes to disk utilizing Joblib
            save_state(state)
            
            progress_bar.progress(100)
            
            elapsed = time.time() - start_time
            status_text.text(f"✅ Pipeline correctly processed entirely in {elapsed:.2f} seconds!")
            st.success("Incremental processing successfully appended to Memory! Explore the sidebar for intelligence.")
            
        except Exception as e:
            status_text.text("❌ Critical Error occurred during processing.")
            st.error(f"Details: {str(e)}")
            progress_bar.empty()
