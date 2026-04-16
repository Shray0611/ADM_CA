import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict

def extract_cluster_labels(documents: List[Dict], labels: List[int], n_words: int = 5) -> Dict[int, str]:
    """
    Extracts top words per cluster using TF-IDF. (Replaces KeyBERT).
    Returns a mapping of cluster id to label string: "word1 · word2 · word3"
    """
    if not documents or len(labels) == 0:
        return {}
        
    # Group documents by cluster
    # Noise points are label = -1, we should also label them or skip? The requirement says label noise as well, or at least Group docs by cluster.
    cluster_docs = {}
    for doc, label in zip(documents, labels):
        if label not in cluster_docs:
            cluster_docs[label] = []
        cluster_docs[label].append(doc['processed_text'])
    
    cluster_labels = {}
    
    # TF-IDF across all clustered text blocks
    # We join all texts of a cluster into one large string
    cluster_texts = []
    cluster_ids_ordered = []
    
    for cluster_id, texts in cluster_docs.items():
        combined_text = " ".join(texts)
        if combined_text.strip():
            cluster_texts.append(combined_text)
            cluster_ids_ordered.append(cluster_id)
            
    if not cluster_texts:
        return {}
        
    # Min_df=1 because we might have very few clusters
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    custom_stop_words = list(ENGLISH_STOP_WORDS) + ["reduce", "document", "text", "page", "file"]
    vectorizer = TfidfVectorizer(stop_words=custom_stop_words, max_df=0.9, min_df=1)
    try:
        tfidf_matrix = vectorizer.fit_transform(cluster_texts)
    except ValueError:
        # Happens if vocabulary is empty
        for cid in cluster_ids_ordered:
            cluster_labels[cid] = "Unnamed Cluster"
        return cluster_labels
        
    feature_names = vectorizer.get_feature_names_out()
    
    for i, cluster_id in enumerate(cluster_ids_ordered):
        if cluster_id == -1:
            cluster_labels[cluster_id] = "Misc / Noise"
            continue
            
        row = tfidf_matrix.getrow(i).toarray().flatten()
        top_indices = row.argsort()[-n_words:][::-1]
        top_words = [feature_names[idx] for idx in top_indices if row[idx] > 0]
        
        if top_words:
            cluster_labels[cluster_id] = " | ".join(top_words[:3])
        else:
            cluster_labels[cluster_id] = f"Cluster {cluster_id}"
            
    return cluster_labels
