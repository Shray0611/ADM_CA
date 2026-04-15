import hdbscan
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

def cluster_embeddings(embeddings: np.ndarray, min_cluster_size: int = 2):
    """
    Clusters the embeddings using HDBSCAN.
    Handles noise (label = -1).
    """
    if embeddings.shape[0] == 0:
        return np.array([])
        
    # Using HDBSCAN as required
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1, metric='euclidean', cluster_selection_method='eom')
    labels = clusterer.fit_predict(embeddings)
    
    # Fallback to KMeans if all labels are noise (-1)
    if all(lbl == -1 for lbl in labels):
        n_clusters = min(3, embeddings.shape[0])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)

    # Post-processing: remove single-document clusters
    counts = Counter(labels)
    filtered_labels = []

    for label in labels:
        if label == -1:
            filtered_labels.append(-1)
        elif counts[label] < 2:
            filtered_labels.append(-1)   # convert to noise
        else:
            filtered_labels.append(label)
            
    return np.array(filtered_labels)
