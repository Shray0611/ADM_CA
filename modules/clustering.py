import hdbscan
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from typing import List, Dict

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
        max_possible_clusters = embeddings.shape[0] // 2
        n_clusters = max(1, min(3, max_possible_clusters))
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

def assign_to_existing_clusters(new_embeddings: np.ndarray, centroids: Dict[int, np.ndarray], threshold: float = 0.7) -> List[int]:
    """
    Compare new embeddings against existing cluster centroids.
    Returns list of assigned cluster IDs. -1 if no centroid matches well.
    """
    labels = []
    if not centroids:
        return [-1] * len(new_embeddings)
        
    cluster_ids = list(centroids.keys())
    centroid_matrix = np.vstack([centroids[cid] for cid in cluster_ids])
    
    sim_matrix = cosine_similarity(new_embeddings, centroid_matrix)
    
    for i in range(len(new_embeddings)):
        best_idx = np.argmax(sim_matrix[i])
        if sim_matrix[i][best_idx] >= threshold:
            labels.append(cluster_ids[best_idx])
        else:
            labels.append(-1)
            
    return labels

def update_centroids(embeddings: np.ndarray, labels: List[int]) -> Dict[int, np.ndarray]:
    """
    Recalculates mean embedding for every cluster (ignoring noise).
    """
    centroids = {}
    labels_arr = np.array(labels)
    unique_clusters = set(labels)
    
    for cid in unique_clusters:
        if cid == -1:
            continue
        indices = np.where(labels_arr == cid)[0]
        if len(indices) > 0:
            cluster_embs = embeddings[indices]
            centroids[cid] = np.mean(cluster_embs, axis=0)
            
    return centroids
