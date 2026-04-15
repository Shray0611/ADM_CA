import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

@st.cache_resource
def load_embedding_model(model_name: str = 'all-MiniLM-L6-v2'):
    """
    Loads and caches the sentence-transformers model.
    """
    return SentenceTransformer(model_name)

def get_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Batch encodes a list of texts into embeddings.
    """
    if not texts:
        return np.array([])
        
    model = load_embedding_model()
    # Batch processing is mandatory per constraints
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
    return embeddings
