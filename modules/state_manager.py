import os
import joblib
import numpy as np
from typing import Dict, Any

STATE_FILE = "state.joblib"

def get_empty_state() -> Dict[str, Any]:
    return {
        "documents": [],
        "embeddings": np.array([]),
        "labels": [],
        "centroids": {},
        "cluster_labels": {},
        "summaries": {},
        "processed": False
    }

def load_state() -> Dict[str, Any]:
    """
    Loads persistent state from disk if available, otherwise returns empty struct.
    """
    if os.path.exists(STATE_FILE):
        try:
            return joblib.load(STATE_FILE)
        except Exception as e:
            print(f"Error loading state from {STATE_FILE}: {e}")
            return get_empty_state()
    return get_empty_state()

def save_state(state: Dict[str, Any]):
    """
    Saves state dictionary to disk.
    """
    joblib.dump(state, STATE_FILE)
