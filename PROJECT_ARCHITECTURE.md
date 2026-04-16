# Fast Document Clustering & Summarization App
**Technical Architecture & Project Documentation**

## 1. System Overview
This project is a high-performance, modular Streamlit web application engineered to parse, embed, semantically cluster, natively label, and algorithmically summarize vast corpuses of `.txt`, `.pdf`, and `.docx` documents. 

It fundamentally bypasses heavy NLP preprocessing pipelines (like spaCy or complex UMAP dimensional reduction) in favor of optimized localized batch-processing. It uses an **Incremental Persistent Memory System**, which securely tracks, scales, and compares new document uploads against a historical centroid memory mathematically, eliminating redundant API LLM calls and redundant embedding generation.

---

## 2. Technology Stack & Core Algorithms
- **Frontend & Routing:** Streamlit (Multipage Configuration).
- **Text Embedding:** `SentenceTransformers` (`all-MiniLM-L6-v2`) generating 384-dimensional dense semantic vectors.
- **Incremental State Persistence:** `Joblib` handling rapid read/write binary serialization of Python global state dicts.
- **Clustering Engine:** 
  - `HDBSCAN`: Used strictly for robust density-based extraction, preventing traditional forced spherical clusters. (`min_cluster_size=2`, `min_samples=1`, `metric='euclidean'`).
  - `KMeans`: Executed mathematically purely as a targeted fallback hook if HDBSCAN noise thresholds reject data that fundamentally still correlates.
- **Label Extraction:** `TfidfVectorizer` mapping high-weight vocabulary tokens organically without requiring heavy LLMs. Stopwords dynamically eradicate common text boilerplates (e.g., `["reduce", "document", "text"]`).
- **Semantic LLM Summarization:** `Groq API` querying `llama-3.3-70b-versatile` operating strictly on JSON constraint payloads.
- **Analytics Visualization:** `Plotly Express` handling Cartesian embeddings representations.

---

## 3. Structural Flow & The Incremental Pipeline

### 3.1 Upload & Parsing (`modules/ingestion.py`)
Documents are ingested using `PyPDF2`/`pdfplumber`, `python-docx`, or standard `utf-8` decoders. They get stripped of excessive carriage returns and non-UTF-8 artifacts dynamically to sanitize the raw corpus.

### 3.2 Global State Definition (`modules/state_manager.py`)
The system tracks the exact mathematical sequence of events permanently in `state.joblib`:
```python
state_dict = {
  "documents": [],        # Original processed text payloads
  "embeddings": np.array, # 384-dimensional memory vectors
  "labels": [],           # Global Cluster Integer Array
  "centroids": {},        # Dictionary: cluster_id -> average embedding array
  "cluster_labels": {},   # Dictionary: cluster_id -> "TF | IDF | Label"
  "summaries": {}         # Dictionary: cluster_id -> LLM structured output
}
```

### 3.3 Incremental Vector Routing (`modules/clustering.py`)
If the corpus already exists in the `joblib` memory cache:
1. New inbound documents bypass normal HDBSCAN pipelines immediately.
2. They are mathematically multiplied directly against the active `state["centroids"]` using `cosine_similarity`.
3. If an embedding scores `>= 0.7` against a predefined centroid's structural locus, it natively snaps directly into that historical cluster globally without touching an external model.
4. If it fails `< 0.7`, the system tags it as `-1` (Noise) and pushes it to the generic `noise_pool`.

### 3.4 The Noise Threshold Event
When the cumulative `len(noise_pool)` hits `>= 2`:
1. The noise vectors are isolated securely and purged through `HDBSCAN`.
2. New relational trends that previously weren't dense enough to count as clusters dynamically spawn into new global cluster objects natively spanning off `max_existing_id + 1`.

### 3.5 Isolated Labeling & Cost-Saving Summarization
Any dynamically shifted matrices (new documents bridging into old clusters, or Noise collapsing into a new cluster) are flagged heavily into `modified_clusters`.
- `extract_cluster_labels()` re-parses the TF-IDF parameters explicitly to adapt the linguistic title. 
- *Crucially*, `modules/summarizer.py` isolates ONLY keys residing inside `modified_clusters`. Rather than submitting 50 clusters to the Groq LLaMA pipeline, it surgically updates only the 1 or 2 matrices physically altered by the Upload Event, saving massive latency and LLM token bloat. LLaMA yields standardized Title, Insights (x3), and structural sentence summaries matching JSON schemas parsed via static RegEx integers.

---

## 4. UI Architecture (`pages/`)

* **`01_upload.py`**: Controls the core data manipulation ingestion loop. Computes vectors -> Resolves Logic Rules -> Saves joblib tracking.
* **`02_clusters.py`**: Interrogates Plotly visually rendering a scatter plot of elements to simulate groupings spatially.
* **`03_embeddings.py`**: Formats the arrays into unique 2-dimensional DataFrames mapping Cosine Multipliers recursively. Includes Semantic Query extraction allowing users to embed strings on the fly and seek top top-5 nearest neighbors traversing the `corpus` dictionary.
* **`04_summaries.py`**: Reads `state["summaries"]` securely bypassing empty queries if `cid == -1` generating stylized HTML component badges directly from our string `" | "` delimiter.
* **`05_documents.py`**: Core table grid rendering UI indexing strictly for specific document metadata matching per grouping.
