import os
import json
from groq import Groq
from typing import List, Dict, Any

def get_cluster_summaries(cluster_docs: Dict[int, List[Dict[str, str]]], cluster_labels: Dict[int, str]) -> Dict[str, Any]:
    """
    Generates summaries for ALL clusters in ONE Groq API call (LLaMA 3.3-70B).
    Requires GROQ_API_KEY to be set in environment.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        return {
            str(cid): {
                "summary": "API Key not found or invalid. Please check your .env file.",
                "insights": ["Requires GROQ_API_KEY", "Update .env file", "Restart application"],
                "title": cluster_labels.get(cid, f"Cluster {cid}")
            } for cid in cluster_docs.keys()
        }
        
    client = Groq(api_key=api_key)
    
    # Constructing the combined prompt for all clusters
    prompt_clusters = []
    
    for cluster_id, docs in cluster_docs.items():
        if cluster_id == -1:
            continue
            
        selected_docs = docs[:20]
        
        cluster_text_parts = []
        doc_names = []
        for d in selected_docs:
            words = d["text"].split()[:150]
            cluster_text_parts.append(d["filename"] + ": " + " ".join(words))
            doc_names.append("- " + d["filename"])
            
        combined_text = "\n".join(cluster_text_parts)
        doc_list = "\n".join(doc_names)
        
        prompt_clusters.append(f"Cluster {cluster_id}:\nTF-IDF Label Hint: {cluster_labels.get(cluster_id, '')}\nDocuments:\n{doc_list}\n\nContent:\n{combined_text}\n")
        
    full_prompt_text = "\n---\n".join(prompt_clusters)
    
    if not full_prompt_text.strip():
        # Only noise exists
        if -1 in cluster_docs:
            return {
                "-1": {
                    "summary": "This cluster contains noise/uncategorized documents that did not fit well into other clusters.",
                    "insights": ["Various mixed topics", "Low density regions", "Outlier documents"],
                    "title": "Misc / Noise"
                }
            }
        return {}
    
    prompt = f"""You are an expert document analyst.

You are given clusters of documents.

For each cluster:
1. Read all documents
2. Generate:
   - Title (max 6 words)
   - Summary (3–5 sentences)
   - 3 insights

Return STRICT JSON:

{{
  "clusters": {{
    "0": {{
      "title": "...",
      "summary": "...",
      "insights": ["...", "...", "..."]
    }},
    "1": {{...}}
  }}
}}

Input data:
{full_prompt_text}
"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that outputs only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
        )
        
        response_content = chat_completion.choices[0].message.content
        if response_content.startswith("```json"):
            response_content = response_content.strip("`").replace("json\n", "", 1)
        elif response_content.startswith("```"):
            response_content = response_content.strip("`")
            
        parsed_json = json.loads(response_content)
        
        output_clusters = parsed_json.get("clusters", {})
        
        if -1 in cluster_docs:
            output_clusters["-1"] = {
                "summary": "This cluster contains noise/uncategorized documents that did not fit well into other clusters.",
                "insights": ["Various mixed topics", "Low density regions", "Outlier documents"],
                "title": "Misc / Noise"
            }
            
        return output_clusters
        
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return {
            str(cid): {
                "summary": "Error generating summary.",
                "insights": ["API Error", str(e)],
                "title": cluster_labels.get(cid, f"Cluster {cid}")
            } for cid in cluster_docs.keys()
        }
