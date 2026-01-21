import numpy as np
from typing import List, Dict, Any
from sklearn.mixture import GaussianMixture
from sentence_transformers import SentenceTransformer
from llm import call_llm

SUMMARIZE_PROMPT = """Summarize the following content concisely (200-300 words), capturing main characters, events, and key details.

Content:
{content}

Summary:"""


def build_raptor_tree(
    chunks: List[Dict[str, Any]],
    llm_choice: str = "gpt",
    reduction_factor: int = 10,
    embedder_model: str = "BAAI/bge-m3"
) -> Dict[str, Dict[str, Any]]:

    # Initialize embedder
    embedder = SentenceTransformer(embedder_model)
    
    tree = {}
    
    # Add leaf chunks (level = -1 to indicate leaves)
    for chunk in chunks:
        tree[chunk["chunk_id"]] = {
            "text": chunk["text"],
            "level": -1,
            "children_ids": None,
            "parent_id": None,
            "leaf_chunk_ids": [chunk["chunk_id"]]
        }
    
    # Build levels bottom-up
    current_ids = [c["chunk_id"] for c in chunks]
    current_texts = [c["text"] for c in chunks]
    level = 0
    
    while len(current_ids) > 1:
        n_clusters = max(1, len(current_ids) // reduction_factor)
        
        # Embed current texts for clustering
        embeddings = embedder.encode(current_texts, show_progress_bar=False)
        
        # Cluster using GMM
        if len(current_ids) <= n_clusters:
            # Fewer nodes than clusters - each node is its own cluster
            clusters = [[i] for i in range(len(current_ids))]
        else:
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            labels = gmm.fit_predict(embeddings)
            clusters = [[] for _ in range(n_clusters)]
            for idx, label in enumerate(labels):
                clusters[label].append(idx)
            clusters = [c for c in clusters if c]  # Remove empty clusters
        
        # Create summaries for each cluster
        next_ids, next_texts = [], []
        
        for i, cluster_indices in enumerate(clusters):
            cluster_ids = [current_ids[j] for j in cluster_indices]
            cluster_texts = [current_texts[j] for j in cluster_indices]
            
            # Summarize cluster
            combined = "\n\n".join(cluster_texts)
            summary = call_llm(SUMMARIZE_PROMPT.format(content=combined), llm_choice)
            
            # Create summary node
            summary_id = f"summary_{level}_{i}"
            
            # Collect all leaf chunk IDs covered by this summary
            leaf_ids = []
            for nid in cluster_ids:
                leaf_ids.extend(tree[nid]["leaf_chunk_ids"])
            
            tree[summary_id] = {
                "text": summary,
                "level": level,
                "children_ids": cluster_ids,
                "parent_id": None,
                "leaf_chunk_ids": leaf_ids
            }
            
            # Set parent for children
            for nid in cluster_ids:
                tree[nid]["parent_id"] = summary_id
            
            next_ids.append(summary_id)
            next_texts.append(summary)
        
        print(f"Level {level}: {len(current_ids)} nodes → {len(clusters)} summaries")
        current_ids, current_texts = next_ids, next_texts
        level += 1
    
    print(f"Tree complete: {len(tree)} total nodes, {level} levels")
    return tree