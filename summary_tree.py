import time
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from sklearn.mixture import GaussianMixture
from sentence_transformers import SentenceTransformer
from llm import call_llm

SUMMARIZE_PROMPT = """Summarize the following content concisely (200-300 words), capturing main characters, events, and key details.

Content:
{content}

Summary:"""

MAX_CONTENT_CHARS = 12000


def build_raptor_tree(
    chunks: List[Dict[str, Any]],
    embedder: SentenceTransformer,  # Pass in, don't create
    llm_choice: str = "gpt",
    reduction_factor: int = 10,
    cache_path: Optional[str] = None
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Build RAPTOR-style summary tree using GMM clustering.
    
    Args:
        chunks: List of chunk dicts with chunk_id and text
        embedder: Pre-initialized SentenceTransformer (shared across books)
        llm_choice: Which LLM to use
        reduction_factor: Target chunks per cluster
        cache_path: Where to save/load cached tree
    
    Returns:
        tree: dict mapping node_id -> node_data
        stats: dict with build statistics
    """
    # Check cache
    if cache_path and os.path.exists(cache_path):
        print(f"Loading tree from cache: {cache_path}")
        with open(cache_path, "r") as f:
            tree = json.load(f)
        return tree, {"cached": True}
    
    start_time = time.time()
    tree = {}
    
    # Add leaf chunks
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
    llm_calls = 0
    
    while len(current_ids) > 1:
        n_clusters = max(1, len(current_ids) // reduction_factor)
        
        # Embed and cluster
        embeddings = embedder.encode(current_texts, show_progress_bar=False)
        
        if len(current_ids) <= n_clusters:
            clusters = [[i] for i in range(len(current_ids))]
        else:
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            labels = gmm.fit_predict(embeddings)
            clusters = [[] for _ in range(n_clusters)]
            for idx, label in enumerate(labels):
                clusters[label].append(idx)
            clusters = [c for c in clusters if c]
        
        # Create summaries
        next_ids, next_texts = [], []
        
        for i, cluster_indices in enumerate(clusters):
            cluster_ids = [current_ids[j] for j in cluster_indices]
            cluster_texts = [current_texts[j] for j in cluster_indices]
            
            # Combine with truncation
            combined = "\n\n".join(cluster_texts)
            if len(combined) > MAX_CONTENT_CHARS:
                combined = combined[:MAX_CONTENT_CHARS] + "\n\n[Truncated]"
            
            # Generate summary
            try:
                summary = call_llm(SUMMARIZE_PROMPT.format(content=combined), llm_choice)
                llm_calls += 1
            except Exception as e:
                print(f"Warning: LLM failed for level {level} cluster {i}: {e}")
                summary = combined[:500] + "..."
            
            # Create node
            summary_id = f"summary_{level}_{i}"
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
            
            for nid in cluster_ids:
                tree[nid]["parent_id"] = summary_id
            
            next_ids.append(summary_id)
            next_texts.append(summary)
        
        print(f"Level {level}: {len(current_ids)} → {len(clusters)} summaries")
        current_ids, current_texts = next_ids, next_texts
        level += 1
    
    build_time = time.time() - start_time
    
    # Save cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(tree, f, indent=2)
    
    stats = {
        "build_time": build_time,
        "num_nodes": len(tree),
        "num_levels": level,
        "num_leaves": len(chunks),
        "llm_calls": llm_calls
    }
    
    print(f"Tree complete: {len(tree)} nodes, {level} levels, {build_time:.2f}s")
    return tree, stats