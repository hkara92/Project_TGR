"""
summary_tree.py - Minimal RAPTOR-style summary tree

Core algorithm:
1. Global UMAP + GMM clustering
2. Local UMAP + GMM within each global cluster  
3. Summarize clusters → embed → repeat until few nodes left
"""

import os
import json
import numpy as np
import umap
from sklearn.mixture import GaussianMixture


def get_optimal_k(embeddings, max_k=50):
    """Find optimal cluster count using BIC."""
    max_k = min(max_k, len(embeddings))
    if max_k <= 1:
        return 1
    
    bics = []
    for k in range(1, max_k):
        gm = GaussianMixture(n_components=k, random_state=42)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    
    return np.argmin(bics) + 1


def gmm_cluster(embeddings, threshold=0.1):
    """Soft GMM clustering. Returns list of cluster indices per node."""
    if len(embeddings) <= 1:
        return [[0]], 1
    
    k = get_optimal_k(embeddings)
    gm = GaussianMixture(n_components=k, random_state=42)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    
    # Soft assignment: node in cluster if prob > threshold
    labels = []
    for prob in probs:
        clusters = np.where(prob > threshold)[0].tolist()
        if not clusters:
            clusters = [np.argmax(prob)]
        labels.append(clusters)
    
    return labels, k


def cluster_nodes(embeddings, dim=10, threshold=0.1):
    """Two-stage clustering: global UMAP+GMM → local UMAP+GMM per cluster."""
    n = len(embeddings)
    if n < 3:
        return [[0]] * n
    
    # Global UMAP
    global_dim = min(dim, n - 2)
    global_neighbors = max(2, int(np.sqrt(n)))
    reduced_global = umap.UMAP(
        n_neighbors=min(global_neighbors, n-1),
        n_components=global_dim,
        metric="cosine",
        random_state=42
    ).fit_transform(embeddings)
    
    global_labels, n_global = gmm_cluster(reduced_global, threshold)
    
    # Local clustering within each global cluster
    final_labels = [[] for _ in range(n)]
    total_clusters = 0
    
    for g in range(n_global):
        indices = [i for i, labels in enumerate(global_labels) if g in labels]
        
        if len(indices) <= dim + 1:
            # Too small - one local cluster
            for i in indices:
                final_labels[i].append(total_clusters)
            total_clusters += 1
        else:
            # Local UMAP + GMM
            local_emb = embeddings[indices]
            reduced_local = umap.UMAP(
                n_neighbors=min(10, len(indices)-1),
                n_components=min(dim, len(indices)-2),
                metric="cosine",
                random_state=42
            ).fit_transform(local_emb)
            
            local_labels, n_local = gmm_cluster(reduced_local, threshold)
            
            for j, i in enumerate(indices):
                for c in local_labels[j]:
                    final_labels[i].append(c + total_clusters)
            total_clusters += n_local
    
    return final_labels


def build_tree(chunks, embedder, summarizer, min_nodes=3):
    """
    Build summary tree from chunks.
    
    Args:
        chunks: List of {"chunk_id": str, "text": str}
        embedder: func(List[str]) -> np.ndarray
        summarizer: func(List[str]) -> str
        min_nodes: Stop when fewer nodes than this
    
    Returns:
        nodes: dict[node_id] -> {text, embedding, children, leaves}
        levels: dict[level] -> list of node_ids
    """
    nodes = {}
    levels = {}
    
    # Level 0: leaf nodes
    texts = [c["text"] for c in chunks]
    embeddings = embedder(texts)
    
    current = []
    for i, chunk in enumerate(chunks):
        node_id = f"L0_{chunk['chunk_id']}"
        nodes[node_id] = {
            "text": chunk["text"],
            "embedding": embeddings[i],
            "children": [],
            "leaves": [chunk["chunk_id"]]
        }
        current.append(node_id)
    
    levels[0] = current
    level = 0
    
    # Build higher levels
    while len(current) >= min_nodes:
        level += 1
        embs = np.array([nodes[nid]["embedding"] for nid in current])
        
        # Cluster
        cluster_labels = cluster_nodes(embs)
        
        # Group by cluster
        clusters = {}
        for nid, labels in zip(current, cluster_labels):
            for c in labels:
                if c not in clusters:
                    clusters[c] = []
                clusters[c].append(nid)
        
        # Create parent nodes
        parents = []
        for c in sorted(clusters.keys()):
            children = clusters[c]
            child_texts = [nodes[nid]["text"] for nid in children]
            
            # Summarize and embed
            summary = summarizer(child_texts)
            emb = embedder([summary])[0]
            
            # Collect all leaves
            leaves = []
            for nid in children:
                leaves.extend(nodes[nid]["leaves"])
            
            parent_id = f"L{level}_C{c}"
            nodes[parent_id] = {
                "text": summary,
                "embedding": emb,
                "children": children,
                "leaves": list(set(leaves))
            }
            parents.append(parent_id)
        
        levels[level] = parents
        current = parents
        print(f"Level {level}: {len(parents)} nodes from {len(clusters)} clusters")
    
    return nodes, levels


def save_tree(nodes, levels, cache_dir):
    """Save tree to disk."""
    tree_dir = os.path.join(cache_dir, "summary_tree")
    os.makedirs(tree_dir, exist_ok=True)
    
    # Save structure (without embeddings)
    structure = {nid: {k: v for k, v in n.items() if k != "embedding"} 
                 for nid, n in nodes.items()}
    with open(os.path.join(tree_dir, "nodes.json"), "w") as f:
        json.dump(structure, f, indent=2)
    
    with open(os.path.join(tree_dir, "levels.json"), "w") as f:
        json.dump({str(k): v for k, v in levels.items()}, f, indent=2)
    
    # Save embeddings
    embs = {nid: n["embedding"] for nid, n in nodes.items()}
    np.savez(os.path.join(tree_dir, "embeddings.npz"), **embs)


def load_tree(cache_dir):
    """Load tree from disk."""
    tree_dir = os.path.join(cache_dir, "summary_tree")
    
    with open(os.path.join(tree_dir, "nodes.json")) as f:
        nodes = json.load(f)
    
    with open(os.path.join(tree_dir, "levels.json")) as f:
        levels = {int(k): v for k, v in json.load(f).items()}
    
    embs = np.load(os.path.join(tree_dir, "embeddings.npz"))
    for nid in nodes:
        nodes[nid]["embedding"] = embs[nid]
    
    return nodes, levels