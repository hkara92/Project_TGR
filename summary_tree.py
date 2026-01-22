"""
summary_tree.py

RAPTOR-style hierarchical summary tree builder.
Matches the ACTUAL behavior of the original RAPTOR code:

1. UMAP dimensionality reduction (global + local)
2. Soft GMM clustering with BIC for optimal K
3. Two-stage clustering: global clusters → local clusters within each
4. NO token budget enforcement (just logging)
5. Recursive until too few nodes to cluster

Key parameters (from original code):
- UMAP dim: 10 (or min(10, len(nodes)-2))
- Global UMAP n_neighbors: sqrt(len(nodes))
- Local UMAP n_neighbors: 10 (fixed)
- GMM threshold: 0.1 (soft assignment)
- BIC max_clusters: 50
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

# External dependencies
import umap
from sklearn.mixture import GaussianMixture


# ============ CONFIG ============

@dataclass
class TreeConfig:
    """Configuration for summary tree building."""
    # UMAP settings
    umap_dim: int = 10
    umap_metric: str = "cosine"
    local_n_neighbors: int = 10
    
    # GMM settings
    gmm_threshold: float = 0.1  # Soft assignment threshold
    bic_max_clusters: int = 50
    random_seed: int = 42
    
    # Stopping condition
    min_nodes_to_cluster: int = 3  # Stop if fewer nodes than this


# ============ DATA STRUCTURES ============

@dataclass
class TreeNode:
    """A node in the summary tree."""
    node_id: str
    level: int
    text: str
    embedding: Optional[np.ndarray] = None
    children_ids: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    
    # For leaf nodes: which original chunk this is
    chunk_id: Optional[str] = None
    
    # For non-leaf nodes: all leaf chunks under this subtree
    leaf_chunk_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict (without embedding)."""
        return {
            "node_id": self.node_id,
            "level": self.level,
            "text": self.text,
            "children_ids": self.children_ids,
            "parent_id": self.parent_id,
            "chunk_id": self.chunk_id,
            "leaf_chunk_ids": self.leaf_chunk_ids
        }


# ============ CLUSTERING (matches original RAPTOR) ============

def get_optimal_clusters_bic(embeddings: np.ndarray, max_clusters: int = 50, random_seed: int = 42) -> int:
    """
    Find optimal number of clusters using BIC (Bayesian Information Criterion).
    Returns K with lowest BIC score.
    """
    max_clusters = min(max_clusters, len(embeddings))
    
    if max_clusters <= 1:
        return 1
    
    bics = []
    for n in range(1, max_clusters):
        try:
            gm = GaussianMixture(n_components=n, random_state=random_seed)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        except Exception:
            # If fitting fails, use a high BIC value
            bics.append(float('inf'))
    
    return np.argmin(bics) + 1


def global_umap_reduce(embeddings: np.ndarray, dim: int = 10, metric: str = "cosine") -> np.ndarray:
    """
    Global UMAP reduction.
    n_neighbors = sqrt(len(embeddings))
    """
    n_samples = len(embeddings)
    
    if n_samples <= dim + 1:
        # Too few samples for UMAP, return as-is or pad
        return embeddings
    
    actual_dim = min(dim, n_samples - 2)
    n_neighbors = max(2, int(np.sqrt(n_samples)))
    n_neighbors = min(n_neighbors, n_samples - 1)
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=actual_dim,
        metric=metric,
        random_state=42
    )
    
    return reducer.fit_transform(embeddings)


def local_umap_reduce(embeddings: np.ndarray, dim: int = 10, n_neighbors: int = 10, metric: str = "cosine") -> np.ndarray:
    """
    Local UMAP reduction (within a cluster).
    n_neighbors = 10 (fixed)
    """
    n_samples = len(embeddings)
    
    if n_samples <= dim + 1:
        return embeddings
    
    actual_dim = min(dim, n_samples - 2)
    actual_neighbors = min(n_neighbors, n_samples - 1)
    
    reducer = umap.UMAP(
        n_neighbors=actual_neighbors,
        n_components=actual_dim,
        metric=metric,
        random_state=42
    )
    
    return reducer.fit_transform(embeddings)


def gmm_soft_cluster(embeddings: np.ndarray, threshold: float = 0.1, random_seed: int = 42) -> Tuple[List[List[int]], int]:
    """
    Soft GMM clustering.
    
    Returns:
        labels: List of cluster indices per node (can have multiple due to soft assignment)
        n_clusters: Number of clusters found
    """
    if len(embeddings) <= 1:
        return [[0]], 1
    
    n_clusters = get_optimal_clusters_bic(embeddings, random_seed=random_seed)
    
    gm = GaussianMixture(n_components=n_clusters, random_state=random_seed)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    
    # Soft assignment: node belongs to cluster if prob > threshold
    labels = []
    for prob in probs:
        cluster_indices = np.where(prob > threshold)[0].tolist()
        # Ensure at least one cluster assignment
        if not cluster_indices:
            cluster_indices = [np.argmax(prob)]
        labels.append(cluster_indices)
    
    return labels, n_clusters


def perform_clustering(embeddings: np.ndarray, config: TreeConfig) -> List[List[int]]:
    """
    Two-stage clustering: Global UMAP+GMM → Local UMAP+GMM per global cluster.
    Matches the original RAPTOR implementation.
    
    Returns:
        List of final cluster indices per node (soft assignment)
    """
    n_samples = len(embeddings)
    
    if n_samples < config.min_nodes_to_cluster:
        # All nodes in one cluster
        return [[0]] * n_samples
    
    # Stage 1: Global UMAP + GMM
    reduced_global = global_umap_reduce(embeddings, config.umap_dim, config.umap_metric)
    global_clusters, n_global = gmm_soft_cluster(reduced_global, config.gmm_threshold, config.random_seed)
    
    # Initialize final cluster assignments
    all_local_clusters = [[] for _ in range(n_samples)]
    total_clusters = 0
    
    # Stage 2: Local UMAP + GMM within each global cluster
    for global_idx in range(n_global):
        # Get indices of nodes belonging to this global cluster
        global_cluster_indices = [
            idx for idx, gc in enumerate(global_clusters) 
            if global_idx in gc
        ]
        
        if len(global_cluster_indices) == 0:
            continue
        
        if len(global_cluster_indices) <= config.umap_dim + 1:
            # Too small to cluster locally - put all in one local cluster
            for idx in global_cluster_indices:
                all_local_clusters[idx].append(total_clusters)
            total_clusters += 1
        else:
            # Do local UMAP + GMM
            cluster_embeddings = embeddings[global_cluster_indices]
            reduced_local = local_umap_reduce(
                cluster_embeddings, 
                config.umap_dim, 
                config.local_n_neighbors, 
                config.umap_metric
            )
            local_clusters, n_local = gmm_soft_cluster(
                reduced_local, 
                config.gmm_threshold, 
                config.random_seed
            )
            
            # Assign local cluster indices (offset by total_clusters)
            for j, idx in enumerate(global_cluster_indices):
                for local_cluster in local_clusters[j]:
                    all_local_clusters[idx].append(local_cluster + total_clusters)
            
            total_clusters += n_local
    
    # Ensure every node has at least one cluster
    for i, clusters in enumerate(all_local_clusters):
        if not clusters:
            all_local_clusters[i] = [0]
    
    return all_local_clusters


# ============ SUMMARY TREE BUILDER ============

class SummaryTreeBuilder:
    """
    Builds a RAPTOR-style hierarchical summary tree.
    
    Usage:
        builder = SummaryTreeBuilder(embedder, summarizer, config)
        tree = builder.build(chunks)
    """
    
    def __init__(
        self,
        embedder,  # Function: List[str] -> np.ndarray
        summarizer,  # Function: List[str] -> str
        tokenizer=None,  # Optional: for token counting/logging
        config: TreeConfig = None
    ):
        self.embedder = embedder
        self.summarizer = summarizer
        self.tokenizer = tokenizer
        self.config = config or TreeConfig()
        
        # Build artifacts
        self.nodes: Dict[str, TreeNode] = {}
        self.levels: Dict[int, List[str]] = {}  # level -> list of node_ids
    
    def build(self, chunks: List[Dict[str, Any]]) -> Dict[str, TreeNode]:
        """
        Build the summary tree from chunks.
        
        Args:
            chunks: List of {"chunk_id": str, "text": str, ...}
        
        Returns:
            Dict of all nodes keyed by node_id
        """
        print(f"\n{'='*60}")
        print("BUILDING SUMMARY TREE")
        print(f"{'='*60}")
        print(f"Input chunks: {len(chunks)}")
        
        # Step 1: Create leaf nodes (level 0)
        print("\n[Level 0] Creating leaf nodes...")
        leaf_nodes = self._create_leaf_nodes(chunks)
        self.levels[0] = [n.node_id for n in leaf_nodes]
        
        for node in leaf_nodes:
            self.nodes[node.node_id] = node
        
        print(f"  Created {len(leaf_nodes)} leaf nodes")
        
        # Step 2: Recursively cluster and summarize
        current_level = 0
        current_nodes = leaf_nodes
        
        while len(current_nodes) >= self.config.min_nodes_to_cluster:
            current_level += 1
            print(f"\n[Level {current_level}] Clustering {len(current_nodes)} nodes...")
            
            # Get embeddings
            embeddings = np.array([n.embedding for n in current_nodes])
            
            # Cluster
            cluster_assignments = perform_clustering(embeddings, self.config)
            
            # Group nodes by cluster
            cluster_to_nodes: Dict[int, List[TreeNode]] = {}
            for node, clusters in zip(current_nodes, cluster_assignments):
                for cluster_id in clusters:
                    if cluster_id not in cluster_to_nodes:
                        cluster_to_nodes[cluster_id] = []
                    cluster_to_nodes[cluster_id].append(node)
            
            print(f"  Found {len(cluster_to_nodes)} clusters")
            
            # Create parent nodes for each cluster
            parent_nodes = []
            for cluster_id in sorted(cluster_to_nodes.keys()):
                children = cluster_to_nodes[cluster_id]
                parent = self._create_parent_node(children, current_level, cluster_id)
                parent_nodes.append(parent)
                self.nodes[parent.node_id] = parent
                
                # Update children's parent_id
                for child in children:
                    child.parent_id = parent.node_id
            
            self.levels[current_level] = [n.node_id for n in parent_nodes]
            print(f"  Created {len(parent_nodes)} parent nodes")
            
            # Log token counts (NO enforcement, just logging like original)
            if self.tokenizer:
                for parent in parent_nodes:
                    children = [self.nodes[cid] for cid in parent.children_ids]
                    child_texts = " ".join([c.text for c in children])
                    child_tokens = len(self.tokenizer.encode(child_texts))
                    parent_tokens = len(self.tokenizer.encode(parent.text))
                    print(f"    {parent.node_id}: {child_tokens} child tokens → {parent_tokens} summary tokens")
            
            current_nodes = parent_nodes
        
        print(f"\n{'='*60}")
        print(f"TREE COMPLETE: {len(self.nodes)} total nodes, {current_level + 1} levels")
        print(f"{'='*60}")
        
        return self.nodes
    
    def _create_leaf_nodes(self, chunks: List[Dict[str, Any]]) -> List[TreeNode]:
        """Create leaf nodes from chunks and embed them."""
        # Extract texts
        texts = [c["text"] for c in chunks]
        
        # Embed all at once
        embeddings = self.embedder(texts)
        
        # Create nodes
        nodes = []
        for i, chunk in enumerate(chunks):
            node = TreeNode(
                node_id=f"leaf_{chunk['chunk_id']}",
                level=0,
                text=chunk["text"],
                embedding=embeddings[i],
                chunk_id=chunk["chunk_id"],
                leaf_chunk_ids=[chunk["chunk_id"]]
            )
            nodes.append(node)
        
        return nodes
    
    def _create_parent_node(self, children: List[TreeNode], level: int, cluster_id: int) -> TreeNode:
        """Create a parent node by summarizing children."""
        # Concatenate child texts
        child_texts = [c.text for c in children]
        combined_text = "\n\n".join(child_texts)
        
        # Summarize
        summary = self.summarizer(child_texts)
        
        # Embed summary
        embedding = self.embedder([summary])[0]
        
        # Collect all leaf chunk IDs from children
        leaf_chunk_ids = []
        for child in children:
            leaf_chunk_ids.extend(child.leaf_chunk_ids)
        leaf_chunk_ids = list(set(leaf_chunk_ids))  # Deduplicate
        
        return TreeNode(
            node_id=f"L{level}_C{cluster_id}",
            level=level,
            text=summary,
            embedding=embedding,
            children_ids=[c.node_id for c in children],
            leaf_chunk_ids=leaf_chunk_ids
        )
    
    def get_level_nodes(self, level: int) -> List[TreeNode]:
        """Get all nodes at a specific level."""
        if level not in self.levels:
            return []
        return [self.nodes[nid] for nid in self.levels[level]]
    
    def get_leaves_for_node(self, node_id: str) -> List[str]:
        """Get all leaf chunk IDs under a node."""
        if node_id not in self.nodes:
            return []
        return self.nodes[node_id].leaf_chunk_ids


# ============ SAVE / LOAD ============

def save_tree(nodes: Dict[str, TreeNode], levels: Dict[int, List[str]], cache_dir: str):
    """Save tree structure to disk (without embeddings)."""
    tree_dir = os.path.join(cache_dir, "summary_tree")
    os.makedirs(tree_dir, exist_ok=True)
    
    # Save nodes (without embeddings)
    nodes_data = {nid: node.to_dict() for nid, node in nodes.items()}
    with open(os.path.join(tree_dir, "nodes.json"), "w", encoding="utf-8") as f:
        json.dump(nodes_data, f, ensure_ascii=False, indent=2)
    
    # Save levels
    levels_data = {str(k): v for k, v in levels.items()}
    with open(os.path.join(tree_dir, "levels.json"), "w", encoding="utf-8") as f:
        json.dump(levels_data, f, ensure_ascii=False, indent=2)
    
    # Save embeddings separately (numpy format)
    embeddings = {}
    for nid, node in nodes.items():
        if node.embedding is not None:
            embeddings[nid] = node.embedding
    
    np.savez(os.path.join(tree_dir, "embeddings.npz"), **embeddings)
    
    print(f"Tree saved to {tree_dir}")


def load_tree(cache_dir: str) -> Tuple[Dict[str, TreeNode], Dict[int, List[str]]]:
    """Load tree from disk."""
    tree_dir = os.path.join(cache_dir, "summary_tree")
    
    # Load nodes
    with open(os.path.join(tree_dir, "nodes.json"), "r", encoding="utf-8") as f:
        nodes_data = json.load(f)
    
    # Load levels
    with open(os.path.join(tree_dir, "levels.json"), "r", encoding="utf-8") as f:
        levels_data = json.load(f)
    levels = {int(k): v for k, v in levels_data.items()}
    
    # Load embeddings
    embeddings_data = np.load(os.path.join(tree_dir, "embeddings.npz"))
    
    # Reconstruct nodes
    nodes = {}
    for nid, data in nodes_data.items():
        node = TreeNode(
            node_id=data["node_id"],
            level=data["level"],
            text=data["text"],
            embedding=embeddings_data[nid] if nid in embeddings_data.files else None,
            children_ids=data["children_ids"],
            parent_id=data["parent_id"],
            chunk_id=data.get("chunk_id"),
            leaf_chunk_ids=data["leaf_chunk_ids"]
        )
        nodes[nid] = node
    
    print(f"Tree loaded from {tree_dir}: {len(nodes)} nodes, {len(levels)} levels")
    return nodes, levels


# ============ EXAMPLE USAGE ============

if __name__ == "__main__":
    # This is a test with mock embedder and summarizer
    
    def mock_embedder(texts: List[str]) -> np.ndarray:
        """Mock embedder: returns random vectors."""
        return np.random.randn(len(texts), 384)
    
    def mock_summarizer(texts: List[str]) -> str:
        """Mock summarizer: concatenates first 50 chars of each."""
        snippets = [t[:50] + "..." for t in texts]
        return f"Summary of {len(texts)} texts: " + " | ".join(snippets[:3])
    
    # Create test chunks
    chunks = [
        {"chunk_id": f"chunk_{i}", "text": f"This is the content of chunk {i}. " * 10}
        for i in range(20)
    ]
    
    # Build tree
    config = TreeConfig(
        umap_dim=5,  # Smaller for test
        min_nodes_to_cluster=3
    )
    
    builder = SummaryTreeBuilder(
        embedder=mock_embedder,
        summarizer=mock_summarizer,
        config=config
    )
    
    nodes = builder.build(chunks)
    
    # Print tree structure
    print("\n--- TREE STRUCTURE ---")
    for level in sorted(builder.levels.keys()):
        level_nodes = builder.get_level_nodes(level)
        print(f"\nLevel {level}: {len(level_nodes)} nodes")
        for node in level_nodes[:3]:  # Show first 3
            print(f"  {node.node_id}: {len(node.leaf_chunk_ids)} leaves, {len(node.children_ids)} children")