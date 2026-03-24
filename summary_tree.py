"""
This file builds our hierarchical summary tree, similar to the RAPTOR paper.

Here is how it works:
1. We cluster everything globally using UMAP and Gaussian Mixture Models (GMM).
2. We then take each large cluster and break it down further into smaller local clusters.
3. We summarize those clusters, get their embeddings, and repeat the whole process until we have a single root node.

We use two different prompts:
- For the first layer (Level 1), our prompt extracts concrete facts from the raw text chunks.
- For higher layers (Level 2 and above), our prompt synthesizes summaries of summaries to capture the broader picture.
"""

import os
import json
import numpy as np
import umap
from sklearn.mixture import GaussianMixture
from tqdm import tqdm



LEAF_PROMPT = """Summarize the following text passage from a novel. Include:
- Main characters and their actions
- Key plot events
- Important details (locations, conflicts, revelations)

Write 200-300 words. Do NOT add information not in the text. Start your summary immediately without preamble.

TEXT:
{text}

SUMMARY:"""


SUMMARY_PROMPT = """Combine the following summaries into a single cohesive summary. Preserve all key characters, events, and details. Write 200-300 words. Start immediately without preamble.

SUMMARIES:
{text}

COMBINED SUMMARY:"""


def create_summarizer(llm_fn):
    """
    Sets up a summarization function that knows which prompt to use based on how high up the tree we are.
    It takes in your LLM generation function and returns a new function that accepts a list of texts and their current tree level.
    """
    def summarizer(texts, level):
        combined = "\n\n---\n\n".join(texts)
        
        # If we are at the bottom of the tree, we use the detailed extraction prompt.
        # Otherwise, we use the synthesis prompt for combining existing summaries.
        if level == 1:
            prompt = LEAF_PROMPT.format(text=combined)
        else:
            prompt = SUMMARY_PROMPT.format(text=combined)
        
        return llm_fn(prompt)
    
    return summarizer


# ============ CLUSTERING LOGIC ============

def get_optimal_k(embeddings, max_k=50):
    """Figures out the best number of clusters to use by checking the Bayesian Information Criterion (BIC) score."""
    max_k = min(max_k, max(1, len(embeddings) // 2))
    
    if max_k <= 1:
        return 1
    
    bics = []
    for k in range(1, max_k):
        try:
            gm = GaussianMixture(n_components=k, random_state=42, reg_covar=1e-5)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        except (ValueError, np.linalg.LinAlgError):
            # Sometimes when there's not enough data, the GMM fails to fit, so we just stop looking for more clusters.
            break
    
    if not bics:
        return 1
    
    return np.argmin(bics) + 1


def gmm_cluster(embeddings, threshold=0.1):
    """Runs soft clustering. Soft clustering means one node might belong to multiple clusters if it sits right on the boundary."""
    if len(embeddings) <= 1:
        return [[0]], 1
    
    k = get_optimal_k(embeddings)
    try:
        gm = GaussianMixture(n_components=k, random_state=42, reg_covar=1e-5)
        gm.fit(embeddings)
    except Exception:
        # If the clustering completely breaks down, we just lump everything into one big cluster to stay safe.
        return [[0] for _ in embeddings], 1
        
    probs = gm.predict_proba(embeddings)
    
    # Assign the node to any cluster where its probability beats our threshold.
    labels = []
    for prob in probs:
        clusters = np.where(prob > threshold)[0].tolist()
        if not clusters:
            clusters = [np.argmax(prob)]
        labels.append(clusters)
    
    return labels, k


def cluster_nodes(embeddings, dim=10, threshold=0.1):
    """The main RAPTOR clustering logic. It does a global clustering pass first, and then refines things locally."""
    n = len(embeddings)
    if n <= 1:
        return [[0]] * n
    
    # When we only have a few nodes, spectral initialization can crash, so we just use random initialization.
    init_mode = "spectral"
    if n < 15:
        init_mode = "random"
    
    # First, let's look at the big picture and cluster globally.
    global_dim = min(dim, max(1, n - 2))
    global_neighbors = max(2, int(np.sqrt(n)))
    
    reduced_global = umap.UMAP(
        n_neighbors=min(global_neighbors, n-1),
        n_components=global_dim,
        metric="cosine",
        init=init_mode,  # This fixes the crashes when n is small
        random_state=42
    ).fit_transform(embeddings)
    
    global_labels, n_global = gmm_cluster(reduced_global, threshold)
    
    # Now we zoom into each big cluster and break them down into smaller, tighter groups.
    final_labels = [[] for _ in range(n)]
    total_clusters = 0
    
    for g in range(n_global):
        indices = [i for i, labels in enumerate(global_labels) if g in labels]
        
        if not indices: continue
        
        cluster_size = len(indices)
        
        # If the cluster is already super small, we don't need to break it down any further.
        if cluster_size <= dim + 1:
            for i in indices:
                final_labels[i].append(total_clusters)
            total_clusters += 1
        else:
            local_emb = embeddings[indices]
            
            local_dim = min(dim, max(1, cluster_size - 2))
            local_neighbors = max(2, int(np.sqrt(cluster_size)))
            
            local_init = "random" if cluster_size < 15 else "spectral"
            
            reduced_local = umap.UMAP(
                n_neighbors=min(local_neighbors, cluster_size-1),
                n_components=local_dim,
                metric="cosine",
                init=local_init,
                random_state=42
            ).fit_transform(local_emb)
            
            local_labels, n_local = gmm_cluster(reduced_local, threshold)
            
            for j, i in enumerate(indices):
                for c in local_labels[j]:
                    final_labels[i].append(c + total_clusters)
            total_clusters += n_local
    
    return final_labels


# ============ TREE BUILDING LOGIC ============

def build_tree(chunks, embedder, summarizer, min_nodes=5):
    """
    Constructs the entire hierarchical tree starting from the bottom leaf chunks and working its way up.
    We stop building parents when we have fewer than `min_nodes` left at the top.
    Returns the organized nodes and the layer structure.
    """
    nodes = {}
    levels = {}
    
    # At Level 0, we just have the raw text chunks exactly as they came in.
    texts = [c["text"] for c in chunks]
    embeddings = embedder(texts)
    
    current = []
    for i, chunk in enumerate(chunks):
        node_id = f"L0_{chunk['chunk_id']}"
        nodes[node_id] = {
            "text": chunk["text"],
            "embedding": embeddings[i],
            "children": [],
            "parents": [],
            "leaves": [chunk["chunk_id"]]
        }
        current.append(node_id)
    
    levels[0] = current
    level = 0
    
    # Keep clustering and summarizing until we reach the top of the tree.
    while len(current) > min_nodes:
        level += 1
        print(f"  Building Level {level} from {len(current)} nodes...")
        
        # Print out some helpful info so we can see what's happening.
        prompt_type = "LEAF_PROMPT (extractive)" if level == 1 else "SUMMARY_PROMPT (synthetic)"
        print(f"    > Using {prompt_type}")
        
        embs = np.array([nodes[nid]["embedding"] for nid in current])
        
        # Group the nodes at this level into clusters.
        print("    > Clustering nodes (UMAP + GMM)...")
        cluster_labels = cluster_nodes(embs)
        
        # Organize the IDs so we can work with each cluster directly.
        clusters = {}
        for nid, labels in zip(current, cluster_labels):
            for c in labels:
                if c not in clusters:
                    clusters[c] = []
                clusters[c].append(nid)
        
        print(f"    > Found {len(clusters)} clusters. Generating summaries...")
        
        # Sometimes the clustering algorithm fails to compress the data. If that happens, we need to break out so we don't get stuck in an endless loop.
        if len(clusters) >= len(current):
            print(f"    ! Warning: No reduction ({len(clusters)} clusters from {len(current)} nodes). Stopping early.")
            break

        # Now we summarize each cluster to create the next layer of parent nodes.
        parents = []
        sorted_clusters = sorted(clusters.keys())
        for c in tqdm(sorted_clusters, desc=f"    > Summarizing Level {level}", unit="cluster"):
            children = clusters[c]
            child_texts = [nodes[nid]["text"] for nid in children]
            
            # Pass the level index to the summarizer so it knows which prompt to use.
            summary = summarizer(child_texts, level)
            emb = embedder([summary])[0]
            
            # A parent needs to know every single raw leaf chunk it sits above, so we pass those up the chain.
            leaves = []
            for nid in children:
                leaves.extend(nodes[nid]["leaves"])
            
            parent_id = f"L{level}_C{c}"
            nodes[parent_id] = {
                "text": summary,
                "embedding": emb,
                "children": children,
                "parents": [],  # This is empty for now, but will be filled by whoever becomes the parent of this node
                "leaves": list(set(leaves))
            }
            parents.append(parent_id)
            
            # It's important that children know who their parent is, so we establish the relationship here.
            for child_id in children:
                nodes[child_id]["parents"].append(parent_id)
        
        levels[level] = parents
        current = parents
        print(f"  ✓ Level {level} complete: {len(parents)} summary nodes created.\n")
    
    return nodes, levels


#  SAVE AND LOAD UTILS

def save_tree(nodes, levels, cache_dir):
    """Saves the fully built tree and all its embeddings to your cache folder."""
    tree_dir = os.path.join(cache_dir, "summary_tree")
    os.makedirs(tree_dir, exist_ok=True)
    
    # Split the embeddings out from the text so the JSON files don't become massive.
    structure = {nid: {k: v for k, v in n.items() if k != "embedding"} 
                 for nid, n in nodes.items()}
    with open(os.path.join(tree_dir, "nodes.json"), "w") as f:
        json.dump(structure, f, indent=2)
    
    with open(os.path.join(tree_dir, "levels.json"), "w") as f:
        json.dump({str(k): v for k, v in levels.items()}, f, indent=2)
    
    # Save the heavy numpy embeddings in a separate compressed file.
    embs = {nid: n["embedding"] for nid, n in nodes.items()}
    np.savez(os.path.join(tree_dir, "embeddings.npz"), **embs)


def load_tree(cache_dir):
    """Reads the previously saved tree back into memory, re-attaching the embeddings."""
    tree_dir = os.path.join(cache_dir, "summary_tree")
    
    with open(os.path.join(tree_dir, "nodes.json")) as f:
        nodes = json.load(f)
    
    with open(os.path.join(tree_dir, "levels.json")) as f:
        levels = {int(k): v for k, v in json.load(f).items()}
    
    embs = np.load(os.path.join(tree_dir, "embeddings.npz"))
    for nid in nodes:
        nodes[nid]["embedding"] = embs[nid]
    
    return nodes, levels
