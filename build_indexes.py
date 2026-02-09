"""
extract_indexes.py

Combines indexing operations for the RAG pipeline:
1. FAISS Vector Index (for semantic search on Summary Tree).
2. Inverted Indexes (I_s2e, I_e2s) mapping Summaries <-> Entities.

Dependencies:
- Summary Tree (nodes.json, embeddings.npz) -> Created in Step 2.
- Entities (I_c2e.json) -> Created in Step 4.

Therefore, this script should run AFTER Entity Extraction.
"""

import os
import json
import numpy as np
import faiss

def build_faiss_index(book_cache_dir):
    """
    Load tree embeddings and build/save a FAISS index.
    """
    print(f"  [FAISS] Building index for: {book_cache_dir}")
    tree_dir = os.path.join(book_cache_dir, "summary_tree")
    emb_path = os.path.join(tree_dir, "embeddings.npz")
    
    if not os.path.exists(emb_path):
        print(f"    ! Error: No embeddings found at {emb_path}")
        return

    # 1. Load Data
    data = np.load(emb_path)
    # The .npz stores keys as 'L0_chunk_0', 'L1_C0', etc.
    # We need to guarantee a fixed order for FAISS (which uses int IDs 0, 1, 2...)
    node_ids = sorted(data.files) 
    
    # Create matrix (N, D)
    embeddings = np.array([data[nid] for nid in node_ids]).astype('float32')
    
    # 2. Normalize for Cosine Similarity
    faiss.normalize_L2(embeddings)

    # 3. Create Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # Inner Product (Cosine sim after norm)
    index.add(embeddings)
    
    # 4. Save Index & ID Mapping
    # Saving to summary_tree folder
    index_path = os.path.join(tree_dir, "tree.index")
    mapping_path = os.path.join(tree_dir, "faiss_id_map.json")
    
    faiss.write_index(index, index_path)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(node_ids, f, indent=2)
        
    print(f"    FAISS Index saved ({len(node_ids)} vectors)")


def build_inverted_indexes(book_cache_dir):
    """
    Builds summary-entity indexes: I_s2e (Summary->Entities) and I_e2s (Entity->Summaries).
    """
    print(f"  [Inverted] Building summary-entity indexes...")
    
    # Load Tree Nodes
    tree_path = os.path.join(book_cache_dir, "summary_tree", "nodes.json")
    if not os.path.exists(tree_path):
        print(f"    ! Error: No tree nodes found at {tree_path}")
        return

    with open(tree_path, "r", encoding="utf-8") as f:
        tree_nodes = json.load(f)
    
    # Load Entity Map
    entities_path = os.path.join(book_cache_dir, "entities", "I_c2e.json")
    if not os.path.exists(entities_path):
        print(f"    ! Error: No entity map found at {entities_path}. (Did you run Entity Extraction?)")
        return

    with open(entities_path, "r", encoding="utf-8") as f:
        I_c2e = json.load(f)
    
    # Build I_s2e
    I_s2e = {}
    for node_id, node_data in tree_nodes.items():
        leaf_chunks = node_data.get("leaves", [])
        entities = set()
        for chunk_id in leaf_chunks:
            lookup_id = chunk_id
            if lookup_id not in I_c2e:
                # Try removing prefix (if present)
                lookup_id = chunk_id.replace("L0_", "")
            
            if lookup_id not in I_c2e:
                # Try adding prefix (if missing)
                lookup_id = f"L0_{chunk_id}"
                
            for entity in I_c2e.get(lookup_id, []):
                entities.add(entity)
        I_s2e[node_id] = list(entities)
    
    # Build I_e2s
    I_e2s = {}
    for node_id, entities in I_s2e.items():
        for entity in entities:
            if entity not in I_e2s:
                I_e2s[entity] = []
            I_e2s[entity].append(node_id)
    
    # Save to summary_tree folder
    indexes_dir = os.path.join(book_cache_dir, "summary_tree")
    os.makedirs(indexes_dir, exist_ok=True)
    
    with open(os.path.join(indexes_dir, "I_s2e.json"), "w", encoding="utf-8") as f:
        json.dump(I_s2e, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(indexes_dir, "I_e2s.json"), "w", encoding="utf-8") as f:
        json.dump(I_e2s, f, indent=2, ensure_ascii=False)
    
    print(f"    Inverted Indexes saved to {indexes_dir}")
    print(f"    Stats: {len(I_s2e)} summaries, {len(I_e2s)} entities mapped.")


def extract_all_indexes(book_cache_dir):
    """
    Main entry point to run both indexing steps for a single book.
    """
    build_faiss_index(book_cache_dir)
    build_inverted_indexes(book_cache_dir)


def main():
    # Standalone execution for ALL books in cache
    BASE_CACHE_DIR = os.path.abspath("./cache/InfiniteChoice")
    
    if not os.path.exists(BASE_CACHE_DIR):
        print(f"Error: Base directory not found at {BASE_CACHE_DIR}")
        return

    try:
        book_ids = sorted(
            [d for d in os.listdir(BASE_CACHE_DIR) if os.path.isdir(os.path.join(BASE_CACHE_DIR, d))],
            key=lambda x: int(x) if x.isdigit() else x
        )
    except ValueError:
        book_ids = sorted([d for d in os.listdir(BASE_CACHE_DIR) if os.path.isdir(os.path.join(BASE_CACHE_DIR, d))])
    
    print(f"Found {len(book_ids)} books to process.")

    for book_id in book_ids:
        print(f"\n[{book_id}] Processing Indexes...")
        book_cache = os.path.join(BASE_CACHE_DIR, book_id)
        extract_all_indexes(book_cache)
    
    print("\nDone.")

if __name__ == "__main__":
    main()
