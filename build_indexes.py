"""
build_indexes.py

Builds FAISS vector index and inverted indexes (summary-entity
mappings) for each book. Requires the summary tree and entity
extraction to have been completed first.
"""

import os
import json
import numpy as np
import faiss


def build_faiss_index(book_cache_dir):
    """Load tree embeddings and build a FAISS index."""
    print(f"  Building FAISS index for: {book_cache_dir}")
    tree_dir = os.path.join(book_cache_dir, "summary_tree")
    emb_path = os.path.join(tree_dir, "embeddings.npz")

    if not os.path.exists(emb_path):
        print(f"    No embeddings found at {emb_path}")
        return

    # load embeddings and sort node IDs for a stable ordering
    data = np.load(emb_path)
    node_ids = sorted(data.files)
    embeddings = np.array([data[nid] for nid in node_ids]).astype("float32")

    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # build and save
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    index_path = os.path.join(tree_dir, "tree.index")
    mapping_path = os.path.join(tree_dir, "faiss_id_map.json")

    faiss.write_index(index, index_path)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(node_ids, f, indent=2)

    print(f"    FAISS index saved ({len(node_ids)} vectors)")


def build_inverted_indexes(book_cache_dir):
    """Build I_s2e (summary -> entities) and I_e2s (entity -> summaries)."""
    print(f"  Building inverted indexes...")

    tree_path = os.path.join(book_cache_dir, "summary_tree", "nodes.json")
    if not os.path.exists(tree_path):
        print(f"    No tree nodes found at {tree_path}")
        return

    with open(tree_path, "r", encoding="utf-8") as f:
        tree_nodes = json.load(f)

    entities_path = os.path.join(book_cache_dir, "entities", "I_c2e.json")
    if not os.path.exists(entities_path):
        print(f"    No entity map found at {entities_path}")
        return

    with open(entities_path, "r", encoding="utf-8") as f:
        I_c2e = json.load(f)

    # for each tree node, collect all entities from its leaf chunks
    I_s2e = {}
    for node_id, node_data in tree_nodes.items():
        leaf_chunks = node_data.get("leaves", [])
        entities = set()
        for chunk_id in leaf_chunks:
            lookup_id = chunk_id
            if lookup_id not in I_c2e:
                lookup_id = chunk_id.replace("L0_", "")
            if lookup_id not in I_c2e:
                lookup_id = f"L0_{chunk_id}"
            for entity in I_c2e.get(lookup_id, []):
                entities.add(entity)
        I_s2e[node_id] = list(entities)

    # reverse the mapping
    I_e2s = {}
    for node_id, entities in I_s2e.items():
        for entity in entities:
            if entity not in I_e2s:
                I_e2s[entity] = []
            I_e2s[entity].append(node_id)

    # save
    indexes_dir = os.path.join(book_cache_dir, "summary_tree")
    os.makedirs(indexes_dir, exist_ok=True)

    with open(os.path.join(indexes_dir, "I_s2e.json"), "w", encoding="utf-8") as f:
        json.dump(I_s2e, f, indent=2, ensure_ascii=False)

    with open(os.path.join(indexes_dir, "I_e2s.json"), "w", encoding="utf-8") as f:
        json.dump(I_e2s, f, indent=2, ensure_ascii=False)

    print(f"    Saved: {len(I_s2e)} summaries, {len(I_e2s)} entities")


def extract_all_indexes(book_cache_dir):
    """Run both indexing steps for one book."""
    build_faiss_index(book_cache_dir)
    build_inverted_indexes(book_cache_dir)


def main():
    """Standalone: build indexes for all books in the cache."""
    base_dir = os.path.abspath("./cache/InfiniteChoice")

    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    try:
        book_ids = sorted(
            [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))],
            key=lambda x: int(x) if x.isdigit() else x,
        )
    except ValueError:
        book_ids = sorted(
            [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        )

    print(f"Found {len(book_ids)} books.")

    for book_id in book_ids:
        print(f"\n[{book_id}] Building indexes...")
        extract_all_indexes(os.path.join(base_dir, book_id))

    print("\nDone.")


if __name__ == "__main__":
    main()
