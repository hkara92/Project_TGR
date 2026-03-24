"""
This script takes all the embeddings we generated for our summary tree 
and packs them into a fast FAISS vector search index. 
It also builds some look-up dictionaries so we can quickly see which entities belong to which summary nodes without searching the graph every time.
Note: You must run the summary tree and entity extraction steps before running this!
"""

import os
import json
import numpy as np
import faiss


def build_faiss_index(book_cache_dir):
    """Loads up all the numpy embeddings we saved earlier and shoves them into a FAISS Inner-Product index for super fast similarity searches."""
    print(f"  Building FAISS index for: {book_cache_dir}")
    tree_dir = os.path.join(book_cache_dir, "summary_tree")
    emb_path = os.path.join(tree_dir, "embeddings.npz")

    if not os.path.exists(emb_path):
        print(f"    No embeddings found at {emb_path}")
        return

    # We need to sort the node IDs alphabetically so that the order of our vectors in the index stays consistent.
    data = np.load(emb_path)
    node_ids = sorted(data.files)
    embeddings = np.array([data[nid] for nid in node_ids]).astype("float32")

    # We normalize the vectors first. Because we use an Inner Product (FlatIP) FAISS index, normalizing turns the search into Cosine Similarity.
    faiss.normalize_L2(embeddings)

    # Create the index and give it the correct vector dimensions.
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
    """
    Builds two quick-reference dictionaries:
    One to find all entities mentioned underneath a higher-level summary node.
    One to find all summary nodes that contain a specific entity somewhere beneath them.
    """
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

    # We walk through every parent node in the tree and gather up every single entity mentioned by any leaf chunk underneath it.
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

    # Now we just flip that dictionary around so we can search it the other way.
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
    """A simple helper to run both the FAISS vector indexing and the dictionary mapping in one go."""
    build_faiss_index(book_cache_dir)
    build_inverted_indexes(book_cache_dir)


def main():
    """If you run this script directly, it will loop through the cache and build the indexes for every book it finds."""
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
