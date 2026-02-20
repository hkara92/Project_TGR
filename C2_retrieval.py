"""
C2_retrieval.py

Region-restricted graph retrieval. Identifies relevant narrative
regions via the summary tree, selects seed entities, traverses
the graph within those regions, and reranks the resulting chunks
with a cross-encoder.
"""

import os
import re
import json
import logging
import numpy as np
import faiss
import spacy
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "tree_top_m": 10,
    "tree_rerank_top_m": 10,
    "use_tree_rerank": True,
    "max_hops": 1,
    "max_chunks_per_entity": 10,
    "max_candidates": 50,
    "final_top_k": 10,
    "min_seeds": 1,
    "min_candidates": 3,
}

NER_SKIP_WORDS = {"he", "she", "it", "they", "we", "i", "you", "this", "that"}
NER_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "EVENT"}


def extract_entities(text, nlp):
    """Extract and normalize named entities from text."""
    if "\nA. " in text:
        text = text.split("\nA. ")[0]

    doc = nlp(text)
    entities = []
    seen = set()

    for ent in doc.ents:
        if ent.label_ not in NER_LABELS:
            continue

        name = re.sub(r"\s+", " ", ent.text.lower().strip())
        name = re.sub(r"^[\"']+|[\"']+$", "", name)

        if len(name) >= 3 and name not in NER_SKIP_WORDS and name not in seen:
            entities.append(name)
            seen.add(name)

    return entities


def tree_pruning(query, faiss_index, node_id_list, embedder_func,
                 tree_nodes, cross_encoder, config):
    """
    Find the most relevant summary nodes via FAISS search,
    optionally reranked with a cross-encoder.
    """
    query_vec = np.array(embedder_func(query)).reshape(1, -1).astype("float32")
    faiss.normalize_L2(query_vec)
    distances, indices = faiss_index.search(query_vec, config["tree_top_m"] * 10)

    candidates = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(node_id_list):
            node_id = node_id_list[idx]
            if not node_id.startswith("L0_"):
                candidates.append((node_id, float(distances[0][i])))

    # optional cross-encoder rerank
    if config["use_tree_rerank"] and cross_encoder:
        pairs = [(query, tree_nodes.get(nid, {}).get("text", "")) for nid, _ in candidates]
        scores = cross_encoder.predict(pairs)
        candidates = sorted(
            zip([c[0] for c in candidates], scores),
            key=lambda x: x[1], reverse=True,
        )

    reranked_top = [cid for cid, _ in candidates[:config["tree_rerank_top_m"]]]
    print(f"\n[DEBUG] FAISS candidates: {len(candidates)}")
    print(f"[DEBUG] FAISS top {config['tree_top_m']}: {[c[0] for c in candidates[:config['tree_top_m']]]}")
    print(f"[DEBUG] Reranked top {config['tree_rerank_top_m']}: {reranked_top}")

    return reranked_top


def define_region(summary_ids, tree_nodes, I_s2e):
    """
    Expand selected summaries into a region of allowed chunk IDs
    and allowed entity names.
    """
    C_region = set()
    E_region = set()

    for summary_id in summary_ids:
        leaves = tree_nodes.get(summary_id, {}).get("leaves", [])
        for leaf in leaves:
            chunk_id = leaf[3:] if leaf.startswith("L0_") else leaf
            C_region.add(chunk_id)

        E_region.update(I_s2e.get(summary_id, []))

    print(f"[DEBUG] Region: {len(C_region)} chunks, {len(E_region)} entities")
    return C_region, E_region


def select_seeds(query_entities, E_region, C_region, I_c2e):
    """
    Pick starting entities for graph traversal.
    First tries intersection of query entities with region entities,
    falls back to most frequent entities in the region.
    """
    seeds = [e for e in query_entities if e in E_region]
    if seeds:
        print(f"[DEBUG] Seeds (intersection): {seeds}")
        return seeds

    logger.info("No direct seeds, falling back to entity frequency.")
    entity_freq = defaultdict(int)
    for chunk_id in C_region:
        for ent in I_c2e.get(chunk_id, []):
            if ent in E_region:
                entity_freq[ent] += 1

    seeds = [e for e, _ in sorted(entity_freq.items(), key=lambda x: x[1], reverse=True)[:3]]
    print(f"[DEBUG] Seeds (frequency): {seeds}")
    return seeds


def traverse_graph(seeds, E_region, C_region, book_id, neo4j_driver):
    """
    Run a Cypher query to find 1-hop neighbors of the seed entities,
    constrained to the region.
    """
    query = """
    MATCH (seed:Entity)-[r:RELATION]-(neighbor:Entity)
    WHERE seed.book_id = $book_id
      AND seed.name IN $seeds
      AND neighbor.name IN $e_region
    WITH seed.name AS source,
         neighbor.name AS target,
         r.relation AS relation,
         r.weight AS weight,
         [c IN r.chunk_ids WHERE c IN $c_region] AS evidence
    WHERE SIZE(evidence) > 0
    RETURN source, target, relation, weight, evidence
    """

    with neo4j_driver.session() as session:
        result = session.run(
            query,
            book_id=book_id,
            seeds=list(seeds),
            e_region=list(E_region),
            c_region=list(C_region),
        )
        records = list(result)
        visited = set(seeds)
        for r in records:
            visited.add(r["source"])
            visited.add(r["target"])
        print(f"[DEBUG] Visited entities ({len(visited)}): {list(visited)}")
        return records


def collect_candidates(records, visited_entities, C_region,
                       I_e2c, I_c2e, query_entities, tree_nodes, config):
    """
    Build a list of candidate chunks from graph edges and entity
    mentions, all within the region.
    """
    chunk_data = {}
    query_entity_set = set(query_entities)
    relations = []

    # chunks that appear as edge evidence
    for rec in records:
        s, t = rec["source"], rec["target"]
        evidence = rec["evidence"]
        visited_entities.add(s)
        visited_entities.add(t)

        relations.append({
            "source": s, "target": t,
            "relation": rec["relation"],
            "evidence_chunks": evidence,
        })

        for cid in evidence:
            if cid not in chunk_data:
                chunk_data[cid] = {"is_edge_evidence": True, "sources": {"edge"}}
            else:
                chunk_data[cid]["is_edge_evidence"] = True
                chunk_data[cid]["sources"].add("edge")

    # chunks from entity mentions
    for entity in visited_entities:
        valid_chunks = [c for c in I_e2c.get(entity, []) if c in C_region]
        for cid in valid_chunks[:config["max_chunks_per_entity"]]:
            if cid not in chunk_data:
                chunk_data[cid] = {"is_edge_evidence": False, "sources": {"entity"}}
            else:
                chunk_data[cid]["sources"].add("entity")

    # build candidate list
    candidates = []
    for cid, data in chunk_data.items():
        node_id = f"L0_{cid}" if not cid.startswith("L0_") else cid
        text = tree_nodes.get(node_id, {}).get("text") or tree_nodes.get(cid, {}).get("text")
        if not text:
            continue

        chunk_ents = set(I_c2e.get(cid, []))
        q_count = len(query_entity_set & chunk_ents)

        candidates.append({
            "chunk_id": cid,
            "text": text,
            "sources": data["sources"],
            "is_edge_evidence": data.get("is_edge_evidence", False),
            "query_entity_count": q_count,
            "score": 0.0,
        })

    candidates.sort(key=lambda x: (x["is_edge_evidence"], x["query_entity_count"]), reverse=True)
    print(f"[DEBUG] Candidates collected: {len(candidates)}")
    return candidates[:config["max_candidates"]]


def rerank_chunks(query, candidates, cross_encoder, top_k):
    """Score candidates with the cross-encoder and return the top k."""
    if not candidates:
        return []

    pairs = [(query, c["text"]) for c in candidates]
    scores = cross_encoder.predict(pairs)

    for i, c in enumerate(candidates):
        c["score"] = float(scores[i])

    return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]


def tree_only_fallback(query, C_region, I_c2e, query_entities,
                       tree_nodes, cross_encoder, config):
    """Fallback: rerank all chunks in the region directly."""
    logger.info("Using tree-only fallback.")

    candidates = []
    query_entity_set = set(query_entities)

    for cid in C_region:
        node_id = f"L0_{cid}" if not cid.startswith("L0_") else cid
        text = tree_nodes.get(node_id, {}).get("text") or tree_nodes.get(cid, {}).get("text")
        if not text:
            continue

        q_count = len(query_entity_set & set(I_c2e.get(cid, [])))
        candidates.append({
            "chunk_id": cid, "text": text,
            "query_entity_count": q_count, "score": 0.0,
        })

    candidates.sort(key=lambda x: x["query_entity_count"], reverse=True)
    candidates = candidates[:config["max_candidates"]]

    return rerank_chunks(query, candidates, cross_encoder, config["final_top_k"])


def build_result(top_chunks, query_entities, seeds, mode, stats):
    """Package the retrieval output."""
    context = "\n\n".join([c["text"] for c in top_chunks])
    return {
        "chunks": top_chunks,
        "context": context,
        "query_entities": query_entities,
        "seeds": list(seeds),
        "retrieval_mode": mode,
        "stats": stats,
    }


# main entry point

def retrieve(question, options, resources, config=None, verbose=False):
    """
    Run the full region-restricted retrieval pipeline for a single
    question. Returns a dict with chunks, context, and metadata.
    """
    config = config or DEFAULT_CONFIG
    stats = {}

    faiss_index = resources["faiss_index"]
    node_id_list = resources["node_id_list"]
    tree_nodes = resources["tree_nodes"]
    I_s2e = resources["I_s2e"]
    I_e2s = resources["I_e2s"]
    I_e2c = resources["I_e2c"]
    I_c2e = resources["I_c2e"]
    neo4j_driver = resources["neo4j_driver"]
    book_id = resources["book_id"]
    embedder_func = resources["embedder_func"]
    nlp = resources["nlp"]
    cross_encoder = resources["cross_encoder"]

    # dense query includes options for better embedding match
    opts_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]) if options else ""
    dense_query = f"{question}\n{opts_text}" if options else question

    query_entities = extract_entities(question, nlp)
    stats["query_entities"] = query_entities

    # identify region
    top_summaries = tree_pruning(
        dense_query, faiss_index, node_id_list,
        embedder_func, tree_nodes, cross_encoder, config,
    )
    C_region, E_region = define_region(top_summaries, tree_nodes, I_s2e)
    stats["region_size"] = len(C_region)

    # pick seeds
    seeds = select_seeds(query_entities, E_region, C_region, I_c2e)
    stats["seeds"] = seeds

    if len(seeds) < config["min_seeds"]:
        return build_result(
            tree_only_fallback(dense_query, C_region, I_c2e, query_entities,
                               tree_nodes, cross_encoder, config),
            query_entities, seeds, "fallback_no_seeds", stats,
        )

    # graph traversal
    try:
        records = traverse_graph(seeds, E_region, C_region, book_id, neo4j_driver)
    except Exception as e:
        logger.error(f"Graph traversal failed: {e}")
        records = []

    stats["relations_found"] = len(records)

    # collect and filter candidates
    visited = set(seeds)
    candidates = collect_candidates(
        records, visited, C_region, I_e2c, I_c2e,
        query_entities, tree_nodes, config,
    )
    stats["candidates"] = len(candidates)

    if len(candidates) < config["min_candidates"]:
        return build_result(
            tree_only_fallback(dense_query, C_region, I_c2e, query_entities,
                               tree_nodes, cross_encoder, config),
            query_entities, seeds, "fallback_no_candidates", stats,
        )

    # final rerank
    top_chunks = rerank_chunks(dense_query, candidates, cross_encoder, config["final_top_k"])
    print(f"[DEBUG] Final selection: {len(top_chunks)} chunks")

    return build_result(top_chunks, query_entities, seeds, "graph_traversal", stats)


def load_retriever(cache_dir, book_id, neo4j_uri, neo4j_user, neo4j_password,
                   embedder_func, spacy_model="en_core_web_lg",
                   cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Load all resources needed by the retriever for one book."""
    from neo4j import GraphDatabase
    from sentence_transformers import CrossEncoder

    logger.info(f"Loading retriever for {book_id}...")

    def load_json(fname):
        with open(os.path.join(cache_dir, fname), "r", encoding="utf-8") as f:
            return json.load(f)

    tree_nodes = load_json("summary_tree/nodes.json")
    node_id_list = load_json("summary_tree/faiss_id_map.json")
    I_s2e = load_json("summary_tree/I_s2e.json")
    I_e2s = load_json("summary_tree/I_e2s.json")
    I_e2c = load_json("entities/I_e2c.json")
    I_c2e = load_json("entities/I_c2e.json")

    faiss_index = faiss.read_index(os.path.join(cache_dir, "summary_tree", "tree.index"))

    neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    with neo4j_driver.session() as session:
        session.run("RETURN 1")

    return {
        "faiss_index": faiss_index,
        "node_id_list": node_id_list,
        "tree_nodes": tree_nodes,
        "I_s2e": I_s2e, "I_e2s": I_e2s, "I_e2c": I_e2c, "I_c2e": I_c2e,
        "neo4j_driver": neo4j_driver,
        "book_id": book_id,
        "embedder_func": embedder_func,
        "nlp": spacy.load(spacy_model),
        "cross_encoder": CrossEncoder(cross_encoder_model),
    }
