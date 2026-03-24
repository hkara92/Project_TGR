"""
Region-restricted retrieval using batched shortest-path queries in Neo4j.
This is our primary C2 approach used for the thesis evaluation.
"""

import os
import re
import json
import logging
import numpy as np
import faiss
import spacy
from rapidfuzz import fuzz as _fuzz

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "tree_top_m": 10,               # how many summary nodes to keep after cross-encoder reranking
    "use_tree_rerank": True,        # whether to use the cross-encoder on summary nodes
    "max_candidates": 50,           # max chunks in the raw candidate pool before MMR
    "max_chunks_per_entity": 10,    # max mention chunks per entity
    "use_entity_mention_chunks": True,  # also include chunks that just mention visited entities
    "mmr_top_k": 20,                # how many chunks MMR should output
    "mmr_lambda": 0.5,              # balances relevance vs diversity (higher = more relevance)
    "final_top_k": 15,              # final chunks sent to the LLM
    "min_candidates": 3,            # if fewer than this, trigger the global fallback
    "fuzzy_threshold": 85,          # minimum fuzzy score to accept an entity match
    "fuzzy_min_length": 5,          # skip very short entity names during fuzzy matching

    # Shortest-path specific
    "key_top_k": 50,                # how many key target entities to select for pathfinding
    "entity_topn_chunks": 10,       # top-N chunks per entity used when ranking key targets
    "shortest_max_hops": 4,         # max hops allowed in the shortestPath Cypher query
}

NER_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "EVENT"}
NER_SKIP   = {"he", "she", "it", "they", "we", "i", "you", "this", "that"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_options(text):
    """Removes appended A/B/C/D options so we only work with the question text."""
    return text.split("\nA. ")[0].strip() if "\nA. " in text else text.strip()


def extract_entities(text, nlp):
    """Runs SpaCy NER on the question and cleans up the entity names."""
    doc = nlp(strip_options(text))
    seen, entities = set(), []
    for ent in doc.ents:
        if ent.label_ not in NER_LABELS:
            continue
        name = re.sub(r"\s+", " ", ent.text.lower().strip())
        name = re.sub(r"^[\"']+|[\"']+$", "", name)
        if len(name) >= 3 and name not in NER_SKIP and name not in seen:
            entities.append(name)
            seen.add(name)
    return entities


def fuzzy_match(query_entities, E_region, threshold=85, min_len=5):
    """Finds the closest matching entity in the region for each query entity."""
    matched = set()
    for qe in query_entities:
        if len(qe) < min_len:
            continue
        best, best_score = None, 0
        for re_ in E_region:
            s = _fuzz.token_sort_ratio(qe, re_)
            if s > best_score:
                best_score, best = s, re_
        if best_score >= threshold and best:
            matched.add(best)
    return list(matched)


# ---------------------------------------------------------------------------
# MMR (Maximal Marginal Relevance)
# ---------------------------------------------------------------------------

def mmr_select(query, candidates, embedder_func, top_k, lambda_param=0.5, chunk_embs=None):
    """Picks a diverse yet relevant set of chunks to avoid sending redundant info to the LLM."""
    if len(candidates) <= top_k:
        return candidates

    q_emb = np.array(embedder_func([query])[0]).astype("float32")
    q_emb = q_emb / max(np.linalg.norm(q_emb), 1e-9)

    if chunk_embs is not None:
        emb_list = []
        for c in candidates:
            cid = c["chunk_id"]
            emb = chunk_embs.get(cid)
            if emb is None:
                emb = np.array(embedder_func([c["text"]])[0]).astype("float32")
            emb_list.append(emb)
        embs = np.array(emb_list).astype("float32")
    else:
        embs = np.array(embedder_func([c["text"] for c in candidates])).astype("float32")

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs  = embs / np.where(norms == 0, 1e-9, norms)

    relevance = embs @ q_emb

    n = len(candidates)
    selected, remaining = [], list(range(n))
    for _ in range(top_k):
        if not remaining:
            break
        if not selected:
            best = remaining[int(np.argmax(relevance[remaining]))]
        else:
            sel_embs = embs[selected]
            best, best_score = None, float("-inf")
            for i in remaining:
                max_sim = float((embs[i] @ sel_embs.T).max())
                score   = lambda_param * relevance[i] - (1 - lambda_param) * max_sim
                if score > best_score:
                    best_score, best = score, i
        selected.append(best)
        remaining.remove(best)

    print(f"[MMR] {n} -> {len(selected)} chunks (lambda={lambda_param}, precomputed={'yes' if chunk_embs else 'no'})")
    return [candidates[i] for i in selected]


# ---------------------------------------------------------------------------
# Lost in the Middle
# ---------------------------------------------------------------------------

def lost_in_middle_reorder(chunks):
    """Reorders so the best chunks sit at the start and end, preventing the LLM from ignoring them."""
    if len(chunks) <= 2:
        return chunks
    result, l, r = [None] * len(chunks), 0, len(chunks) - 1
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            result[l] = chunk; l += 1
        else:
            result[r] = chunk; r -= 1
    print(f"[LostInMiddle] Reordered {len(result)} chunks.")
    return result


# ---------------------------------------------------------------------------
# Tree pruning
# ---------------------------------------------------------------------------

def tree_pruning(dense_query, question_only, faiss_index, node_id_list,
                 embedder_func, tree_nodes, cross_encoder, config):
    """Finds the most relevant summary nodes using FAISS, then refines them with the cross-encoder."""
    qv = np.array(embedder_func([dense_query])[0]).reshape(1, -1).astype("float32")
    faiss.normalize_L2(qv)
    _, indices = faiss_index.search(qv, config["tree_top_m"] * 10)

    # Only keep summary nodes, skip raw L0 chunks
    candidates = [
        node_id_list[i]
        for i in indices[0]
        if 0 <= i < len(node_id_list) and not node_id_list[i].startswith("L0_")
    ]
    print(f"[TreePruning] FAISS -> {len(candidates)} summary node candidates.")

    if not candidates:
        return []

    if config["use_tree_rerank"] and cross_encoder:
        pairs  = [(question_only, tree_nodes.get(nid, {}).get("text", "")) for nid in candidates]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top    = [nid for nid, _ in ranked[:config["tree_top_m"]]]
    else:
        top = candidates[:config["tree_top_m"]]

    print(f"[TreePruning] Cross-encoder selected {len(top)} summary nodes: {top}")
    return top


# ---------------------------------------------------------------------------
# Region definition
# ---------------------------------------------------------------------------

def define_region(summary_ids, tree_nodes, I_s2e):
    """Expands summary nodes into their leaf chunks (C_region) and associated entities (E_region)."""
    C_region, E_region = set(), set()
    for sid in summary_ids:
        for leaf in tree_nodes.get(sid, {}).get("leaves", []):
            C_region.add(leaf if leaf.startswith("L0_") else f"L0_{leaf}")
        E_region.update(I_s2e.get(sid, []))
    print(f"[Region] C_region={len(C_region)} chunks, E_region={len(E_region)} entities")
    return C_region, E_region


# ---------------------------------------------------------------------------
# Seed selection
# ---------------------------------------------------------------------------

def select_seeds(query_entities, E_region, config):
    """Finds starting points for graph traversal by matching query entities into the region."""
    if not query_entities:
        print("[Seeds] No query entities.")
        return [], "no_query_entities"
    seeds = fuzzy_match(query_entities, E_region,
                        threshold=config["fuzzy_threshold"],
                        min_len=config["fuzzy_min_length"])
    if seeds:
        print(f"[Seeds] Fuzzy matched: {seeds}")
        return seeds, "fuzzy_match"
    print(f"[Seeds] No match for {query_entities}.")
    return [], "no_match"


# ---------------------------------------------------------------------------
# Key entity ranking (shortest-path mode)
# ---------------------------------------------------------------------------

def rank_key_entities(question_only, E_region, I_e2c, chunk_embs, embedder_func,
                      top_k=50, topn_chunks=10, exclude=set()):
    """Ranks entities in the region by how similar their chunks are to the query, picking the best targets for pathfinding."""
    if not E_region:
        return []

    if chunk_embs is None:
        chunk_embs = {}

    q_emb = np.array(embedder_func([question_only])[0]).astype("float32")
    q_emb = q_emb / max(np.linalg.norm(q_emb), 1e-9)

    scored = []
    for entity in E_region:
        if entity in exclude:
            continue

        chunk_ids = I_e2c.get(entity, [])
        if not chunk_ids:
            continue

        # Score each chunk by cosine similarity with the query
        sims = []
        for cid in chunk_ids:
            emb = chunk_embs.get(cid)
            if emb is None:
                continue
            emb = emb / max(np.linalg.norm(emb), 1e-9)
            sims.append(float(emb @ q_emb))

        if not sims:
            continue

        # Average the top-N most similar chunks as the entity's score
        sims.sort(reverse=True)
        top = sims[:topn_chunks]
        score = sum(top) / len(top)
        scored.append((entity, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_entities = [e for e, _ in scored[:top_k]]
    print(f"[KeyTargets] Selected {len(top_entities)} key targets (top_k={top_k}).")
    return top_entities


# ---------------------------------------------------------------------------
# Shortest path (batched UNWIND)
# ---------------------------------------------------------------------------

def shortest_path_batch(seeds, key_targets, C_region, book_id, driver, max_hops=4):
    """Runs batched shortest-path queries between our seeds and key targets, collecting evidence chunks along the way."""
    pairs = [{"seed": s, "target": t} for s in seeds for t in key_targets if s != t]
    if not pairs:
        return [], set(seeds)

    max_hops = int(max_hops)
    if max_hops < 1:
        max_hops = 1

    # Neo4j does not allow parameterized hop limits, so we inject it via f-string
    cypher = f"""
    UNWIND $pairs AS p
    MATCH (a:Entity {{book_id:$book_id, name:p.seed}})
    MATCH (b:Entity {{book_id:$book_id, name:p.target}})
    MATCH path = shortestPath((a)-[:RELATION*..{max_hops}]-(b))
    WHERE ALL(r IN relationships(path) WHERE r.book_id = $book_id)
    WITH p, relationships(path) AS rels
    UNWIND rels AS r
    WITH p.seed AS seed, p.target AS target,
         r.relation AS relation,
         [c IN r.chunk_ids WHERE c IN $c_region] AS evidence
    WHERE SIZE(evidence) > 0
    RETURN seed, target, relation, evidence
    """

    params = {
        "pairs": pairs,
        "book_id": book_id,
        "c_region": list(C_region),
    }

    with driver.session() as session:
        rows = list(session.run(cypher, **params))

    visited = set(seeds)
    records = []
    for r in rows:
        visited.update([r["seed"], r["target"]])
        records.append({
            "source": r["seed"],
            "target": r["target"],
            "relation": r["relation"],
            "evidence": r["evidence"],
        })

    print(f"[ShortestPath] pairs={len(pairs)} -> edges={len(records)} visited={len(visited)}")
    return records, visited


# ---------------------------------------------------------------------------
# Candidate collection
# ---------------------------------------------------------------------------

def collect_candidates(records, visited, C_region, I_e2c, I_c2e,
                       query_entities, tree_nodes, config):
    """Gathers all text chunks referenced by our graph edges, plus optionally any chunks that mention visited entities."""
    chunk_data = {}

    # Chunks cited as evidence by graph edges
    for rec in records:
        for cid in rec["evidence"]:
            if cid not in chunk_data:
                chunk_data[cid] = {"is_edge_evidence": True, "sources": {"edge"}, "edge_count": 1}
            else:
                chunk_data[cid]["is_edge_evidence"] = True
                chunk_data[cid]["sources"].add("edge")
                chunk_data[cid]["edge_count"] = chunk_data[cid].get("edge_count", 0) + 1

    print(f"[Candidates] Edge evidence: {len(chunk_data)} chunks.")

    # Optionally add chunks that just mention entities we visited
    if config.get("use_entity_mention_chunks", True):
        for entity in visited:
            for cid in [c for c in I_e2c.get(entity, []) if c in C_region][:config["max_chunks_per_entity"]]:
                if cid not in chunk_data:
                    chunk_data[cid] = {"is_edge_evidence": False, "sources": {"entity"}}
                else:
                    chunk_data[cid]["sources"].add("entity")
        print(f"[Candidates] After entity mentions: {len(chunk_data)} total (deduplicated).")

    q_ents = set(query_entities)
    candidates = []
    for cid, data in chunk_data.items():
        text = tree_nodes.get(cid, {}).get("text") or tree_nodes.get(cid.replace("L0_", ""), {}).get("text")
        if not text:
            continue

        candidates.append({
            "chunk_id":            cid,
            "text":                text,
            "sources":             data["sources"],
            "is_edge_evidence":    data["is_edge_evidence"],
            "edge_evidence_count": data.get("edge_count", 0),
            "query_entity_count":  len(q_ents & set(I_c2e.get(cid, []))),
            "score":               0.0,
        })

    # Prioritize chunks backed by more edges and mentioning more query entities
    candidates.sort(
        key=lambda x: (x["is_edge_evidence"], x["edge_evidence_count"], x["query_entity_count"]),
        reverse=True
    )
    candidates = candidates[:config["max_candidates"]]
    print(f"[Candidates] Final pool: {len(candidates)} chunks.")
    return candidates


# ---------------------------------------------------------------------------
# Cross-encoder rerank
# ---------------------------------------------------------------------------

def rerank_chunks(question_only, candidates, cross_encoder, top_k):
    """Uses the cross-encoder to precisely score each chunk against the question."""
    if not candidates:
        return []
    pairs  = [(question_only, c["text"]) for c in candidates]
    scores = cross_encoder.predict(pairs)
    for i, c in enumerate(candidates):
        c["score"] = float(scores[i])
    result = sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]
    print(f"[CrossEncoder] {len(candidates)} -> top {len(result)} chunks.")
    return result


# ---------------------------------------------------------------------------
# Global fallback
# ---------------------------------------------------------------------------

def global_tree_dense_fallback(question_only, faiss_index, node_id_list, tree_nodes,
                                I_c2e, query_entities, embedder_func, cross_encoder,
                                config, chunk_embs=None, faiss_k_mult=20):
    """When the graph gives us nothing useful, we fall back to searching the entire RAPTOR tree with dense retrieval."""
    print("[Fallback] Using GLOBAL tree dense fallback (entire RAPTOR tree).")

    # Search the full FAISS index
    qv = np.array(embedder_func([question_only])[0]).reshape(1, -1).astype("float32")
    faiss.normalize_L2(qv)
    k = max(config["max_candidates"] * faiss_k_mult, 200)
    _, idx = faiss_index.search(qv, k)

    # Expand any summary nodes into their leaf chunks
    chunk_ids = []
    for i in idx[0]:
        if i < 0 or i >= len(node_id_list):
            continue
        nid = node_id_list[i]
        if nid.startswith("L0_"):
            chunk_ids.append(nid)
        else:
            leaves = tree_nodes.get(nid, {}).get("leaves", [])
            for leaf in leaves:
                chunk_ids.append(leaf if leaf.startswith("L0_") else f"L0_{leaf}")

    # Deduplicate while preserving ranking order
    seen, chunk_ids_dedup = set(), []
    for cid in chunk_ids:
        if cid not in seen:
            chunk_ids_dedup.append(cid)
            seen.add(cid)

    # Build candidate dicts from the deduped chunks
    q_ents = set(query_entities)
    candidates = []
    for cid in chunk_ids_dedup:
        text = tree_nodes.get(cid, {}).get("text") or tree_nodes.get(cid.replace("L0_", ""), {}).get("text")
        if not text:
            continue
        candidates.append({
            "chunk_id": cid,
            "text": text,
            "sources": {"global_fallback"},
            "is_edge_evidence": False,
            "edge_evidence_count": 0,
            "query_entity_count": len(q_ents & set(I_c2e.get(cid, []))),
            "score": 0.0,
        })
        if len(candidates) >= config["max_candidates"]:
            break

    print(f"[Fallback] Global candidate pool: {len(candidates)} chunks.")

    # Run MMR then cross-encoder on the fallback pool
    candidates = mmr_select(question_only, candidates, embedder_func,
                            config["mmr_top_k"], config["mmr_lambda"], chunk_embs=chunk_embs)

    return rerank_chunks(question_only, candidates, cross_encoder, config["final_top_k"])


# ---------------------------------------------------------------------------
# Result packaging
# ---------------------------------------------------------------------------

def build_result(chunks, query_entities, seeds, mode, stats):
    """Packages up the final chunks and metadata so they can be fed to the generator LLM."""
    for c in chunks:
        if isinstance(c.get("sources"), set):
            c["sources"] = list(c["sources"])
    return {
        "chunks":         chunks,
        "context":        "\n\n".join(c["text"] for c in chunks),
        "query_entities": query_entities,
        "seeds":          list(seeds),
        "retrieval_mode": mode,
        "stats":          stats,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def retrieve(question, options, resources, config=None):
    """Runs the full shortest-path retrieval pipeline for one question."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    stats = {}

    faiss_index   = resources["faiss_index"]
    node_id_list  = resources["node_id_list"]
    tree_nodes    = resources["tree_nodes"]
    I_s2e         = resources["I_s2e"]
    I_e2c         = resources["I_e2c"]
    I_c2e         = resources["I_c2e"]
    neo4j_driver  = resources["neo4j_driver"]
    book_id       = resources["book_id"]
    embedder_func = resources["embedder_func"]
    nlp           = resources["nlp"]
    cross_encoder = resources["cross_encoder"]
    chunk_embs    = resources.get("chunk_embs")

    dense_query   = question
    question_only = strip_options(question)

    print(f"\n{'='*60}")
    print(f"[Retrieve] Q: {question_only[:80]}...")

    # Extract named entities from the question
    query_entities          = extract_entities(question_only, nlp)
    stats["query_entities"] = query_entities
    print(f"[Entities] {query_entities}")

    # Find the best summary nodes using FAISS + cross-encoder
    top_summaries          = tree_pruning(dense_query, question_only, faiss_index,
                                          node_id_list, embedder_func, tree_nodes,
                                          cross_encoder, cfg)
    stats["top_summaries"] = top_summaries

    if not top_summaries:
        fb = global_tree_dense_fallback(
            question_only=question_only, faiss_index=faiss_index, node_id_list=node_id_list,
            tree_nodes=tree_nodes, I_c2e=I_c2e, query_entities=query_entities,
            embedder_func=embedder_func, cross_encoder=cross_encoder, config=cfg, chunk_embs=chunk_embs)
        return build_result(lost_in_middle_reorder(fb), query_entities,
                            [], "fallback_no_summaries", stats)

    # Define the allowed region from the selected summaries
    C_region, E_region       = define_region(top_summaries, tree_nodes, I_s2e)
    stats["region_chunks"]   = len(C_region)
    stats["region_entities"] = len(E_region)

    # Find seed entities in the region
    seeds, seed_strategy     = select_seeds(query_entities, E_region, cfg)
    stats["seeds"]           = seeds
    stats["seed_strategy"]   = seed_strategy

    if not seeds:
        fb = global_tree_dense_fallback(
            question_only=question_only, faiss_index=faiss_index, node_id_list=node_id_list,
            tree_nodes=tree_nodes, I_c2e=I_c2e, query_entities=query_entities,
            embedder_func=embedder_func, cross_encoder=cross_encoder, config=cfg, chunk_embs=chunk_embs)
        return build_result(lost_in_middle_reorder(fb), query_entities,
                            [], f"fallback_{seed_strategy}", stats)

    # Rank entities in the region to find the best pathfinding targets
    print(f"[ShortestPath] max_hops={cfg['shortest_max_hops']}  seeds={seeds}")
    key_targets = rank_key_entities(
        question_only=question_only,
        E_region=E_region,
        I_e2c=I_e2c,
        chunk_embs=chunk_embs,
        embedder_func=embedder_func,
        top_k=cfg["key_top_k"],
        topn_chunks=cfg["entity_topn_chunks"],
        exclude=set(seeds),
    )
    stats["key_targets"] = key_targets

    # Run batched shortest-path queries from seeds to key targets
    try:
        records, visited = shortest_path_batch(
            seeds=seeds,
            key_targets=key_targets,
            C_region=C_region,
            book_id=book_id,
            driver=neo4j_driver,
            max_hops=cfg["shortest_max_hops"],
        )
    except Exception as e:
        logger.error(f"Shortest path failed: {e}")
        records, visited = [], set(seeds)

    stats["relations_found"] = len(records)
    print(f"[ShortestPath] {len(records)} edges found, {len(visited)} entities visited")

    # Collect all text chunks referenced by the graph edges
    candidates                     = collect_candidates(records, visited, C_region,
                                                        I_e2c, I_c2e, query_entities,
                                                        tree_nodes, cfg)
    stats["candidates_before_mmr"] = len(candidates)
    print(f"[Candidates] {len(candidates)} after collection")

    if len(candidates) < cfg["min_candidates"]:
        print(f"[Candidates] Too few (<{cfg['min_candidates']}), using GLOBAL fallback.")
        fb = global_tree_dense_fallback(
            question_only=question_only, faiss_index=faiss_index, node_id_list=node_id_list,
            tree_nodes=tree_nodes, I_c2e=I_c2e, query_entities=query_entities,
            embedder_func=embedder_func, cross_encoder=cross_encoder, config=cfg, chunk_embs=chunk_embs)
        return build_result(lost_in_middle_reorder(fb), query_entities,
                            seeds, "fallback_no_candidates", stats)

    # Diversify with MMR
    candidates = mmr_select(question_only, candidates, embedder_func,
                            cfg["mmr_top_k"], cfg["mmr_lambda"], chunk_embs=chunk_embs)
    stats["candidates_after_mmr"] = len(candidates)
    print(f"[MMR] {len(candidates)} chunks after diversity filtering")

    # Final precision scoring with the cross-encoder
    top_chunks = rerank_chunks(question_only, candidates, cross_encoder, cfg["final_top_k"])
    stats["final_chunks"] = len(top_chunks)
    print(f"[CrossEncoder] {len(top_chunks)} chunks after reranking")

    # Reorder so the LLM doesn't lose track of the best chunks
    top_chunks = lost_in_middle_reorder(top_chunks)
    print(f"[Done] {len(top_chunks)} chunks -> sending to LLM")
    print(f"{'='*60}\n")
    return build_result(top_chunks, query_entities, seeds, "shortest_path", stats)


# ---------------------------------------------------------------------------
# Resource loader
# ---------------------------------------------------------------------------

def load_retriever(cache_dir, book_id, neo4j_uri, neo4j_user, neo4j_password,
                   embedder_func, spacy_model="en_core_web_lg",
                   cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Loads all the cached data, models, and Neo4j connection for one book."""
    from neo4j import GraphDatabase
    from sentence_transformers import CrossEncoder

    def load_json(path):
        with open(os.path.join(cache_dir, path), "r", encoding="utf-8") as f:
            return json.load(f)

    raw_embs = np.load(os.path.join(cache_dir, "summary_tree", "embeddings.npz"))
    chunk_embs = {nid: raw_embs[nid].astype("float32") for nid in raw_embs.files}

    resources = {
        "tree_nodes":    load_json("summary_tree/nodes.json"),
        "node_id_list":  load_json("summary_tree/faiss_id_map.json"),
        "I_s2e":         load_json("summary_tree/I_s2e.json"),
        "I_e2s":         load_json("summary_tree/I_e2s.json"),
        "I_e2c":         load_json("entities/I_e2c.json"),
        "I_c2e":         load_json("entities/I_c2e.json"),
        "faiss_index":   faiss.read_index(os.path.join(cache_dir, "summary_tree", "tree.index")),
        "book_id":       book_id,
        "embedder_func": embedder_func,
        "nlp":           spacy.load(spacy_model),
        "cross_encoder": CrossEncoder(cross_encoder_model),
        "chunk_embs":    chunk_embs,
    }

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    with driver.session() as s:
        s.run("RETURN 1")
    resources["neo4j_driver"] = driver

    print(f"[Load] Resources ready for book {book_id}.")
    return resources