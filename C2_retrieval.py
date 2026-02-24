"""
C2_retrieval.py

Region-restricted retrieval pipeline:
  1. Extract query entities (SpaCy)
  2. FAISS + cross-encoder -> top summary nodes (region selection)
  3. define_region -> C_region (chunks) + E_region (entities)
  4. Fuzzy seed selection
  5. Graph traversal (1-hop or 2-hop) within region
  6. Collect candidate chunks (edge evidence + entity mentions)
  7. MMR -> diverse subset
  8. Cross-encoder rerank -> top 15
  9. Lost-in-the-Middle reorder -> LLM
"""

import os
import re
import json
import logging
import numpy as np
import faiss
import spacy

try:
    from rapidfuzz import fuzz as _fuzz
    _FUZZY = True
except ImportError:
    _FUZZY = False
    print("[WARN] rapidfuzz not installed: pip install rapidfuzz --break-system-packages")

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "tree_top_m": 10,               # summary nodes kept after cross-encoder
    "use_tree_rerank": True,
    "max_hops": 1,                  # 1 or 2
    "max_candidates": 50,
    "max_chunks_per_entity": 10,
    "use_entity_mention_chunks": True,
    "mmr_top_k": 20,
    "mmr_lambda": 0.5,
    "final_top_k": 15,              # chunks given to LLM
    "min_candidates": 3,
    "fuzzy_threshold": 85,
    "fuzzy_min_length": 5,
}

NER_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "EVENT"}
NER_SKIP   = {"he", "she", "it", "they", "we", "i", "you", "this", "that"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_options(text):
    """Return question text only, dropping appended A./B./C./D. options."""
    return text.split("\nA. ")[0].strip() if "\nA. " in text else text.strip()


def extract_entities(text, nlp):
    """SpaCy NER on question text (options stripped first)."""
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
    """Match query entities to E_region using fuzzy string similarity."""
    if not _FUZZY:
        return [e for e in query_entities if e in E_region]
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
    """
    Select a diverse + relevant subset using MMR.

    relevance(c) = cosine(query_embedding, chunk_embedding)
    MMR score    = lambda * relevance(c) - (1-lambda) * max_sim(c, already_selected)

    chunk_embs: optional dict {chunk_id -> np.array} of precomputed embeddings.
                If provided, skips re-embedding candidate texts (much faster).
    """
    if len(candidates) <= top_k:
        return candidates

    # Embed query (always needed at query time, cheap single call)
    q_emb = np.array(embedder_func([query])[0]).astype("float32")
    q_emb = q_emb / max(np.linalg.norm(q_emb), 1e-9)

    # Get candidate embeddings: use precomputed if available, else embed now
    if chunk_embs is not None:
        emb_list = []
        for c in candidates:
            node_id = f"L0_{c['chunk_id']}" if not c["chunk_id"].startswith("L0_") else c["chunk_id"]
            emb = chunk_embs.get(node_id) or chunk_embs.get(c["chunk_id"])
            if emb is None:
                # Fallback: embed on the fly if missing from precomputed
                emb = np.array(embedder_func([c["text"]])[0]).astype("float32")
            emb_list.append(emb)
        embs = np.array(emb_list).astype("float32")
    else:
        embs = np.array(embedder_func([c["text"] for c in candidates])).astype("float32")

    # Normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs  = embs / np.where(norms == 0, 1e-9, norms)

    relevance = embs @ q_emb   # cosine similarity (n,)

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
    """Place most-relevant chunks at start and end; least-relevant in middle."""
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
    """FAISS finds summary node candidates; cross-encoder reranks them."""
    qv = np.array(embedder_func(dense_query)).reshape(1, -1).astype("float32")
    faiss.normalize_L2(qv)
    _, indices = faiss_index.search(qv, config["tree_top_m"] * 10)

    # Filter out raw chunk nodes (L0), keep only summary nodes
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
    """Expand summary nodes into C_region (chunks) and E_region (entities)."""
    C_region, E_region = set(), set()
    for sid in summary_ids:
        for leaf in tree_nodes.get(sid, {}).get("leaves", []):
            C_region.add(leaf[3:] if leaf.startswith("L0_") else leaf)
        E_region.update(I_s2e.get(sid, []))
    print(f"[Region] C_region={len(C_region)} chunks, E_region={len(E_region)} entities")
    return C_region, E_region


# ---------------------------------------------------------------------------
# Seed selection
# ---------------------------------------------------------------------------

def select_seeds(query_entities, E_region, config):
    """Fuzzy-match query entities to E_region. No frequency fallback."""
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
# Graph traversal
# ---------------------------------------------------------------------------

def traverse_graph(seeds, E_region, C_region, book_id, driver, max_hops=1):
    """Traverse Neo4j from seeds, constrained to E_region and C_region."""
    if max_hops == 1:
        cypher = """
        MATCH (s:Entity)-[r:RELATION]-(n:Entity)
        WHERE s.book_id = $book_id AND s.name IN $seeds AND n.name IN $e_region
        WITH s.name AS source, n.name AS target, r.relation AS relation,
             r.weight AS weight,
             [c IN r.chunk_ids WHERE c IN $c_region] AS evidence
        WHERE SIZE(evidence) > 0
        RETURN source, target, relation, weight, evidence
        """
    else:
        cypher = """
        MATCH (s:Entity)-[r1:RELATION]-(m:Entity)-[r2:RELATION]-(n:Entity)
        WHERE s.book_id = $book_id AND s.name IN $seeds
          AND m.name IN $e_region AND n.name IN $e_region
          AND m <> s AND n <> s AND n <> m
        WITH s.name AS src, m.name AS mid, n.name AS tgt,
             r1.relation AS rel1, r2.relation AS rel2, r1.weight AS w1, r2.weight AS w2,
             [c IN r1.chunk_ids WHERE c IN $c_region] AS ev1,
             [c IN r2.chunk_ids WHERE c IN $c_region] AS ev2
        WHERE SIZE(ev1) > 0 OR SIZE(ev2) > 0
        RETURN src, mid, tgt, rel1, rel2, w1, w2, ev1, ev2
        """

    params = dict(book_id=book_id, seeds=list(seeds),
                  e_region=list(E_region), c_region=list(C_region))

    with driver.session() as session:
        rows = list(session.run(cypher, **params))

    visited, records = set(seeds), []
    if max_hops == 1:
        for r in rows:
            visited.update([r["source"], r["target"]])
            records.append({"source": r["source"], "target": r["target"],
                            "relation": r["relation"], "evidence": r["evidence"]})
    else:
        for r in rows:
            visited.update([r["src"], r["mid"], r["tgt"]])
            if r["ev1"]:
                records.append({"source": r["src"],  "target": r["mid"],
                                "relation": r["rel1"], "evidence": r["ev1"]})
            if r["ev2"]:
                records.append({"source": r["mid"],  "target": r["tgt"],
                                "relation": r["rel2"], "evidence": r["ev2"]})

    print(f"[GraphTraversal] {max_hops}-hop: {len(records)} edges, {len(visited)} entities visited.")
    return records, visited


# ---------------------------------------------------------------------------
# Candidate collection
# ---------------------------------------------------------------------------

def collect_candidates(records, visited, C_region, I_e2c, I_c2e,
                       query_entities, tree_nodes, config):
    """Build deduplicated candidate pool from edge evidence + entity mentions."""
    chunk_data = {}  # chunk_id -> {is_edge_evidence, sources}

    # Source A: edge evidence chunks
    for rec in records:
        for cid in rec["evidence"]:
            if cid not in chunk_data:
                chunk_data[cid] = {"is_edge_evidence": True, "sources": {"edge"}}
            else:
                chunk_data[cid]["is_edge_evidence"] = True
                chunk_data[cid]["sources"].add("edge")
    print(f"[Candidates] Edge evidence: {len(chunk_data)} chunks.")

    # Source B: entity mention chunks (optional)
    if config.get("use_entity_mention_chunks", True):
        for entity in visited:
            for cid in [c for c in I_e2c.get(entity, []) if c in C_region][:config["max_chunks_per_entity"]]:
                if cid not in chunk_data:
                    chunk_data[cid] = {"is_edge_evidence": False, "sources": {"entity"}}
                else:
                    chunk_data[cid]["sources"].add("entity")
        print(f"[Candidates] After entity mentions: {len(chunk_data)} total (deduplicated).")

    q_ents     = set(query_entities)
    candidates = []
    for cid, data in chunk_data.items():
        node_id = f"L0_{cid}" if not cid.startswith("L0_") else cid
        text    = tree_nodes.get(node_id, {}).get("text") or tree_nodes.get(cid, {}).get("text")
        if not text:
            continue
        candidates.append({
            "chunk_id":           cid,
            "text":               text,
            "sources":            data["sources"],
            "is_edge_evidence":   data["is_edge_evidence"],
            "query_entity_count": len(q_ents & set(I_c2e.get(cid, []))),
            "score":              0.0,
        })

    candidates.sort(key=lambda x: (x["is_edge_evidence"], x["query_entity_count"]), reverse=True)
    candidates = candidates[:config["max_candidates"]]
    print(f"[Candidates] Final pool: {len(candidates)} chunks.")
    return candidates


# ---------------------------------------------------------------------------
# Cross-encoder rerank
# ---------------------------------------------------------------------------

def rerank_chunks(question_only, candidates, cross_encoder, top_k):
    """Score (question, chunk) pairs with cross-encoder; return top_k."""
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
# Tree-only fallback
# ---------------------------------------------------------------------------

def tree_only_fallback(question_only, C_region, I_c2e, query_entities,
                       tree_nodes, embedder_func, cross_encoder, config, chunk_embs=None):
    """Fallback: skip graph, rerank region chunks with MMR + cross-encoder."""
    print("[Fallback] Using tree-only fallback.")
    q_ents     = set(query_entities)
    candidates = []
    for cid in C_region:
        node_id = f"L0_{cid}" if not cid.startswith("L0_") else cid
        text    = tree_nodes.get(node_id, {}).get("text") or tree_nodes.get(cid, {}).get("text")
        if not text:
            continue
        candidates.append({
            "chunk_id": cid, "text": text, "sources": {"fallback"},
            "is_edge_evidence": False,
            "query_entity_count": len(q_ents & set(I_c2e.get(cid, []))),
            "score": 0.0,
        })
    candidates.sort(key=lambda x: x["query_entity_count"], reverse=True)
    candidates = candidates[:config["max_candidates"]]
    candidates = mmr_select(question_only, candidates, embedder_func, config["mmr_top_k"], config["mmr_lambda"], chunk_embs=chunk_embs)
    return rerank_chunks(question_only, candidates, cross_encoder, config["final_top_k"])


# ---------------------------------------------------------------------------
# Result packaging
# ---------------------------------------------------------------------------

def build_result(chunks, query_entities, seeds, mode, stats):
    """Package retrieval output. Converts sources set->list for JSON safety."""
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
    """Run the full C2 retrieval pipeline for one question."""
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

    # FAISS gets question + options; cross-encoder gets question only
    # dataloader already bakes options into question for InfiniteChoice.
    # For InfiniteQA there are no options so question is already clean.
    dense_query   = question                 # used for FAISS (broad semantic match)
    question_only = strip_options(question)  # used for cross-encoder (precise scoring)

    print(f"\n{'='*60}")
    print(f"[Retrieve] Q: {question_only[:80]}...")

    # Step 1: entity extraction (question only)
    query_entities        = extract_entities(question_only, nlp)
    stats["query_entities"] = query_entities
    print(f"[Entities] {query_entities}")

    # Step 2+3: tree pruning (FAISS + cross-encoder on summary nodes)
    top_summaries          = tree_pruning(dense_query, question_only, faiss_index,
                                          node_id_list, embedder_func, tree_nodes,
                                          cross_encoder, cfg)
    stats["top_summaries"] = top_summaries

    if not top_summaries:
        fb = tree_only_fallback(question_only, set(), I_c2e, query_entities,
                                tree_nodes, embedder_func, cross_encoder, cfg,
                                chunk_embs=resources.get("chunk_embs"))
        return build_result(lost_in_middle_reorder(fb), query_entities,
                            [], "fallback_no_summaries", stats)

    # Step 4: define region
    C_region, E_region        = define_region(top_summaries, tree_nodes, I_s2e)
    stats["region_chunks"]     = len(C_region)
    stats["region_entities"]   = len(E_region)

    # Step 5: seed selection (fuzzy matching only, no frequency fallback)
    seeds, seed_strategy       = select_seeds(query_entities, E_region, cfg)
    stats["seeds"]             = seeds
    stats["seed_strategy"]     = seed_strategy

    if not seeds:
        fb = tree_only_fallback(question_only, C_region, I_c2e, query_entities,
                                tree_nodes, embedder_func, cross_encoder, cfg,
                                chunk_embs=resources.get("chunk_embs"))
        return build_result(lost_in_middle_reorder(fb), query_entities,
                            [], f"fallback_{seed_strategy}", stats)

    # Step 6: graph traversal (1-hop or 2-hop within region)
    try:
        records, visited = traverse_graph(seeds, E_region, C_region,
                                          book_id, neo4j_driver, cfg["max_hops"])
    except Exception as e:
        logger.error(f"Graph traversal failed: {e}")
        records, visited = [], set(seeds)

    stats["relations_found"] = len(records)

    # Step 7: collect candidates (deduplicated by chunk_data dict)
    candidates                     = collect_candidates(records, visited, C_region,
                                                        I_e2c, I_c2e, query_entities,
                                                        tree_nodes, cfg)
    stats["candidates_before_mmr"] = len(candidates)

    if len(candidates) < cfg["min_candidates"]:
        fb = tree_only_fallback(question_only, C_region, I_c2e, query_entities,
                                tree_nodes, embedder_func, cross_encoder, cfg,
                                chunk_embs=resources.get("chunk_embs"))
        return build_result(lost_in_middle_reorder(fb), query_entities,
                            seeds, "fallback_no_candidates", stats)

    # Step 8: MMR -> diverse subset (e.g. 50 -> 20)
    candidates                    = mmr_select(question_only, candidates, embedder_func,
                                               cfg["mmr_top_k"], cfg["mmr_lambda"],
                                               chunk_embs=resources.get("chunk_embs"))
    stats["candidates_after_mmr"] = len(candidates)

    # Step 9: cross-encoder rerank (question only, no options) -> top 15
    top_chunks              = rerank_chunks(question_only, candidates,
                                            cross_encoder, cfg["final_top_k"])
    stats["final_chunks"]   = len(top_chunks)

    # Step 10: Lost-in-the-Middle reorder before passing to LLM
    top_chunks = lost_in_middle_reorder(top_chunks)

    print(f"[Retrieve] Done. mode=graph_traversal, chunks={len(top_chunks)}")
    print(f"{'='*60}\n")
    return build_result(top_chunks, query_entities, seeds, "graph_traversal", stats)


# ---------------------------------------------------------------------------
# Resource loader
# ---------------------------------------------------------------------------

def load_retriever(cache_dir, book_id, neo4j_uri, neo4j_user, neo4j_password,
                   embedder_func, spacy_model="en_core_web_lg",
                   cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Load all artifacts for one book from cache."""
    from neo4j import GraphDatabase
    from sentence_transformers import CrossEncoder

    def load_json(path):
        with open(os.path.join(cache_dir, path), "r", encoding="utf-8") as f:
            return json.load(f)

    # Load precomputed embeddings for all nodes (L0 chunks + summary nodes)
    # These were saved during indexing in save_tree() -> embeddings.npz
    # Used by MMR to avoid re-embedding chunks at query time
    raw_embs  = np.load(os.path.join(cache_dir, "summary_tree", "embeddings.npz"))
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
        "chunk_embs":    chunk_embs,   # precomputed embeddings, keyed by node_id
    }
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    with driver.session() as s:
        s.run("RETURN 1")
    resources["neo4j_driver"] = driver

    print(f"[Load] Resources ready for book {book_id}.")
    return resources