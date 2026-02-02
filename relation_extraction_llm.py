"""
relation_extraction.py

Simplified LLM-based relation extraction.
- Minimal prompt, minimal output format
- Exact entity matching only (case-insensitive canonicalized match)
- Per-chunk caching
- Simple merge at the end
"""

import os
import re
import json
from typing import List, Dict

from llm import call_llm


# PROMPT
PROMPT = """---Goal---
You are a knowledge-graph relation extractor. Given a text chunk and a list of entities, extract ALL meaningful relationships among the entities.

---Instructions---
1. Consider ONLY the entities provided below. Do NOT invent new entities.
2. For each pair of entities that are related in this text, extract:
   - source: entity name (exactly as provided)
   - target: entity name (exactly as provided)
   - relation: a short verb phrase (1-5 words), e.g., "loves", "works for", "distrusts", "travels to"
   - evidence: 1-2 sentences copied from the text that support this relationship

---What Counts as a Relationship---
Include relationships that are:
- Explicitly stated ("X married Y")
- Implied by actions ("X helped Y escape" → X helps Y)
- Implied by dialogue ("X shouted at Y" → X is angry with Y)
- Social, emotional, or hierarchical (friend, enemy, boss, servant)
- Cooperative or antagonistic

Do NOT extract:
- Trivial co-mentions ("X and Y were in the room")
- Redundant relations (if you extract "X loves Y", don't also extract "Y is loved by X")
- Relations not supported by this specific text

---Output Format---
Return a JSON array. If no relationships exist, return [].

[
  {"source": "...", "target": "...", "relation": "...", "evidence": "..."},
  ...
]

---Example---
ENTITIES: ["Alice", "Bob", "The Castle"]

TEXT: Alice glared at Bob across the courtyard. She had trusted him once, but his betrayal at the castle still burned. "You sold us out," she whispered. Bob looked away, unable to meet her eyes.

OUTPUT:
[
  {"source": "Alice", "target": "Bob", "relation": "distrusts", "evidence": "She had trusted him once, but his betrayal at the castle still burned."},
  {"source": "Bob", "target": "Alice", "relation": "betrayed", "evidence": "his betrayal at the castle still burned. \\"You sold us out,\\" she whispered."},
  {"source": "Bob", "target": "The Castle", "relation": "betrayed allies at", "evidence": "his betrayal at the castle still burned."}
]

---Now Extract---
ENTITIES:
{entity_list}

TEXT:
{chunk_text}

OUTPUT:
"""


# ============== CANONICALIZATION ==============

TITLE_PREFIXES = {"mr", "mrs", "ms", "miss", "dr", "sir", "lady", "lord"}


def canonicalize(s: str) -> str:
    """Normalize entity name (removes title prefixes like Mr., Mrs., etc.)."""
    s = re.sub(r'\s+', ' ', (s or '').lower().strip())
    s = re.sub(r'\.', '', s)  # remove dots: "mr." -> "mr"
    parts = s.split()
    if len(parts) >= 2 and parts[0] in TITLE_PREFIXES:
        s = " ".join(parts[1:])  # drop title word
    return s


def canonicalize_relation(s: str) -> str:
    """Normalize relation phrase (no title removal, just lowercase and whitespace)."""
    return re.sub(r'\s+', ' ', (s or '').lower().strip())


# ============== JSON PARSING ==============

def parse_json_array(text: str) -> List[Dict]:
    """Extract JSON array from LLM response."""
    text = (text or '').strip()
    if not text:
        return []

    # Strip markdown code blocks if present
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    # Find JSON array (greedy - safer for evidence containing brackets)
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        print(f"[WARN] No JSON array found in LLM response")
        return []

    try:
        result = json.loads(match.group())
        return result if isinstance(result, list) else []
    except json.JSONDecodeError as e:
        print(f"[WARN] JSON parse failed: {e}")
        return []


# ============== SINGLE CHUNK EXTRACTION ==============

def extract_relations_from_chunk(
    chunk_text: str,
    entities: List[str],
    llm_choice: str = "gpt",
    verbose: bool = True
) -> List[Dict]:
    """
    Extract relations from a single chunk.

    Validation:
    - source/target must match provided entity list (case-insensitive via canonicalize)
    - no self loops
    - dedup within chunk using canonicalized keys
    """
    if len(entities) < 2:
        return []

    # Build prompt
    entity_list = "\n".join(f"- {e}" for e in entities)
    prompt = PROMPT.format(entity_list=entity_list, chunk_text=chunk_text[:8000])

    # Call LLM with error handling
    try:
        response = call_llm(prompt, llm_choice=llm_choice, max_tokens=1024)
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        return []

    raw_relations = parse_json_array(response)

    if verbose:
        print(f"[DEBUG] {len(entities)} entities | {len(raw_relations)} relations parsed")

    # Canonicalized entity set for matching
    entity_norm_set = {canonicalize(e) for e in entities}

    validated = []
    seen = set()

    for rel in raw_relations:
        if not isinstance(rel, dict):
            continue

        source = rel.get("source", "")
        target = rel.get("target", "")
        relation = rel.get("relation", "")
        evidence = rel.get("evidence", "")

        if not (source and target and relation):
            continue

        s_norm = canonicalize(source)
        t_norm = canonicalize(target)
        r_norm = canonicalize_relation(relation)  # use relation-specific normalizer

        # Must match entity list (canonicalized)
        if s_norm not in entity_norm_set or t_norm not in entity_norm_set:
            continue

        # No self-loops (after canonicalization)
        if s_norm == t_norm:
            continue

        # Deduplicate within chunk (canonicalized)
        key = (s_norm, r_norm, t_norm)
        if key in seen:
            continue
        seen.add(key)

        validated.append({
            "source": s_norm,
            "relation": r_norm,
            "target": t_norm,
            "evidence": (evidence or "")[:500]
        })

    return validated


# ============== BATCH PROCESSING WITH CACHE ==============

def get_cache_path(cache_dir: str, chunk_id: str) -> str:
    """Get cache file path for a chunk."""
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', chunk_id)
    return os.path.join(cache_dir, "relations", f"{safe_id}.json")


def extract_relations_batch(
    chunks: List[Dict],
    I_c2e: Dict[str, List[str]],
    llm_choice: str = "gpt",
    cache_dir: str = "./cache"
) -> List[Dict]:
    """
    Extract relations from all chunks with caching.

    Returns:
        List of all relation dicts with chunk_id added
    """
    relations_dir = os.path.join(cache_dir, "relations")
    os.makedirs(relations_dir, exist_ok=True)

    all_relations = []
    to_process = []

    skipped = 0
    cached = 0

    # Check cache
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        entities = I_c2e.get(chunk_id, [])

        if len(entities) < 2:
            skipped += 1
            continue

        cache_path = get_cache_path(cache_dir, chunk_id)

        if os.path.exists(cache_path):
            cached += 1
            # Load from cache (use copy to avoid mutation)
            with open(cache_path, 'r', encoding="utf-8") as f:
                cached_obj = json.load(f)
            for rel in cached_obj.get("relations", []):
                rel_copy = rel.copy()
                rel_copy["chunk_id"] = chunk_id
                all_relations.append(rel_copy)
        else:
            to_process.append(chunk)

    print(f"Relation extraction: {len(to_process)} to process, cached={cached}, skipped(<2 ents)={skipped}")

    # Process remaining chunks
    for i, chunk in enumerate(to_process):
        chunk_id = chunk["chunk_id"]
        text = chunk["text"]
        entities = I_c2e.get(chunk_id, [])

        print(f"[{i+1}/{len(to_process)}] {chunk_id}...", end=" ", flush=True)

        relations = extract_relations_from_chunk(text, entities, llm_choice, verbose=False)

        # Save to cache
        cache_path = get_cache_path(cache_dir, chunk_id)
        with open(cache_path, 'w', encoding="utf-8") as f:
            json.dump({"chunk_id": chunk_id, "relations": relations}, f, indent=2, ensure_ascii=False)

        # Add to results
        for rel in relations:
            rel_copy = rel.copy()
            rel_copy["chunk_id"] = chunk_id
            all_relations.append(rel_copy)

        print(f"{len(relations)} relations")

    print(f"Total relations extracted: {len(all_relations)}")
    return all_relations


# ============== MERGE INTO WEIGHTED EDGES ==============

def merge_relations(relations: List[Dict]) -> List[Dict]:
    """
    Merge duplicate relations across chunks into weighted edges.
    Keeps up to 3 UNIQUE evidences.
    """
    edge_map = {}  # (source, relation, target) -> edge dict

    for rel in relations:
        key = (rel["source"], rel["relation"], rel["target"])

        if key not in edge_map:
            edge_map[key] = {
                "source": rel["source"],
                "relation": rel["relation"],
                "target": rel["target"],
                "weight": 0,
                "chunk_ids": [],
                "evidences": []
            }

        edge = edge_map[key]
        edge["weight"] += 1

        chunk_id = rel.get("chunk_id", "")
        if chunk_id and chunk_id not in edge["chunk_ids"]:
            edge["chunk_ids"].append(chunk_id)

        evidence = rel.get("evidence", "")
        if evidence and evidence not in edge["evidences"] and len(edge["evidences"]) < 3:
            edge["evidences"].append(evidence)

    edges = sorted(edge_map.values(), key=lambda x: x["weight"], reverse=True)
    return edges


# ============== SAVE / LOAD ==============

def save_edges(edges: List[Dict], cache_dir: str):
    """Save merged edges to JSON."""
    filepath = os.path.join(cache_dir, "edges.json")
    with open(filepath, 'w', encoding="utf-8") as f:
        json.dump(edges, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(edges)} edges to {filepath}")


def load_edges(cache_dir: str) -> List[Dict]:
    """Load merged edges from JSON."""
    filepath = os.path.join(cache_dir, "edges.json")
    with open(filepath, 'r', encoding="utf-8") as f:
        edges = json.load(f)
    print(f"Loaded {len(edges)} edges from {filepath}")
    return edges


# ============== MAIN ENTRY POINT ==============

def extract_and_merge_relations(
    chunks: List[Dict],
    I_c2e: Dict[str, List[str]],
    llm_choice: str = "gpt",
    cache_dir: str = "./cache"
) -> List[Dict]:
    """Full pipeline: extract relations from chunks and merge into edges."""
    relations = extract_relations_batch(chunks, I_c2e, llm_choice, cache_dir)
    edges = merge_relations(relations)
    save_edges(edges, cache_dir)
    return edges