"""
This script handles the heavy lifting of finding relationships between entities. 
We pass the text chunks and our extracted entities to an LLM, ask it to find the connections, 
and then we merge all those connections to build the edges of our graph. 
Since hitting the LLM takes time and money, everything is heavily cached!
"""

import os
import re
import json

from llm import call_llm


PROMPT = """---Goal---
You are a knowledge-graph relation extractor. Given a text chunk and a list of entities, extract ALL meaningful relationships among the entities.

---Instructions---
1. Consider ONLY the entities provided below. Do NOT invent new entities.
2. For each pair of entities that are related in this text, extract:
   - source: entity name (exactly as provided)
   - target: entity name (exactly as provided)
   - relation: a short verb phrase (1-5 words), e.g., "loves", "works for", "distrusts", "travels to"
3. Do NOT treat interjections/utterances (e.g., 'hoorray') as entities
4. RELATION QUALITY RULES (STRICT):
   - DO NOT output weak communication/reporting verbs as relations:
     said, says, tell/told, ask/asked, speak/spoke, talk/talked, mention/mentioned, discuss/discussed, reply/replied.
   - Exception: if the utterance implies a stronger relation, map it to one of these:
     warns, orders, accuses, threatens, begs, convinces, praises, insults, rejects, agrees_with, argues_with.
   - Prefer stable, reusable relations (choose the best fit):
     family_of, friend_of, enemy_of, helps, threatens, protects, works_for, owns, lives_in, travels_to, loves, hates, trusts, distrusts, supports, opposes.



---What Counts as a Relationship---
Include relationships that are:
- Explicitly stated ("X married Y")
- Implied by actions (e.g. "X helped Y escape" becomes "X helps Y")
- Implied by dialogue (e.g. "X shouted at Y" becomes "X is angry with Y")
- Social, emotional, or hierarchical (friend, enemy, boss, servant)
- Cooperative or antagonistic

Do NOT extract:
- Trivial co-mentions ("X and Y were in the room")
- Redundant relations (if you extract "X loves Y", don't also extract "Y is loved by X")
- Relations not supported by this specific text

---Output Format---
Return ONLY a JSON array. No explanation, no commentary, no markdown.

[
  {{
    "source": "...",
    "target": "...",
    "relation": "..."
  }},
  ...
]

---Example---
ENTITIES: ["Alice", "Bob", "The Castle"]

TEXT: Alice glared at Bob across the courtyard. She had trusted him once, but his betrayal at the castle still burned. "You sold us out," she whispered. Bob looked away, unable to meet her eyes.

OUTPUT:
[
  {{
    "source": "Alice",
    "target": "Bob",
    "relation": "distrusts"
  }},
  {{
    "source": "Bob",
    "target": "Alice",
    "relation": "betrayed"
  }},
  {{
    "source": "Bob",
    "target": "The Castle",
    "relation": "betrayed allies at"
  }}
]

---Now Extract---
ENTITIES:
{entity_list}

TEXT:
{chunk_text}

Return ONLY the JSON array:
"""


TITLE_PREFIXES = {"mr", "mrs", "ms", "miss", "dr", "sir", "lady", "lord"}


def canonicalize(s):
    """Cleans up the entity name and strips out common titles like 'Mr.' or 'Dr.' so 'Mr. Darcy' and 'Darcy' match."""
    if isinstance(s, list):
        s = s[0] if s else ""
    if not isinstance(s, str):
        s = str(s) if s else ""
    s = re.sub(r"\s+", " ", (s or "").lower().strip())
    s = re.sub(r"\.", "", s)
    parts = s.split()
    if len(parts) >= 2 and parts[0] in TITLE_PREFIXES:
        s = " ".join(parts[1:])
    return s


def canonicalize_relation(s):
    """A quick little cleanup function to make sure the relation phrases are formatted consistently."""
    if isinstance(s, list):
        s = s[0] if s else ""
    if not isinstance(s, str):
        s = str(s) if s else ""
    return re.sub(r"\s+", " ", (s or "").lower().strip())


def parse_json_array(text):
    """Sometimes the LLM wraps its JSON in markdown blocks or adds extra text. This function rips out just the JSON array we need."""
    text = (text or "").strip()
    if not text:
        return []

    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        print("[WARN] The LLM didn't return a valid JSON array like we asked.")
        return None

    try:
        result = json.loads(match.group())
        return result if isinstance(result, list) else []
    except json.JSONDecodeError as e:
        print(f"[WARN] JSON parse failed: {e}")
        return None


def extract_relations_from_chunk(chunk_text, entities, model="gpt", verbose=True):
    """
    Sends a single chunk of text and its entities to the LLM to see how they are connected.
    If something goes completely wrong, it just returns None so we can skip it.
    """
    if len(entities) < 2:
        return []

    entity_list = "\n".join(f"- {e}" for e in entities)
    prompt = PROMPT.format(entity_list=entity_list, chunk_text=chunk_text[:8000])

    try:
        response = call_llm(prompt, model=model, max_tokens=2048)
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        return None

    raw_relations = parse_json_array(response)
    if raw_relations is None:
        print("[DEBUG] We couldn't parse the LLM output. Saving exactly what it said to debug_llm_failure.txt so we can investigate later.")
        with open("debug_llm_failure.txt", "w", encoding="utf-8") as f:
            f.write(response)
        return None

    if verbose:
        print(f"[DEBUG] {len(entities)} entities | {len(raw_relations)} relations parsed")

    entity_norm_set = {canonicalize(e) for e in entities}
    validated = []
    seen = set()

    for rel in raw_relations:
        if not isinstance(rel, dict):
            continue

        source = rel.get("source", "")
        target = rel.get("target", "")
        relation = rel.get("relation", "")
        if not (source and target and relation):
            continue

        s_norm = canonicalize(source)
        t_norm = canonicalize(target)
        r_norm = canonicalize_relation(relation)

        if s_norm not in entity_norm_set or t_norm not in entity_norm_set:
            continue

        if s_norm == t_norm:
            continue

        key = (s_norm, r_norm, t_norm)
        if key in seen:
            continue
        seen.add(key)

        validated.append({"source": s_norm, "relation": r_norm, "target": t_norm})

    return validated


def get_cache_path(cache_dir, chunk_id):
    """Calculates exactly where to save or load the cache file for a specific chunk."""
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", chunk_id)
    return os.path.join(cache_dir, "relations", f"{safe_id}.json")


def extract_relations_batch(chunks, I_c2e, llm_choice="gpt", cache_dir="./cache"):
    """
    Loops through every single text chunk in our dataset and checks for relationships.
    It's smart about caching, so if we've already processed a chunk in the past, it just loads it from the disk instead of calling the LLM again.
    """
    relations_dir = os.path.join(cache_dir, "relations")
    os.makedirs(relations_dir, exist_ok=True)

    all_relations = []
    to_process = []
    skipped = 0
    cached = 0

    for chunk in chunks:
        chunk_id = f"L0_{chunk['chunk_id']}"
        entities = I_c2e.get(chunk_id, [])

        # We need at least two entities to form a relationship!
        if len(entities) < 2:
            skipped += 1
            continue

        cache_path = get_cache_path(cache_dir, chunk_id)

        if os.path.exists(cache_path):
            cached += 1
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_obj = json.load(f)
            for rel in cached_obj.get("relations", []):
                rel_copy = rel.copy()
                rel_copy["chunk_id"] = chunk_id
                all_relations.append(rel_copy)
        else:
            to_process.append(chunk)

    print(f"Relation extraction: {len(to_process)} to process, cached={cached}, skipped(<2 ents)={skipped}")

    for i, chunk in enumerate(to_process):
        chunk_id = f"L0_{chunk['chunk_id']}"
        text = chunk["text"]
        entities = I_c2e.get(chunk_id, [])

        print(f"[{i+1}/{len(to_process)}] {chunk_id}...", end=" ", flush=True)

        relations = extract_relations_from_chunk(text, entities, model=llm_choice, verbose=False)

        if relations is not None:
            cache_path = get_cache_path(cache_dir, chunk_id)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"chunk_id": chunk_id, "relations": relations}, f, indent=2, ensure_ascii=False)

            for rel in relations:
                rel_copy = rel.copy()
                rel_copy["chunk_id"] = chunk_id
                all_relations.append(rel_copy)

            print(f"{len(relations)} relations")
        else:
            print("FAILED (skipping cache save)")

    print(f"Total relations extracted: {len(all_relations)}")
    return all_relations


def merge_relations(relations):
    """
    If multiple chunks mention that 'Alice loves Bob', we don't want a hundred separate edges. 
    Instead, we merge them into a single edge and bump up its 'weight' to show how well-supported it is.
    """
    edge_map = {}

    for rel in relations:
        key = (rel["source"], rel["relation"], rel["target"])

        if key not in edge_map:
            edge_map[key] = {
                "source": rel["source"],
                "relation": rel["relation"],
                "target": rel["target"],
                "weight": 0,
                "chunk_ids": [],
            }

        edge = edge_map[key]
        edge["weight"] += 1

        chunk_id = rel.get("chunk_id", "")
        if chunk_id and chunk_id not in edge["chunk_ids"]:
            edge["chunk_ids"].append(chunk_id)

    edges = sorted(edge_map.values(), key=lambda x: x["weight"], reverse=True)
    return edges


def save_edges(edges, cache_dir):
    """Dumps all our consolidated graph edges into a final JSON file."""
    filepath = os.path.join(cache_dir, "edges.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(edges, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(edges)} edges to {filepath}")


def load_edges(cache_dir):
    """Pulls the previously saved edges back into memory."""
    filepath = os.path.join(cache_dir, "edges.json")
    with open(filepath, "r", encoding="utf-8") as f:
        edges = json.load(f)
    print(f"Loaded {len(edges)} edges from {filepath}")
    return edges


def extract_and_merge_relations(chunks, I_c2e, llm_choice="gpt", cache_dir="./cache"):
    """The master function that runs the whole relation extraction and merging pipeline from start to finish."""
    relations = extract_relations_batch(chunks, I_c2e, llm_choice, cache_dir)
    edges = merge_relations(relations)
    save_edges(edges, cache_dir)
    return edges