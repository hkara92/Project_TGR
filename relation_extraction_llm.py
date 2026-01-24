"""
relation_extraction_llm.py

LLM-based relation extraction for Case 1.
Extracts typed relations between entities using GPT-4o-mini.

Guardrails:
----------
1. Clearly related entities only (prompt + validation)
2. Chunk locality (only use entities from I_c2e[chunk_id])
3. Exact string matching (canonicalize during merge)
4. Edge aggregation (merge duplicates, count weight)
5. Evidence validation (must be substring of chunk)
"""

import os
import json
import re
import time
from typing import List, Dict, Tuple, Optional

from llm import call_llm


# ============ CONFIG ============

PROMPT_VERSION = "v3"
MAX_EVIDENCES_PER_EDGE = 3
MAX_RETRIES = 2
MAX_CHUNK_CHARS = 3000
MIN_EVIDENCE_LENGTH = 10
DEFAULT_MAX_RELATIONS = 15
MIN_STRENGTH_THRESHOLD = 3  # Reject relations with strength < 3


# ============ BLACKLISTS ============

GENERIC_RELATIONS = {
    "related to", "associated with", "connected to",
    "involves", "about", "mentions", "describes",
    "talks about", "refers to", "in context of",
    "is related with", "is connected with",
    "pertains to", "concerns", "regarding",
    "in relation to", "linked to", "tied to",
    "mentioned with", "appears with"
}

DISCOURSE_WORDS = {"mention", "describe", "talk", "refer", "discuss", "state"}

DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "my", "his", "her", "their", "our"}


# ============ PROMPT ============

EXTRACTION_PROMPT = """You are a relationship extractor for building a knowledge graph from novel text.

TASK
Given:
1) a TEXT CHUNK
2) a list of ENTITIES that appear in this chunk (canonical strings)

Extract ONLY relationships between entities that are CLEARLY and EXPLICITLY related in the text.
Do NOT connect entities just because they are mentioned in the same chunk.

TEXT CHUNK:
{chunk_text}

ENTITIES (use ONLY these; do not invent new entities):
{entity_list}

INSTRUCTIONS
1) Consider ONLY pairs (source entity, target entity) from the entity list.
2) Output a relationship only if the text explicitly supports a meaningful connection
   (e.g., speech to someone, family relation, employment, role/title, location, ownership,
   direct action by one on another, meeting, threat, etc.).
3) Avoid weak/vague relations:
   - do not output "related to", "associated with", "connected to", "mentioned with", "about".
4) Each relationship MUST include:
   - source: entity string exactly as in the entity list
   - target: entity string exactly as in the entity list
   - relation: short verb phrase (2–6 words), lowercase, NO articles
              (e.g., "said to", "works for", "married to", "lives in")
   - relationship_description: 1 short sentence explaining why this relation holds
   - strength: integer 1–10 (10 = very explicit and important)
   - evidence: an exact quote from the chunk that supports the relationship
5) Keep the graph sparse: return at most {max_relations} relationships.
   Prefer the strongest / clearest relationships.

OUTPUT FORMAT (JSON ONLY)
Return a single JSON array. No markdown. No extra text.
[
  {{
    "source": "...",
    "target": "...",
    "relation": "...",
    "relationship_description": "...",
    "strength": 8,
    "evidence": "..."
  }}
]

If no clear relationships exist, return: []"""


# ============ UTILITIES ============

def canonicalize(text: str) -> str:
    """Normalize text: lowercase, strip, single spaces."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def strip_determiners(text: str) -> str:
    """Remove leading determiners."""
    text = canonicalize(text)
    words = text.split()
    if words and words[0] in DETERMINERS:
        return " ".join(words[1:])
    return text


def strip_possessive(text: str) -> str:
    """Remove trailing 's or '."""
    text = re.sub(r"'s$", "", text)
    text = re.sub(r"'$", "", text)
    return text


def normalize_entity(text: str) -> str:
    """Full normalization: canonicalize + strip determiners + strip possessive."""
    text = canonicalize(text)
    text = strip_determiners(text)
    text = strip_possessive(text)
    return text.strip()


def match_entity_to_allowed(entity_text: str, allowed_entities: List[str]) -> Optional[str]:
    """
    Match LLM output entity to allowed entities list.
    Returns the matched allowed entity or None.
    """
    normalized = normalize_entity(entity_text)
    if not normalized:
        return None
    
    # Build normalized versions of allowed entities
    allowed_normalized = {normalize_entity(e): e for e in allowed_entities}
    
    # 1. Exact match
    if normalized in allowed_normalized:
        return allowed_normalized[normalized]
    
    # 2. Substring match (if unambiguous)
    substring_matches = []
    for norm_allowed, original in allowed_normalized.items():
        if normalized in norm_allowed or norm_allowed in normalized:
            substring_matches.append(original)
    
    if len(substring_matches) == 1:
        return substring_matches[0]
    
    return None


def filter_entities_in_text(entities: List[str], text: str) -> List[str]:
    """Filter entities to only those that appear in the text."""
    text_lower = text.lower()
    return [e for e in entities if e.lower() in text_lower]


# ============ VALIDATION ============

def is_generic_relation(rel: str) -> bool:
    """Check if relation is too generic/vague."""
    rel_lower = rel.lower().strip()
    
    if rel_lower in GENERIC_RELATIONS:
        return True
    
    for word in DISCOURSE_WORDS:
        if word in rel_lower:
            return True
    
    if len(rel_lower) < 3:
        return True
    
    return False


def validate_evidence(evidence: str, chunk_text: str) -> bool:
    """Validate that evidence exists in chunk text (anti-hallucination)."""
    if not evidence:
        return False
    
    if len(evidence) < MIN_EVIDENCE_LENGTH:
        return False
    
    evidence_clean = canonicalize(evidence)
    chunk_clean = canonicalize(chunk_text)
    
    # Direct substring check
    if evidence_clean in chunk_clean:
        return True
    
    # Try with first 50 chars (LLM might paraphrase end)
    if len(evidence_clean) > 50:
        if evidence_clean[:50] in chunk_clean:
            return True
    
    return False


def validate_relation(
    rel: Dict,
    allowed_entities: List[str],
    chunk_text: str
) -> Tuple[bool, Dict]:
    """
    Validate a single relation.
    Returns (is_valid, normalized_relation).
    """
    # Must have required fields
    required_fields = ["source", "target", "relation", "evidence"]
    if not all(k in rel for k in required_fields):
        return False, {}
    
    # Match source to allowed entities
    source_matched = match_entity_to_allowed(rel["source"], allowed_entities)
    if not source_matched:
        return False, {}
    
    # Match target to allowed entities
    target_matched = match_entity_to_allowed(rel["target"], allowed_entities)
    if not target_matched:
        return False, {}
    
    # No self-loops
    if canonicalize(source_matched) == canonicalize(target_matched):
        return False, {}
    
    # Validate relation phrase
    relation = canonicalize(rel.get("relation", ""))
    if not relation:
        return False, {}
    
    if is_generic_relation(relation):
        return False, {}
    
    # Validate strength (if provided)
    strength = rel.get("strength", 5)
    try:
        strength = int(strength)
    except (ValueError, TypeError):
        strength = 5
    
    if strength < MIN_STRENGTH_THRESHOLD:
        return False, {}
    
    # Validate evidence (key hallucination guardrail)
    evidence = rel.get("evidence", "")
    if not validate_evidence(evidence, chunk_text):
        return False, {}
    
    # All checks passed
    return True, {
        "source": canonicalize(source_matched),
        "relation": relation,
        "target": canonicalize(target_matched),
        "description": rel.get("relationship_description", "")[:200],
        "strength": strength,
        "evidence": evidence[:200]
    }


def validate_and_filter(
    relations: List[Dict],
    allowed_entities: List[str],
    chunk_text: str
) -> List[Dict]:
    """Filter relations to only valid ones."""
    valid = []
    
    for rel in relations:
        is_valid, normalized = validate_relation(rel, allowed_entities, chunk_text)
        if is_valid:
            valid.append(normalized)
    
    return valid


def dedup_within_chunk(relations: List[Dict]) -> List[Dict]:
    """Remove duplicate (source, relation, target) within same chunk."""
    seen = set()
    unique = []
    
    for rel in relations:
        key = (rel["source"], rel["relation"], rel["target"])
        if key not in seen:
            seen.add(key)
            unique.append(rel)
    
    return unique


# ============ PROMPT BUILDING ============

def build_extraction_prompt(
    chunk_text: str,
    entities: List[str],
    max_relations: int = DEFAULT_MAX_RELATIONS
) -> Tuple[str, str, List[str]]:
    """
    Build the prompt for relation extraction.
    Returns (prompt, truncated_text, filtered_entities).
    """
    # Truncate chunk if too long
    if len(chunk_text) > MAX_CHUNK_CHARS:
        chunk_text = chunk_text[:MAX_CHUNK_CHARS]
    
    # Filter entities to only those in (possibly truncated) text
    filtered_entities = filter_entities_in_text(entities, chunk_text)
    
    # Format entity list
    entity_list = "\n".join(f"- {e}" for e in filtered_entities)
    
    prompt = EXTRACTION_PROMPT.format(
        chunk_text=chunk_text,
        entity_list=entity_list,
        max_relations=max_relations
    )
    
    return prompt, chunk_text, filtered_entities


# ============ RESPONSE PARSING ============

def parse_llm_response(response_text: str) -> List[Dict]:
    """Parse LLM response into list of relations."""
    response_text = response_text.strip()
    
    if not response_text or response_text == "[]":
        return []
    
    # Find JSON array in response
    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group()
    
    try:
        relations = json.loads(response_text)
        if isinstance(relations, list):
            return relations
        return []
    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON")
        return []


# ============ PER-CHUNK EXTRACTION ============

def get_chunk_cache_path(chunk_id: str, cache_dir: str) -> str:
    """Get path for per-chunk relation cache."""
    relations_dir = os.path.join(cache_dir, "relations_llm")
    os.makedirs(relations_dir, exist_ok=True)
    return os.path.join(relations_dir, f"{chunk_id}.json")


def chunk_already_processed(chunk_id: str, cache_dir: str) -> bool:
    """Check if chunk was already processed."""
    return os.path.exists(get_chunk_cache_path(chunk_id, cache_dir))


def save_chunk_relations(chunk_id: str, relations: List[Dict], cache_dir: str, model: str):
    """Save relations for a single chunk."""
    cache_path = get_chunk_cache_path(chunk_id, cache_dir)
    
    instance = {
        "chunk_id": chunk_id,
        "relations": relations,
        "model": model,
        "prompt_version": PROMPT_VERSION
    }
    
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(instance, f, ensure_ascii=False, indent=2)


def extract_relations_for_chunk(
    chunk_text: str,
    allowed_entities: List[str],
    llm_choice: str,
    max_relations: int = DEFAULT_MAX_RELATIONS
) -> List[Dict]:
    """Extract relations from a single chunk using LLM."""
    # Build prompt
    prompt, truncated_text, filtered_entities = build_extraction_prompt(
        chunk_text, allowed_entities, max_relations
    )
    
    # Need at least 2 entities
    if len(filtered_entities) < 2:
        return []
    
    # Call LLM with retries
    for attempt in range(MAX_RETRIES):
        try:
            response = call_llm(prompt, llm_choice=llm_choice, max_tokens=1024)
            relations = parse_llm_response(response)
            relations = validate_and_filter(relations, filtered_entities, truncated_text)
            relations = dedup_within_chunk(relations)
            return relations
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
    
    return []


# ============ MAIN EXTRACTION ============

def extract_relations_llm(
    chunks: List[Dict],
    I_c2e: Dict[str, List[str]],
    llm_choice: str = "gpt",
    cache_dir: str = "./cache",
    max_relations: int = DEFAULT_MAX_RELATIONS
) -> None:
    """
    Main entry point: Extract relations from all chunks using LLM.
    Saves per-chunk results to cache_dir/relations_llm/
    """
    print(f"\n{'='*50}")
    print("LLM RELATION EXTRACTION")
    print(f"{'='*50}")
    print(f"Chunks: {len(chunks)} | LLM: {llm_choice} | Max relations/chunk: {max_relations}")
    
    # Count chunks to process
    chunks_to_process = []
    chunks_skipped = 0
    chunks_cached = 0
    
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        entities = I_c2e.get(chunk_id, [])
        
        if len(entities) < 2:
            chunks_skipped += 1
            continue
        
        if chunk_already_processed(chunk_id, cache_dir):
            chunks_cached += 1
            continue
        
        chunks_to_process.append(chunk)
    
    print(f"Skipped (<2 entities): {chunks_skipped}")
    print(f"Already cached: {chunks_cached}")
    print(f"To process: {len(chunks_to_process)}")
    
    if not chunks_to_process:
        print("Nothing to process!")
        return
    
    # Process chunks
    total_relations = 0
    
    for i, chunk in enumerate(chunks_to_process):
        chunk_id = chunk["chunk_id"]
        chunk_text = chunk["text"]
        entities = I_c2e.get(chunk_id, [])
        
        print(f"[{i+1}/{len(chunks_to_process)}] {chunk_id}...", end=" ")
        
        relations = extract_relations_for_chunk(chunk_text, entities, llm_choice, max_relations)
        save_chunk_relations(chunk_id, relations, cache_dir, llm_choice)
        
        total_relations += len(relations)
        print(f"{len(relations)} relations")
    
    print(f"\nExtraction complete: {total_relations} total relations")


# ============ MERGE STEP ============

def load_all_chunk_relations(cache_dir: str) -> List[Dict]:
    """Load all per-chunk relation files."""
    relations_dir = os.path.join(cache_dir, "relations_llm")
    
    if not os.path.exists(relations_dir):
        return []
    
    all_instances = []
    
    for filename in sorted(os.listdir(relations_dir)):
        if filename.endswith(".json") and filename.startswith("chunk_"):
            filepath = os.path.join(relations_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                all_instances.append(json.load(f))
    
    return all_instances


def merge_relation_instances(cache_dir: str) -> List[Dict]:
    """
    Merge all per-chunk relations into final edges.
    Aggregates by (source, relation, target).
    """
    print(f"\n{'='*50}")
    print("MERGING RELATIONS")
    print(f"{'='*50}")
    
    all_instances = load_all_chunk_relations(cache_dir)
    
    if not all_instances:
        print("No instances to merge!")
        return []
    
    print(f"Loaded {len(all_instances)} chunk files")
    
    # Merge edges
    edge_map: Dict[Tuple[str, str, str], Dict] = {}
    
    for instance in all_instances:
        chunk_id = instance["chunk_id"]
        
        for rel in instance["relations"]:
            source = canonicalize(rel["source"])
            target = canonicalize(rel["target"])
            relation = canonicalize(rel["relation"])
            evidence = rel.get("evidence", "")
            strength = rel.get("strength", 5)
            description = rel.get("description", "")
            
            key = (source, relation, target)
            
            if key not in edge_map:
                edge_map[key] = {
                    "source": source,
                    "relation": relation,
                    "target": target,
                    "weight": 0,
                    "avg_strength": 0,
                    "chunk_ids": [],
                    "evidences": [],
                    "descriptions": []
                }
            
            edge_map[key]["weight"] += 1
            
            # Running average of strength
            n = edge_map[key]["weight"]
            old_avg = edge_map[key]["avg_strength"]
            edge_map[key]["avg_strength"] = old_avg + (strength - old_avg) / n
            
            if chunk_id not in edge_map[key]["chunk_ids"]:
                edge_map[key]["chunk_ids"].append(chunk_id)
            
            if evidence and len(edge_map[key]["evidences"]) < MAX_EVIDENCES_PER_EDGE:
                if evidence not in edge_map[key]["evidences"]:
                    edge_map[key]["evidences"].append(evidence)
            
            if description and len(edge_map[key]["descriptions"]) < 2:
                if description not in edge_map[key]["descriptions"]:
                    edge_map[key]["descriptions"].append(description)
    
    edges = list(edge_map.values())
    
    # Sort by weight * avg_strength (combined score)
    edges.sort(key=lambda x: x["weight"] * x["avg_strength"], reverse=True)
    
    print(f"Unique edges: {len(edges)}")
    print(f"Total weight: {sum(e['weight'] for e in edges)}")
    
    if edges:
        print(f"\nTop 5 edges:")
        for e in edges[:5]:
            print(f"  ({e['source']}, {e['relation']}, {e['target']}) - w:{e['weight']} s:{e['avg_strength']:.1f}")
    
    return edges


# ============ SAVE/LOAD ============

def save_merged_edges(edges: List[Dict], cache_dir: str):
    """Save merged edges to JSON."""
    filepath = os.path.join(cache_dir, "edges_merged.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)
    print(f"Saved to {filepath}")


def convert_to_triples_format(edges: List[Dict], cache_dir: str):
    """Convert to triples.json format for graph builder."""
    triples = [{
        "source": e["source"],
        "relation": e["relation"],
        "target": e["target"],
        "weight": e["weight"],
        "strength": round(e["avg_strength"], 1),
        "chunk_ids": e["chunk_ids"]
    } for e in edges]
    
    filepath = os.path.join(cache_dir, "triples.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(triples, f, ensure_ascii=False, indent=2)
    print(f"Saved to {filepath}")


# ============ CONVENIENCE ============

def extract_and_merge(
    chunks: List[Dict],
    I_c2e: Dict[str, List[str]],
    llm_choice: str = "gpt",
    cache_dir: str = "./cache",
    max_relations: int = DEFAULT_MAX_RELATIONS
) -> List[Dict]:
    """Extract + Merge in one call."""
    extract_relations_llm(chunks, I_c2e, llm_choice, cache_dir, max_relations)
    edges = merge_relation_instances(cache_dir)
    save_merged_edges(edges, cache_dir)
    convert_to_triples_format(edges, cache_dir)
    return edges
