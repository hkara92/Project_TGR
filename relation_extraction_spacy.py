"""
Rules implemented:
1. Active SVO (nsubj ← VERB → dobj)
2. Prepositional Object (nsubj ← VERB → prep → pobj)
3. Passive Voice (nsubjpass ← VERB → agent/prep(by) → pobj)
4. Dialogue SAID_TO (nsubj ← say/tell/ask → prep(to) → pobj)
5. Copula IS/HAS_ROLE (nsubj ← is/was → attr)
6. Fallback RELATED_TO (if ≥2 entities but no typed triple)
"""

import re
import os
import json
from itertools import combinations


# ============ UTILITIES ============

def canonicalize(text):
    """Normalize text (same as entity_extraction)."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


# Relation normalization dictionary
RELATION_MAP = {
    # Employment
    ("work", "at"): "WORKS_AT",
    ("work", "for"): "WORKS_FOR",
    ("employ", None): "EMPLOYS",
    ("hire", None): "HIRES",
    ("join", None): "JOINS",
    
    # Location
    ("live", "in"): "LIVES_IN",
    ("live", "at"): "LIVES_AT",
    ("locate", "in"): "LOCATED_IN",
    ("base", "in"): "BASED_IN",
    ("go", "to"): "GOES_TO",
    ("travel", "to"): "TRAVELS_TO",
    ("arrive", "at"): "ARRIVES_AT",
    ("arrive", "in"): "ARRIVES_IN",
    ("visit", None): "VISITS",
    
    # Social
    ("meet", None): "MEETS",
    ("meet", "with"): "MEETS_WITH",
    ("marry", None): "MARRIES",
    ("know", None): "KNOWS",
    ("love", None): "LOVES",
    
    # Actions
    ("kill", None): "KILLS",
    ("buy", None): "BUYS",
    ("sell", None): "SELLS",
    ("sell", "to"): "SELLS_TO",
    ("give", None): "GIVES",
    ("give", "to"): "GIVES_TO",
    ("take", None): "TAKES",
    ("make", None): "MAKES",
    ("build", None): "BUILDS",
    ("found", None): "FOUNDS",
    ("create", None): "CREATES",
    
    # Communication (for dialogue rule)
    ("say", "to"): "SAID_TO",
    ("tell", None): "TELLS",
    ("ask", None): "ASKS",
    ("reply", "to"): "REPLIED_TO",
    ("speak", "to"): "SPEAKS_TO",
    ("shout", "at"): "SHOUTS_AT",
    ("whisper", "to"): "WHISPERS_TO",
    ("call", None): "CALLS",
}

DIALOGUE_VERBS = {"say", "tell", "ask", "reply", "speak", "shout", "whisper", "call", "cry", "exclaim"}


def normalize_relation(verb_lemma, prep=None):
    """Map verb + preposition to canonical relation."""
    key = (verb_lemma, prep)
    if key in RELATION_MAP:
        return RELATION_MAP[key]
    
    # Try without preposition
    key_no_prep = (verb_lemma, None)
    if key_no_prep in RELATION_MAP:
        return RELATION_MAP[key_no_prep]
    
    # Fallback: VERB or VERB_PREP in uppercase
    if prep:
        return f"{verb_lemma}_{prep}".upper()
    return verb_lemma.upper()


# ============ ENTITY MENTION FINDING ============

def find_mentions_in_sentence(sentence_text, chunk_entities):
    """
    Find which entities appear in sentence (longest match first, non-overlapping).
    
    Returns list of mentions: [{"text": "john smith", "start": 0, "end": 10}, ...]
    """
    sentence_lower = sentence_text.lower()
    mentions = []
    
    # Sort entities by length descending (longest match first)
    sorted_entities = sorted(chunk_entities, key=len, reverse=True)
    
    # Track which character positions are already covered
    covered = set()
    
    for entity in sorted_entities:
        # Find all occurrences of this entity in the sentence
        start = 0
        while True:
            pos = sentence_lower.find(entity, start)
            if pos == -1:
                break
            
            end = pos + len(entity)
            
            # Check if this span overlaps with already covered positions
            span_positions = set(range(pos, end))
            if not span_positions & covered:
                # No overlap - add this mention
                mentions.append({
                    "text": entity,
                    "start": pos,
                    "end": end
                })
                covered.update(span_positions)
            
            start = pos + 1
    
    # Sort by position
    mentions.sort(key=lambda x: x["start"])
    return mentions


def match_token_to_mention(token, mentions):
    """
    Match a SpaCy token (or span) to a known entity mention.
    Returns the canonical entity text or None.
    """
    token_start = token.idx
    token_end = token.idx + len(token.text)
    token_text = canonicalize(token.text)
    
    for mention in mentions:
        # Check if token falls within mention span
        if mention["start"] <= token_start and token_end <= mention["end"]:
            return mention["text"]
        
        # Check if token text matches mention
        if token_text == mention["text"]:
            return mention["text"]
        
        # Check if token is part of mention (for multi-word entities)
        if token_text in mention["text"].split():
            return mention["text"]
    
    return None


def get_full_span_text(token, mentions):
    """
    Get the full entity span for a token (handles multi-word entities).
    """
    # First try to match the token directly
    match = match_token_to_mention(token, mentions)
    if match:
        return match
    
    # Try to get the full subtree (for compound nouns like "John Smith")
    span_text = " ".join([t.text for t in token.subtree])
    span_canonical = canonicalize(span_text)
    
    for mention in mentions:
        if span_canonical == mention["text"] or mention["text"] in span_canonical:
            return mention["text"]
    
    return None


# ============ RELATION EXTRACTION RULES ============

def extract_relations_from_sentence(sent, mentions, chunk_entities):
    """
    Extract triples from a single sentence using dependency rules.
    
    Args:
        sent: SpaCy Span (sentence)
        mentions: List of entity mentions found in this sentence
        chunk_entities: Set of all entities in this chunk (for validation)
    
    Returns:
        List of triple dicts
    """
    triples = []
    chunk_entities_set = set(chunk_entities)
    
    # Skip if fewer than 2 mentions
    if len(mentions) < 2:
        return triples
    
    for token in sent:
        # Only process verbs
        if token.pos_ not in {"VERB", "AUX"}:
            continue
        
        verb_lemma = token.lemma_.lower()
        
        # ============ RULE 1: Active SVO ============
        # nsubj ← VERB → dobj
        subject = None
        obj = None
        
        for child in token.children:
            if child.dep_ == "nsubj":
                subject = get_full_span_text(child, mentions)
            elif child.dep_ in {"dobj", "obj"}:
                obj = get_full_span_text(child, mentions)
        
        if subject and obj and subject in chunk_entities_set and obj in chunk_entities_set:
            rel = normalize_relation(verb_lemma, None)
            triples.append({
                "sub": subject,
                "rel": rel,
                "rel_raw": verb_lemma,
                "obj": obj,
                "confidence": 1.0,
                "rule": "active_svo"
            })
        
        # ============ RULE 2: Prepositional Object ============
        # nsubj ← VERB → prep → pobj
        for child in token.children:
            if child.dep_ == "prep":
                prep_text = child.text.lower()
                
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        prep_obj = get_full_span_text(grandchild, mentions)
                        
                        if subject and prep_obj and subject in chunk_entities_set and prep_obj in chunk_entities_set:
                            rel = normalize_relation(verb_lemma, prep_text)
                            rel_raw = f"{verb_lemma} {prep_text}"
                            triples.append({
                                "sub": subject,
                                "rel": rel,
                                "rel_raw": rel_raw,
                                "obj": prep_obj,
                                "confidence": 0.9,
                                "rule": "prep_object"
                            })
        
        # ============ RULE 3: Passive Voice ============
        # nsubjpass ← VERB → agent/prep(by) → pobj
        passive_subject = None
        agent = None
        
        for child in token.children:
            if child.dep_ == "nsubjpass":
                passive_subject = get_full_span_text(child, mentions)
            elif child.dep_ == "agent" or (child.dep_ == "prep" and child.text.lower() == "by"):
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        agent = get_full_span_text(grandchild, mentions)
        
        if passive_subject and agent and passive_subject in chunk_entities_set and agent in chunk_entities_set:
            # Swap direction: agent is the real subject
            rel = normalize_relation(verb_lemma, None)
            triples.append({
                "sub": agent,
                "rel": rel,
                "rel_raw": verb_lemma,
                "obj": passive_subject,
                "confidence": 0.8,
                "rule": "passive"
            })
        
        # ============ RULE 4: Dialogue SAID_TO ============
        if verb_lemma in DIALOGUE_VERBS and subject:
            for child in token.children:
                if child.dep_ == "prep" and child.text.lower() == "to":
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            addressee = get_full_span_text(grandchild, mentions)
                            
                            if addressee and subject in chunk_entities_set and addressee in chunk_entities_set:
                                triples.append({
                                    "sub": subject,
                                    "rel": "SAID_TO",
                                    "rel_raw": f"{verb_lemma} to",
                                    "obj": addressee,
                                    "confidence": 0.8,
                                    "rule": "dialogue"
                                })
        
        # ============ RULE 5: Copula (IS / HAS_ROLE) ============
        if verb_lemma == "be":
            copula_subject = None
            attribute = None
            
            for child in token.children:
                if child.dep_ == "nsubj":
                    copula_subject = get_full_span_text(child, mentions)
                elif child.dep_ in {"attr", "acomp"}:
                    attribute = get_full_span_text(child, mentions)
            
            if copula_subject and attribute and copula_subject in chunk_entities_set and attribute in chunk_entities_set:
                triples.append({
                    "sub": copula_subject,
                    "rel": "IS",
                    "rel_raw": "is",
                    "obj": attribute,
                    "confidence": 0.9,
                    "rule": "copula"
                })
    
    return triples


def fallback_related_to(mentions, chunk_entities, existing_triples):
    """
    RULE 6: Create RELATED_TO edges for entity pairs with no typed relation.
    Only if ≥2 entities in sentence but no typed triple covers them.
    """
    chunk_entities_set = set(chunk_entities)
    fallback_triples = []
    
    # Get entities that already have relations
    covered_pairs = set()
    for t in existing_triples:
        pair = tuple(sorted([t["sub"], t["obj"]]))
        covered_pairs.add(pair)
    
    # Get valid mentions (those in chunk_entities)
    valid_mentions = [m["text"] for m in mentions if m["text"] in chunk_entities_set]
    
    # Create RELATED_TO for uncovered pairs
    for e1, e2 in combinations(valid_mentions, 2):
        pair = tuple(sorted([e1, e2]))
        if pair not in covered_pairs:
            fallback_triples.append({
                "sub": e1,
                "rel": "RELATED_TO",
                "rel_raw": "related_to",
                "obj": e2,
                "confidence": 0.4,
                "rule": "fallback"
            })
            covered_pairs.add(pair)
    
    return fallback_triples


# ============ MAIN EXTRACTION ============

def extract_relations(chunks, I_c2e, nlp):
    """
    Extract relations from all chunks.
    
    Args:
        chunks: List of {"chunk_id": str, "text": str, ...}
        I_c2e: dict[chunk_id] → list[entity]
        nlp: SpaCy model
    
    Returns:
        List of triple dicts with chunk_id, sentence_idx, evidence_text
    """
    all_triples = []
    
    for i, chunk in enumerate(chunks):
        chunk_id = f"L0_{chunk['chunk_id']}"
        text = chunk["text"]
        chunk_entities = I_c2e.get(chunk_id, [])
        
        if len(chunk_entities) < 2:
            continue
        
        # Parse chunk
        doc = nlp(text)
        
        # Process each sentence
        for sent_idx, sent in enumerate(doc.sents):
            sentence_text = sent.text.strip()
            
            # Find entity mentions in this sentence
            mentions = find_mentions_in_sentence(sentence_text, chunk_entities)
            
            if len(mentions) < 2:
                continue
            
            # Extract typed relations
            typed_triples = extract_relations_from_sentence(sent, mentions, chunk_entities)
            
            # Add fallback RELATED_TO
            fallback_triples = fallback_related_to(mentions, chunk_entities, typed_triples)
            
            # Add metadata to all triples
            for triple in typed_triples + fallback_triples:
                triple["chunk_id"] = chunk_id
                triple["sentence_idx"] = sent_idx
                all_triples.append(triple)
        
        # Progress log
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(chunks)} chunks, {len(all_triples)} triples extracted")
    
    print(f"Relation extraction complete: {len(chunks)} chunks, {len(all_triples)} triples")
    return all_triples


# ============ SAVE / LOAD ============

def save_triples(triples, cache_dir):
    """Save triples to JSON."""
    filepath = os.path.join(cache_dir, "triples.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(triples, f, ensure_ascii=False, indent=2)
    print(f"Triples saved to {filepath}")


def load_triples(cache_dir):
    """Load triples from JSON."""
    filepath = os.path.join(cache_dir, "triples.json")
    with open(filepath, "r", encoding="utf-8") as f:
        triples = json.load(f)
    print(f"Loaded {len(triples)} triples from {filepath}")
    return triples


# ============ EXAMPLE USAGE ============

if __name__ == "__main__":
    import spacy
    
    # Load SpaCy
    nlp = spacy.load("en_core_web_lg")
    
    # Sample data
    chunks = [
        {
            "chunk_id": "chunk_0",
            "text": "John Smith works at Microsoft in London. He founded the company in 1990."
        },
        {
            "chunk_id": "chunk_1",
            "text": "Mary called John about the project. She said to him that the deadline was near."
        },
        {
            "chunk_id": "chunk_2",
            "text": "The CEO is John Smith. Microsoft was founded by Bill Gates."
        }
    ]
    
    I_c2e = {
        "chunk_0": ["john smith", "john", "smith", "microsoft", "london", "company"],
        "chunk_1": ["mary", "john", "project", "deadline"],
        "chunk_2": ["ceo", "john smith", "john", "smith", "microsoft", "bill gates", "bill", "gates"]
    }
    
    # Extract relations
    triples = extract_relations(chunks, I_c2e, nlp)
    
    # Print results
    print("\n" + "="*60)
    print("EXTRACTED TRIPLES")
    print("="*60)
    
    for t in triples:
        print(f"\n[{t['rule']}] ({t['sub']}, {t['rel']}, {t['obj']})")
        print(f"  Raw: {t['rel_raw']}, Confidence: {t['confidence']}")
        print(f"  Evidence: {t['evidence_text'][:80]}...")
    
    # Save
    os.makedirs("./cache/test", exist_ok=True)
    save_triples(triples, "./cache/test")