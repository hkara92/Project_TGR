"""
entity_extraction.py

Extract entities from chunks, build lookup indexes, and compute sentence-level
co-occurrence candidate pairs (for later LLM relation extraction).
"""

import re
import os
import json
import spacy
# from itertools import combinations
# from collections import defaultdict


def load_spacy(model_name="en_core_web_lg"):
    """
    Load SpaCy model. Auto-download if not found.
    Call this ONCE in run_indexing.py and pass nlp to other functions.
    """
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"Downloading SpaCy model: {model_name}")
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)

    print(f"SpaCy model loaded: {model_name}")
    return nlp


def canonicalize(text):
    """Normalize entity text."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


PRONOUN_LIKE = {"he", "she", "it", "they", "we", "i", "you", "this", "that"}

# Start small; expand after you inspect outputs on your dataset
GENERIC_NOUNS = {
    "thing", "way", "time", "people", "man", "woman", "day", "year", "lot",
    "system", "method", "model", "data", "process", "approach", "result",
    "information", "analysis", "problem", "value", "study", "work", "use",
}


def extract_entities_from_text(
    text,
    nlp,
    np_min_words: int = 2,
    np_max_words: int = 6,
    np_stopword_ratio: float = 0.6,
    min_token_len: int = 2,
    doc=None,
):
    """
    Extract entities from a single text using SpaCy.

    Extracts:
    1) Named entities (PERSON, ORG, GPE, LOC)
    2) Noun phrases (filtered by length + stopword ratio)
    3) Common nouns (NOUN) - lemmatized (stopwords + generic filtered)
    4) Proper nouns (PROPN) (stopwords filtered)

    Returns list of canonical entity strings (deduplicated).
    """
    doc = doc if doc is not None else nlp(text)
    entities = set()

    # Track token positions that are part of named entities
    ner_token_positions = set()

    # 1) Named Entities (PERSON, ORG, GPE, LOC)
    target_labels = {"PERSON", "ORG", "GPE", "LOC"}

    for ent in doc.ents:
        if ent.label_ not in target_labels:
            continue

        for token in ent:
            ner_token_positions.add(token.i)

        ent_text = canonicalize(ent.text)
        if len(ent_text) < min_token_len:
            continue

        # Keep full PERSON names (do NOT split into parts)
        entities.add(ent_text)

    # 2) Noun Phrases (filtered)
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ in {"PRON", "DET"}:
            continue

        toks = [t for t in chunk if t.is_alpha and not t.is_punct and not t.is_space]
        if not toks:
            continue

        chunk_text = canonicalize(chunk.text)
        if len(chunk_text) < min_token_len:
            continue

        if chunk_text in PRONOUN_LIKE:
            continue

        words = chunk_text.split()
        if not (np_min_words <= len(words) <= np_max_words):
            continue

        stop_cnt = 0
        for w in words:
            if nlp.vocab[w].is_stop:
                stop_cnt += 1
        if (stop_cnt / max(1, len(words))) > np_stopword_ratio:
            continue

        entities.add(chunk_text)

    # 3) Common Nouns (NOUN) - lemmatized
    # 4) Proper Nouns (PROPN)
    for token in doc:
        if token.i in ner_token_positions:
            continue
        if not token.is_alpha:
            continue
        if token.is_stop:
            continue

        if token.pos_ == "NOUN":
            noun_text = canonicalize(token.lemma_)
            if len(noun_text) < min_token_len:
                continue
            if noun_text in GENERIC_NOUNS:
                continue
            entities.add(noun_text)

        elif token.pos_ == "PROPN":
            propn_text = canonicalize(token.text)
            if len(propn_text) < min_token_len:
                continue
            entities.add(propn_text)

    return list(entities)


# def compute_candidate_pairs(doc, nlp, chunk_entities, min_pair_count: int = 1):
#     """
#     Sentence-level co-occurrence pairs within ONE chunk.

#     Returns:
#         pairs: list of {"e1","e2","count"} (JSON-friendly)
#     """
#     E_chunk = set(chunk_entities)
#     pair_counts = defaultdict(int)

#     for sent in doc.sents:
#         # Extract sentence entities using SAME logic (no extra parsing of full chunk)
#         sent_doc = sent.as_doc()
#         sent_entities = set(extract_entities_from_text(sent.text, nlp, doc=sent_doc))
#         sent_entities = sorted(sent_entities.intersection(E_chunk))

#         for e1, e2 in combinations(sent_entities, 2):
#             pair_counts[(e1, e2)] += 1

#     pairs = [
#         {"e1": e1, "e2": e2, "count": c}
#         for (e1, e2), c in pair_counts.items()
#         if c >= min_pair_count
#     ]
#     return pairs

def extract_entities_from_chunks(chunks, nlp):
    """
    Returns:
        I_e2c: dict[entity] -> list[chunk_id]
        I_c2e: dict[chunk_id] -> list[entity]
    """
    I_e2c = {}
    I_c2e = {}

    for i, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]
        text = chunk["text"]

        doc = nlp(text)
        entities = extract_entities_from_text(text, nlp, doc=doc)
        I_c2e[chunk_id] = entities

        for entity in entities:
            I_e2c.setdefault(entity, []).append(chunk_id)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(chunks)} chunks, {len(I_e2c)} unique entities")

    print(f"Entity extraction complete: {len(chunks)} chunks, {len(I_e2c)} unique entities")
    return I_e2c, I_c2e


def save_entities(I_e2c, I_c2e, cache_dir):
    entities_dir = os.path.join(cache_dir, "entities")
    os.makedirs(entities_dir, exist_ok=True)

    with open(os.path.join(entities_dir, "I_e2c.json"), "w", encoding="utf-8") as f:
        json.dump(I_e2c, f, ensure_ascii=False, indent=2)

    with open(os.path.join(entities_dir, "I_c2e.json"), "w", encoding="utf-8") as f:
        json.dump(I_c2e, f, ensure_ascii=False, indent=2)

    print(f"Entities saved to {entities_dir}")


def load_entities(cache_dir):
    entities_dir = os.path.join(cache_dir, "entities")

    with open(os.path.join(entities_dir, "I_e2c.json"), "r", encoding="utf-8") as f:
        I_e2c = json.load(f)

    with open(os.path.join(entities_dir, "I_c2e.json"), "r", encoding="utf-8") as f:
        I_c2e = json.load(f)

    print(f"Entities loaded from {entities_dir}")
    return I_e2c, I_c2e

