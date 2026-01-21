"""
entity_extraction.py

Extract entities from chunks and build lookup indexes.
"""

import re
import os
import json
import spacy


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
    """
    Normalize entity text.
    """
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_entities_from_text(text, nlp):
    """
    Extract entities from a single text using SpaCy.
    
    Extracts:
    1. Named entities (PERSON, ORG, GPE, LOC)
    2. Noun phrases
    3. Common nouns (NOUN) - lemmatized
    4. Proper nouns (PROPN)
    
    Returns list of canonical entity strings (deduplicated).
    """
    doc = nlp(text)
    entities = set()
    
    # Track token positions that are part of named entities
    ner_token_positions = set()
    
    # 1. Named Entities (PERSON, ORG, GPE, LOC)
    target_labels = {"PERSON", "ORG", "GPE", "LOC"}
    
    for ent in doc.ents:
        if ent.label_ not in target_labels:
            continue
        
        for token in ent:
            ner_token_positions.add(token.i)
        
        ent_text = canonicalize(ent.text)
        
        if len(ent_text) < 2:
            continue
        
        entities.add(ent_text)
        
        # Split multi-word names
        if ent.label_ == "PERSON":
            parts = ent_text.split()
            if len(parts) >= 2:
                for part in parts:
                    if len(part) >= 2:
                        entities.add(part)
    
    # 2. Noun Phrases
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ in {"PRON", "DET"}:
            continue
        
        chunk_text = canonicalize(chunk.text)
        
        if len(chunk_text) < 2:
            continue
        
        if chunk_text in {"he", "she", "it", "they", "we", "i", "you", "this", "that"}:
            continue
        
        entities.add(chunk_text)
    
    # 3. Common Nouns (NOUN) - lemmatized
    # 4. Proper Nouns (PROPN)
    for token in doc:
        if token.i in ner_token_positions:
            continue
        
        if not token.is_alpha:
            continue
        
        if token.pos_ == "NOUN":
            noun_text = canonicalize(token.lemma_)
            
            if len(noun_text) < 2:
                continue
            
            if noun_text in {"thing", "way", "time", "people", "man", "woman", "day", "year", "lot"}:
                continue
            
            entities.add(noun_text)
        
        elif token.pos_ == "PROPN":
            propn_text = canonicalize(token.text)
            
            if len(propn_text) < 2:
                continue
            
            entities.add(propn_text)
    
    return list(entities)


def extract_entities_from_chunks(chunks, nlp):
    """
    Process all chunks and build indexes.
    
    Returns:
        I_e2c: dict[entity] → list[chunk_id]
        I_c2e: dict[chunk_id] → list[entity]
        entity_freq: dict[entity] → int
    """
    I_e2c = {}
    I_c2e = {}
    entity_freq = {}
    
    for i, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]
        text = chunk["text"]
        
        entities = extract_entities_from_text(text, nlp)
        
        I_c2e[chunk_id] = entities
        
        for entity in entities:
            if entity not in I_e2c:
                I_e2c[entity] = []
            I_e2c[entity].append(chunk_id)
            
            entity_freq[entity] = entity_freq.get(entity, 0) + 1
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(chunks)} chunks, {len(I_e2c)} unique entities")
    
    print(f"Entity extraction complete: {len(chunks)} chunks, {len(I_e2c)} unique entities")
    return I_e2c, I_c2e, entity_freq


def save_entities(I_e2c, I_c2e, entity_freq, cache_dir):
    """
    Save entity extraction outputs to cache.
    
    Args:
        I_e2c: entity → chunk_ids
        I_c2e: chunk_id → entities
        entity_freq: entity → count
        cache_dir: path to book's cache folder (e.g., "./cache/InfiniteChoice/0")
    """
    entities_dir = os.path.join(cache_dir, "entities")
    os.makedirs(entities_dir, exist_ok=True)
    
    with open(os.path.join(entities_dir, "I_e2c.json"), "w", encoding="utf-8") as f:
        json.dump(I_e2c, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(entities_dir, "I_c2e.json"), "w", encoding="utf-8") as f:
        json.dump(I_c2e, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(entities_dir, "entity_freq.json"), "w", encoding="utf-8") as f:
        json.dump(entity_freq, f, ensure_ascii=False, indent=2)
    
    print(f"Entities saved to {entities_dir}")


def load_entities(cache_dir):
    """
    Load entity extraction outputs from cache.
    
    Returns:
        I_e2c, I_c2e, entity_freq
    """
    entities_dir = os.path.join(cache_dir, "entities")
    
    with open(os.path.join(entities_dir, "I_e2c.json"), "r", encoding="utf-8") as f:
        I_e2c = json.load(f)
    
    with open(os.path.join(entities_dir, "I_c2e.json"), "r", encoding="utf-8") as f:
        I_c2e = json.load(f)
    
    with open(os.path.join(entities_dir, "entity_freq.json"), "r", encoding="utf-8") as f:
        entity_freq = json.load(f)
    
    print(f"Entities loaded from {entities_dir}")
    return I_e2c, I_c2e, entity_freq


# Example usage
if __name__ == "__main__":
    nlp = load_spacy()
    
    sample_chunks = [
        {"chunk_id": "chunk_0", "text": "John Smith works at Microsoft in London."},
        {"chunk_id": "chunk_1", "text": "Mary called John about the project."},
    ]
    
    I_e2c, I_c2e, entity_freq = extract_entities_from_chunks(sample_chunks, nlp)
    
    # Save
    save_entities(I_e2c, I_c2e, entity_freq, "./cache/test")
    
    # Load back
    I_e2c_loaded, I_c2e_loaded, entity_freq_loaded = load_entities("./cache/test")
    
    print("\nLoaded I_e2c:")
    for entity, chunks in list(I_e2c_loaded.items())[:5]:
        print(f"  {entity}: {chunks}")