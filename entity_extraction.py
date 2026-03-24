import os, re, json, spacy
from collections import Counter, defaultdict

# These are the specific types of entities we care about (people, organizations, locations, etc.)
NER_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP","EVENT"}

# Sometimes the model gets confused and thinks pronouns are entities, so we filter these out manually.
PRONOUN_LIKE = {"he", "she", "it", "they", "we", "i", "you", "this", "that"}

def load_spacy(model_name="en_core_web_lg"):
    """Loads the SpaCy language model. If it's not downloaded yet, it will grab it automatically."""
    try:
        return spacy.load(model_name)
    except OSError:
        spacy.cli.download(model_name)
        return spacy.load(model_name)

def canonicalize(text: str):
    """Cleans up the extracted entity text to make sure variations of the same name map to the same string."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", text)
    return text

def extract_ner_entities(doc, min_len=3):
    """Runs through a processed SpaCy document and pulls out all the valid entities it found."""
    ents = set()
    for ent in doc.ents:
        if ent.label_ not in NER_LABELS:
            continue
        e = canonicalize(ent.text)
        if len(e) < min_len or e in PRONOUN_LIKE:
            continue
        ents.add(e)
    return sorted(ents)

def extract_entities_from_chunks(chunks, nlp, batch_size=32):
    """
    Processes a bunch of text chunks and builds two lookup dictionaries:
    - One that tells you which chunks mention a specific entity
    - One that tells you all the entities found inside a specific chunk
    """
    I_c2e = {}
    I_e2c = defaultdict(list)

    texts = [c["text"] for c in chunks]
    for doc, chunk in zip(nlp.pipe(texts, batch_size=batch_size), chunks):
        cid = f"L0_{chunk['chunk_id']}"
        ents = extract_ner_entities(doc)

        I_c2e[cid] = ents
        for e in ents:
            I_e2c[e].append(cid)

    return dict(I_e2c), I_c2e

def save_entities(I_e2c, I_c2e, cache_dir):
    """Saves our entity mapping dictionaries out to JSON files so we can load them later without reprocessing."""
    entities_dir = os.path.join(cache_dir, "entities")
    os.makedirs(entities_dir, exist_ok=True)

    with open(os.path.join(entities_dir, "I_e2c.json"), "w", encoding="utf-8") as f:
        json.dump(I_e2c, f, ensure_ascii=False, indent=2)

    with open(os.path.join(entities_dir, "I_c2e.json"), "w", encoding="utf-8") as f:
        json.dump(I_c2e, f, ensure_ascii=False, indent=2)

    print(f"Entities saved to {entities_dir}")
