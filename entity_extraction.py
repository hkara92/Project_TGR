import os, re, json, spacy
from collections import Counter, defaultdict

NER_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP","EVENT"}
PRONOUN_LIKE = {"he", "she", "it", "they", "we", "i", "you", "this", "that"}

def load_spacy(model_name="en_core_web_lg"):
    try:
        return spacy.load(model_name)
    except OSError:
        spacy.cli.download(model_name)
        return spacy.load(model_name)

def canonicalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", text)
    return text

def extract_ner_entities(doc, min_len=3):
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
    Returns:
        I_e2c: dict[entity] -> list[chunk_id]
        I_c2e: dict[chunk_id] -> list[entity]
    """
    I_c2e = {}
    I_e2c = defaultdict(list)

    texts = [c["text"] for c in chunks]
    for doc, chunk in zip(nlp.pipe(texts, batch_size=batch_size), chunks):
        cid = chunk["chunk_id"]
        ents = extract_ner_entities(doc)

        I_c2e[cid] = ents
        for e in ents:
            I_e2c[e].append(cid)

    return dict(I_e2c), I_c2e

def save_entities(I_e2c, I_c2e, cache_dir):
    entities_dir = os.path.join(cache_dir, "entities")
    os.makedirs(entities_dir, exist_ok=True)

    with open(os.path.join(entities_dir, "I_e2c.json"), "w", encoding="utf-8") as f:
        json.dump(I_e2c, f, ensure_ascii=False, indent=2)

    with open(os.path.join(entities_dir, "I_c2e.json"), "w", encoding="utf-8") as f:
        json.dump(I_c2e, f, ensure_ascii=False, indent=2)

    print(f"Entities saved to {entities_dir}")
