"""
preprocessing.py

Text cleaning and chunking utilities. Supports token-based chunking
(used in the pipeline) and semantic chunking (experimental).
"""

import re
import unicodedata
import numpy as np
import json
import os


def clean_text(text):
    """Normalize unicode and whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_by_tokens(text, tokenizer, chunk_size=1000, overlap=100):
    """Split text into overlapping chunks based on token count."""
    # tiktoken doesn't accept add_special_tokens
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(text)

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))

        try:
            chunk_text = tokenizer.decode(token_ids[start:end], skip_special_tokens=True).strip()
        except TypeError:
            chunk_text = tokenizer.decode(token_ids[start:end]).strip()

        if chunk_text:
            chunks.append({
                "chunk_id": f"chunk_{len(chunks)}",
                "text": chunk_text,
                "order": len(chunks),
            })

        if end >= len(token_ids):
            break
        start += step

    return chunks


def chunk_by_semantic_iqr(text, nlp, min_sentences=3, max_sentences=20):
    """Split by semantic similarity breakpoints using IQR."""
    from llm import get_embeddings

    sentences = [s.text.strip() for s in nlp(text).sents if s.text.strip()]

    if len(sentences) == 0:
        return []

    if len(sentences) <= min_sentences:
        return [{"chunk_id": "chunk_0", "text": " ".join(sentences), "order": 0}]

    embeddings = get_embeddings(sentences)
    embeddings = np.array(embeddings)

    similarities = []
    for i in range(len(embeddings) - 1):
        norm_i = np.linalg.norm(embeddings[i])
        norm_j = np.linalg.norm(embeddings[i + 1])
        if norm_i > 0 and norm_j > 0:
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (norm_i * norm_j)
        else:
            sim = 0.0
        similarities.append(sim)

    q1, q3 = np.percentile(similarities, [25, 75])
    threshold = q1 - 1.5 * (q3 - q1)
    breakpoint_set = set(i + 1 for i, sim in enumerate(similarities) if sim < threshold)

    chunks = []
    current_start = 0

    for i in range(1, len(sentences) + 1):
        chunk_len = i - current_start
        is_breakpoint = i in breakpoint_set
        is_end = i == len(sentences)

        should_split = (is_breakpoint and chunk_len >= min_sentences) or (chunk_len >= max_sentences) or is_end

        if should_split and chunk_len > 0:
            chunk_text = " ".join(sentences[current_start:i])
            chunks.append({
                "chunk_id": f"chunk_{len(chunks)}",
                "text": chunk_text,
                "order": len(chunks),
            })
            current_start = i

    return chunks


def chunk_text(text, method="tokens", **kwargs):
    """Unified interface for chunking."""
    if method == "tokens":
        return chunk_by_tokens(
            text,
            kwargs["tokenizer"],
            kwargs.get("chunk_size", 1000),
            kwargs.get("overlap", 100),
        )
    elif method == "semantic_iqr":
        return chunk_by_semantic_iqr(
            text,
            kwargs["nlp"],
            kwargs["embedder"],
            kwargs.get("min_sentences", 3),
            kwargs.get("max_sentences", 20),
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tokens' or 'semantic_iqr'")


def save_chunks(chunks, cache_dir):
    """Save chunks to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
