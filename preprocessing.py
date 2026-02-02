

import re
import unicodedata
import numpy as np
from typing import List, Dict, Any
import json
import os


def clean_text(text: str) -> str:
    """Normalize unicode, whitespace, preserve paragraphs."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[ \t]+', ' ', text)           # Multiple spaces → single
    text = re.sub(r'\n{3,}', '\n\n', text)        # 3+ newlines → 2
    return text.strip()


def chunk_by_tokens(text: str, tokenizer, chunk_size: int = 1000, overlap: int = 100) -> List[Dict[str, Any]]:
    """Split text into overlapping token-based chunks."""
    # Handle encode arguments (tiktoken doesn't accept add_special_tokens)
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
            chunks.append({"chunk_id": f"chunk_{len(chunks)}", "text": chunk_text, "order": len(chunks)})
        
        if end >= len(token_ids):
            break
        start += step
    
    return chunks



def chunk_by_semantic_iqr(text: str, nlp, min_sentences: int = 3, max_sentences: int = 20) -> List[Dict[str, Any]]:
    """Split by semantic similarity breakpoints using IQR."""
    from llm import get_embeddings  # Use OpenAI embeddings
    
    # Get sentences
    sentences = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    
    if len(sentences) == 0:
        return []
    
    if len(sentences) <= min_sentences:
        return [{"chunk_id": "chunk_0", "text": " ".join(sentences), "order": 0}]
    
    # Embed using OpenAI (replaces sentence-transformers)
    embeddings = get_embeddings(sentences)
    embeddings = np.array(embeddings)  # Convert to numpy
    
    similarities = []
    for i in range(len(embeddings) - 1):
        norm_i = np.linalg.norm(embeddings[i])
        norm_j = np.linalg.norm(embeddings[i + 1])
        # Protect against zero vectors
        if norm_i > 0 and norm_j > 0:
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (norm_i * norm_j)
        else:
            sim = 0.0  # Treat zero vectors as dissimilar
        similarities.append(sim)
    
    # Find breakpoints using IQR threshold
    q1, q3 = np.percentile(similarities, [25, 75])
    threshold = q1 - 1.5 * (q3 - q1)
    breakpoint_set = set(i + 1 for i, sim in enumerate(similarities) if sim < threshold)
    
    # Build chunks with min/max enforcement
    chunks = []
    current_start = 0
    
    for i in range(1, len(sentences) + 1):
        chunk_len = i - current_start
        is_breakpoint = i in breakpoint_set
        is_end = (i == len(sentences))
        
        # Split if: (breakpoint and chunk >= min) OR (chunk >= max) OR (end of sentences)
        should_split = (is_breakpoint and chunk_len >= min_sentences) or (chunk_len >= max_sentences) or is_end
        
        if should_split and chunk_len > 0:
            chunk_text = " ".join(sentences[current_start:i])
            chunks.append({"chunk_id": f"chunk_{len(chunks)}", "text": chunk_text, "order": len(chunks)})
            current_start = i
            current_start = i
    
    return chunks


def chunk_text(text: str, method: str = "tokens", **kwargs) -> List[Dict[str, Any]]:
    """Unified interface for chunking."""
    if method == "tokens":
        return chunk_by_tokens(text, kwargs["tokenizer"], kwargs.get("chunk_size", 1000), kwargs.get("overlap", 100))
    elif method == "semantic_iqr":
        return chunk_by_semantic_iqr(text, kwargs["nlp"], kwargs["embedder"], kwargs.get("min_sentences", 3), kwargs.get("max_sentences", 20))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tokens' or 'semantic_iqr'")


def save_chunks(chunks: List[Dict[str, Any]], cache_dir: str):
    """Save chunks to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
