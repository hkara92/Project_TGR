
from turtle import clear
import re
import unicodedata
import numpy as np
from typing import List, Dict, Any


def clean_text(text: str) -> str:
    """normalize unicode, whitespace, preserve paragraphs."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[ \t]+', ' ', text)           # Multiple spaces → single
    text = re.sub(r'\n{3,}', '\n\n', text)        # 3+ newlines → 2
    return text.strip()


def chunk_by_tokens(text: str, tokenizer, chunk_size: int = 1000, overlap: int = 100) -> List[Dict[str, Any]]:
    """Split text into overlapping token-based chunks."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    
    chunks = []
    start = 0
    step = chunk_size - overlap
    
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_text = tokenizer.decode(token_ids[start:end], skip_special_tokens=True).strip()
        
        if chunk_text:
            chunks.append({"chunk_id": f"chunk_{len(chunks)}", "text": chunk_text, "order": len(chunks)})
        
        if end >= len(token_ids):
            break
        start += step
    
    return chunks


def chunk_by_semantic_iqr(text: str, nlp, embedder, min_sentences: int = 3, max_sentences: int = 20) -> List[Dict[str, Any]]:
    """Split by semantic similarity breakpoints using IQR."""
    # Get sentences
    sentences = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    
    if len(sentences) <= min_sentences:
        return [{"chunk_id": "chunk_0", "text": " ".join(sentences), "order": 0}]
    
    # Embed and compute consecutive similarities
    embeddings = embedder.encode(sentences, show_progress_bar=False)
    similarities = [
        np.dot(embeddings[i], embeddings[i+1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]))
        for i in range(len(embeddings) - 1)
    ]
    
    # Find breakpoints using IQR threshold
    q1, q3 = np.percentile(similarities, [25, 75])
    threshold = q1 - 1.5 * (q3 - q1)
    breakpoints = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]
    
    # Group sentences into chunks
    chunks = []
    start = 0
    
    for bp in breakpoints + [len(sentences)]:
        if bp - start >= min_sentences:
            chunk_text = " ".join(sentences[start:bp])
            chunks.append({"chunk_id": f"chunk_{len(chunks)}", "text": chunk_text, "order": len(chunks)})
            start = bp
    
    # Handle remaining sentences
    if start < len(sentences):
        remaining = " ".join(sentences[start:])
        if chunks:
            chunks[-1]["text"] += " " + remaining  # Merge with last
        else:
            chunks.append({"chunk_id": "chunk_0", "text": remaining, "order": 0})
    
    return chunks


def chunk_text(text: str, method: str = "tokens", **kwargs) -> List[Dict[str, Any]]:
    """Unified interface for chunking."""
    if method == "tokens":
        return chunk_by_tokens(text, kwargs["tokenizer"], kwargs.get("chunk_size", 1000), kwargs.get("overlap", 100))
    elif method == "semantic_iqr":
        return chunk_by_semantic_iqr(text, kwargs["nlp"], kwargs["embedder"], kwargs.get("min_sentences", 3), kwargs.get("max_sentences", 20))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tokens' or 'semantic_iqr'")

