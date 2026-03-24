"""
Here we handle cleaning up the raw text and splitting it into manageable chunks.
We can either chunk by exact token counts or use langchain's recursive character splitter.
"""

import re
import unicodedata
import numpy as np
import json
import os


def clean_text(text):
    """Cleans up the raw text by normalizing unicode characters and fixing weird spacing."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_by_tokens(text, tokenizer, chunk_size=1000, overlap=100):
    """Breaks down the text into smaller chunks using a specific number of tokens. We also add an overlap so we don't cut off important context."""
    # Note: some tokenizers like tiktoken will throw an error if we pass add_special_tokens
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


def chunk_by_recursive(text, chunk_size=4000, overlap=400):
    """Uses LangChain to recursively split the text by characters. It tries to keep paragraphs and sentences together."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        raise ImportError("Please install langchain-text-splitters (e.g., pip install langchain-text-splitters) to use this method.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    
    docs = splitter.create_documents([text])
    
    chunks = []
    for i, doc in enumerate(docs):
        chunks.append({
            "chunk_id": f"chunk_{i}",
            "text": doc.page_content,
            "order": i,
        })
        
    return chunks


def chunk_text(text, method="tokens", **kwargs):
    """A wrapper function that lets us pick which chunking method we want to use."""
    if method == "tokens":
        return chunk_by_tokens(
            text,
            kwargs["tokenizer"],
            kwargs.get("chunk_size", 1000),
            kwargs.get("overlap", 100),
        )
    elif method == "recursive":
        return chunk_by_recursive(
            text,
            kwargs.get("chunk_size", 4000),  # This relies on character count instead of tokens, so it's usually set around 4x higher
            kwargs.get("overlap", 400),
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tokens' or 'recursive'")


def save_chunks(chunks, cache_dir):
    """Saves our generated chunks into a JSON file for later use."""
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
