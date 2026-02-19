"""
run_indexing.py

Indexing pipeline: loads the dataset, chunks the text, builds a
summary tree, extracts entities and relations, and creates FAISS
indexes. Each book is processed independently with results cached
to disk.
"""

import os
import time
import json
import logging
import warnings
import numpy as np
import functools
from transformers import AutoTokenizer

# suppress noisy logs
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = os.path.abspath("./models")
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from dataloader import load_dataset
from preprocessing import clean_text, chunk_text, save_chunks
from summary_tree import build_tree, save_tree, create_summarizer
from entity_extraction import load_spacy, extract_entities_from_chunks, save_entities
from llm import call_llm, get_tokenizer, get_embeddings, reset_token_usage, get_token_usage, unload_model
from relation_extraction_llm import extract_and_merge_relations
from build_indexes import extract_all_indexes

# config
LLM_CHOICE = "qwen"
EMBEDDER_MODEL = "bge" if LLM_CHOICE == "qwen" else "text-embedding-3-large"
DATASET_NAME = "InfiniteQA"
DATASET_PATH = "./data/InfiniteBench/longbook_qa_eng.jsonl"
CACHE_DIR = "./cache"
CHUNK_SIZE = 1200
OVERLAP = 100
CHUNKING_METHOD = "tokens"


# simple wrapper so the embedder can be passed around like an object
class EmbedderWrapper:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, texts, show_progress_bar=False, **kwargs):
        return np.array(get_embeddings(texts, model=self.model_name))


# init
print("\nSetting up models...")
embedder = EmbedderWrapper(EMBEDDER_MODEL)
tokenizer = get_tokenizer(LLM_CHOICE)
nlp = load_spacy("en_core_web_lg")

dataset = load_dataset(DATASET_NAME, DATASET_PATH)
book_ids = list(dataset.keys())
print(f"Models loaded. Found {len(book_ids)} books.")


# process each book
for i, book_id in enumerate(book_ids):
    print(f"\n--- Book {i+1}/{len(book_ids)} [ID: {book_id}] ---\n")
    start_time = time.time()

    book_data = dataset[book_id]
    book_cache_dir = os.path.join(CACHE_DIR, DATASET_NAME, book_id)
    os.makedirs(book_cache_dir, exist_ok=True)

    tree_file = os.path.join(book_cache_dir, "summary_tree", "nodes.json")
    chunks_file = os.path.join(book_cache_dir, "chunks.json")
    chunks = []

    # skip if the tree was already built
    if os.path.exists(tree_file) and os.path.exists(chunks_file):
        print("Tree already exists, skipping.")
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    else:
        # load or create chunks
        if os.path.exists(chunks_file):
            print("Loading chunks from cache...")
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        else:
            print("Creating chunks...")
            chunks = chunk_text(
                text=clean_text(book_data["book_text"]),
                method=CHUNKING_METHOD,
                tokenizer=tokenizer,
                chunk_size=CHUNK_SIZE,
                overlap=OVERLAP,
                nlp=nlp,
                embedder=embedder,
            )
            save_chunks(chunks, book_cache_dir)
            print(f"  Saved {len(chunks)} chunks.")

        # build the summary tree
        print("Building summary tree...")
        reset_token_usage()
        summarizer_fn = create_summarizer(lambda prompt: call_llm(prompt, model=LLM_CHOICE))
        nodes, levels = build_tree(chunks, lambda texts: embedder.encode(texts), summarizer_fn)
        save_tree(nodes, levels, book_cache_dir)

        in_tok, out_tok = get_token_usage()
        print(f"  Token usage: in={in_tok}, out={out_tok}, total={in_tok+out_tok}")

        duration_tree = time.time() - start_time
        with open(os.path.join(book_cache_dir, "indexing_time_tree.txt"), "w") as f:
            f.write(f"{duration_tree:.4f}")
        print(f"  Tree done ({len(nodes)} nodes) in {duration_tree:.2f}s")

    # extract entities with spacy
    print("Extracting entities...")
    entities_file = os.path.join(book_cache_dir, "entities", "I_e2c.json")

    if os.path.exists(entities_file):
        print("  Already extracted, loading from cache.")
        with open(os.path.join(book_cache_dir, "entities", "I_c2e.json"), "r", encoding="utf-8") as f:
            I_c2e = json.load(f)
    else:
        start_ent = time.time()
        I_e2c, I_c2e = extract_entities_from_chunks(chunks, nlp)
        save_entities(I_e2c, I_c2e, book_cache_dir)
        print(f"  Found {len(I_e2c)} entities in {time.time()-start_ent:.2f}s")

    # build FAISS and inverted indexes
    print("Building search indexes...")
    try:
        extract_all_indexes(book_cache_dir)
    except Exception as e:
        print(f"  Index build failed: {e}")

    # extract relations using the LLM
    print("Extracting relations...")
    start_rel = time.time()
    extract_and_merge_relations(chunks, I_c2e, llm_choice=LLM_CHOICE, cache_dir=book_cache_dir)
    print(f"  Relations done ({time.time()-start_rel:.2f}s).")

    print(f"Book {book_id} complete!")

    # free GPU memory between books
    if LLM_CHOICE == "qwen":
        unload_model()

print("\nAll books indexed!")