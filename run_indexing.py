# from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import os
# Fix for Windows joblib/sklearn threading crash/hang
os.environ["LOKY_MAX_CPU_COUNT"] = "1" 
import json
import logging
import warnings

# SILENCE CLUTTER
# 1. Suppress TensorFlow/HuggingFace info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 2. Suppress Python Warnings
warnings.filterwarnings("ignore")

# 3. Suppress Library Logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Set Hugging Face cache to local directory
os.environ["HF_HOME"] = os.path.abspath("./models")

from dataloader import load_dataset
from preprocessing import clean_text, chunk_text, save_chunks
from summary_tree import build_tree, save_tree, create_summarizer
from entity_extraction import load_spacy, extract_entities_from_chunks, save_entities
from llm import call_llm, get_tokenizer, get_embeddings
# from relation_extraction_spacy import extract_relations, save_triples
import numpy as np
import functools

# CONFIG
EMBEDDER_MODEL = "text-embedding-3-large"
LLM_CHOICE = "gpt"
DATASET_NAME = "InfiniteChoice"
DATASET_PATH = "./data/InfiniteBench/longbook_choice_eng.jsonl"
CACHE_DIR = "./cache"
CHUNK_SIZE = 1200
OVERLAP = 100
CHUNKING_METHOD = "tokens"  # Options: "tokens", "semantic_iqr"

# INITIALIZE MODELS ONCE
print("Step 0: Initialize Models")
print(f"Using Embedder Model: {EMBEDDER_MODEL} (via OpenAI API)...")

# Wrapper to make OpenAI embedding look like SentenceTransformer
class EmbedderWrapper:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def encode(self, texts, show_progress_bar=False, **kwargs):
        # The summary_tree and chunking might pass show_progress_bar, we ignore it 
        # because the API call is atomic.
        print(f"    [Embedder] Generating {len(texts)} embeddings...")
        
        # Ensure we return a numpy array as expected by downstream math
        embeddings = get_embeddings(texts, model=self.model_name)
        result = np.array(embeddings)
        
        print(f"    [Embedder] Done. Shape: {result.shape}")
        return result

embedder = EmbedderWrapper(EMBEDDER_MODEL)
print("✓ Embedder ready.")

print(f"Loading Tokenizer for: {LLM_CHOICE}...")
tokenizer = get_tokenizer(LLM_CHOICE)
print("✓ Tokenizer loaded.")

print("Loading spaCy model: en_core_web_lg...")
nlp = load_spacy("en_core_web_lg")
print("✓ spaCy loaded.")

# LOAD DATASET
dataset = load_dataset(DATASET_NAME, DATASET_PATH)
book_ids = list(dataset.keys())
print(f"Loaded {len(book_ids)} books")

import time

# PROCESS ALL BOOKS
for i, book_id in enumerate(book_ids):
    print(f"\n{'='*50}")
    print(f"Processing book {i + 1}/{len(book_ids)} (ID: {book_id})")
    print(f"{'='*50}")
    
    start_time = time.time()  # <--- Start Timer

    book_data = dataset[book_id]
    book_cache_dir = os.path.join(CACHE_DIR, DATASET_NAME, book_id)
    os.makedirs(book_cache_dir, exist_ok=True)
    
    # Check if tree already exists (Resume logic)
    tree_file = os.path.join(book_cache_dir, "summary_tree", "nodes.json")
    tree_exists = os.path.exists(tree_file)
    
    chunks = []
    
    if tree_exists:
        print(f"Skipping Tree Building (Tree already exists: {tree_file})")
        # Load chunks just in case needed for later steps
        with open(os.path.join(book_cache_dir, "chunks.json"), "r", encoding="utf-8") as f:
            chunks = json.load(f)
    else:
        # 1. LOAD/CREATE CHUNKS
        chunks_file = os.path.join(book_cache_dir, "chunks.json")
        if os.path.exists(chunks_file):
            print(f"Loading chunks from cache: {chunks_file}")
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        else:
            # Clean and chunk
            text = clean_text(book_data["book_text"])
            chunks = chunk_text(
                text, 
                method=CHUNKING_METHOD, 
                tokenizer=tokenizer, 
                chunk_size=CHUNK_SIZE, 
                overlap=OVERLAP, 
                nlp=nlp,
                embedder=embedder
            )
            print(f"Created {len(chunks)} chunks")
            save_chunks(chunks, book_cache_dir)
            print("Saved chunks to cache")
        
        # 2. BUILD SUMMARY TREE
        print("Step 2: Building Summary Tree...")
        
        # Use the new two-prompt summarizer factory
        llm_function = lambda prompt: call_llm(prompt, llm_choice=LLM_CHOICE)
        summarizer_fn = create_summarizer(llm_function)
    
        nodes, levels = build_tree(
            chunks=chunks,
            embedder=lambda texts: embedder.encode(texts, show_progress_bar=True),
            summarizer=summarizer_fn
        )
        save_tree(nodes, levels, book_cache_dir)
        print(f"Built tree with {len(nodes)} nodes across {len(levels)} levels")
        
        # Stop Timer & Save
        duration = time.time() - start_time
        print(f"⏱️ Time taken for Tree Indexing: {duration:.2f} seconds")
        
        with open(os.path.join(book_cache_dir, "indexing_time_tree.txt"), "w") as f:
            f.write(f"{duration:.4f}")
        
    # Build FAISS Index automatically
    print("Step 3: Building FAISS Index...")
    faiss_index_path = os.path.join(book_cache_dir, "summary_tree", "tree.index")
    
    if os.path.exists(faiss_index_path):
        print(f"  > Skipping: FAISS index already exists.")
    else:
        try:
            from build_faiss_index import build_and_save_faiss
            build_and_save_faiss(book_cache_dir)
            print("✓ FAISS Index built.")
        except Exception as e:
            print(f"FAISS Build Failed: {e}")
    
    # Extract entities and SAVE
    print("Step 4: Extracting Entities...")
    entities_file = os.path.join(book_cache_dir, "entities", "I_e2c.json")
    
    # Check if entities exist
    if os.path.exists(entities_file):
         print(f"  > Skipping: Entities already exist.")
         # Needed for Step 5: Load existing I_c2e if we are skipping extraction
         with open(os.path.join(book_cache_dir, "entities", "I_c2e.json"), "r", encoding="utf-8") as f:
             I_c2e = json.load(f)
    else:
        # from entity_extraction import extract_entities_from_chunks, save_entities # Already imported at top, but keeping if relying on lazy load semantics or just ensuring it's available
        start_ent = time.time()
        I_e2c, I_c2e = extract_entities_from_chunks(chunks, nlp)
        save_entities(I_e2c, I_c2e, book_cache_dir)
        print(f"  ✓ Extracted {len(I_e2c)} unique entities in {time.time()-start_ent:.2f}s")
    
    # Extract relations (LLM-based) and SAVE
    print("Step 5: Extracting Relations (LLM)...")
    from relation_extraction_llm import extract_and_merge_relations
    
    # We pass the cache_dir so it saves directly there
    extract_and_merge_relations(chunks, I_c2e, llm_choice=LLM_CHOICE, cache_dir=book_cache_dir)
    print("  ✓ Relations extracted and merged.")
    
    print(f"Book {book_id} complete!")

print("\n✓ All books indexed!")