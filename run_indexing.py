from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import os

from dataloader import load_dataset
from preprocessing import clean_text, chunk_text
from summary_tree import build_raptor_tree
from entity_extraction import load_spacy, extract_entities_from_chunks, save_entities

# ============ CONFIG ============
EMBEDDER_MODEL = "BAAI/bge-m3"
LLM_CHOICE = "gpt"
DATASET_NAME = "InfiniteChoice"
DATASET_PATH = "./data/InfiniteBench/longbook_choice_eng.jsonl"
CACHE_DIR = "./cache"
CHUNK_SIZE = 1200
OVERLAP = 100

# ============ INITIALIZE MODELS ONCE ============
print("Loading models...")
embedder = SentenceTransformer(EMBEDDER_MODEL)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
nlp = load_spacy("en_core_web_lg")

# ============ LOAD DATASET ============
dataset = load_dataset(DATASET_NAME, DATASET_PATH)
book_ids = list(dataset.keys())
print(f"Loaded {len(book_ids)} books")

# ============ PROCESS ALL BOOKS ============
for i, book_id in enumerate(book_ids):
    print(f"\n{'='*50}")
    print(f"Processing book {i + 1}/{len(book_ids)} (ID: {book_id})")
    print(f"{'='*50}")
    
    book_data = dataset[book_id]
    book_cache_dir = os.path.join(CACHE_DIR, DATASET_NAME, book_id)
    os.makedirs(book_cache_dir, exist_ok=True)
    
    # Step 1: Clean and chunk
    text = clean_text(book_data["book_text"])
    chunks = chunk_text(text, method="tokens", tokenizer=tokenizer, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    print(f"Created {len(chunks)} chunks")
    
    # Step 2: Build summary tree
    tree = build_raptor_tree(
        chunks=chunks,
        llm_choice=LLM_CHOICE,
        reduction_factor=10
    )
    
    # Step 3: Extract entities and SAVE
    I_e2c, I_c2e, entity_freq = extract_entities_from_chunks(chunks, nlp)
    save_entities(I_e2c, I_c2e, entity_freq, book_cache_dir)
    print(f"Extracted {len(I_e2c)} unique entities")
    
    # Step 4: Extract relations (TODO)
    # triples = extract_relations(chunks, I_c2e, nlp)
    
    # Step 5: Build graph and indexes (TODO)
    # graph = build_graph(triples)
    
    print(f"Book {book_id} complete!")

print("\n✓ All books indexed!")