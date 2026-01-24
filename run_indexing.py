from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import os

# Set Hugging Face cache to local directory
os.environ["HF_HOME"] = os.path.abspath("./models")

from dataloader import load_dataset
from preprocessing import clean_text, chunk_text, save_chunks
from summary_tree import build_tree, save_tree
from entity_extraction import load_spacy, extract_entities_from_chunks, save_entities
from llm import call_llm
# from relation_extraction_spacy import extract_relations, save_triples

# ============ CONFIG ============
EMBEDDER_MODEL = "BAAI/bge-m3"
LLM_CHOICE = "gpt"
DATASET_NAME = "InfiniteChoice"
DATASET_PATH = "./data/InfiniteBench/longbook_choice_eng.jsonl"
CACHE_DIR = "./cache"
CHUNK_SIZE = 1200
OVERLAP = 100
CHUNKING_METHOD = "tokens"  # Options: "tokens", "semantic_iqr"

# ============ INITIALIZE MODELS ONCE ============
print("Step 0: Initialize Models")
print(f"Loading/Downloading Embedder Model: {EMBEDDER_MODEL}...")
print(f"Models will be stored in: {os.environ['HF_HOME']}")
embedder = SentenceTransformer(EMBEDDER_MODEL, cache_folder=os.environ["HF_HOME"])
print("✓ Embedder loaded.")

print("Loading/Downloading Tokenizer: Qwen/Qwen2.5-7B-Instruct...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", cache_dir=os.environ["HF_HOME"])
print("✓ Tokenizer loaded.")

print("Loading spaCy model: en_core_web_lg...")
nlp = load_spacy("en_core_web_lg")
print("✓ spaCy loaded.")

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
    
    # Step 2: Build summary tree
    def summarize_list(texts):
        prompt = f"Write a comprehensive summary of the following text segments. Capture key details and entities:\n\n" + "\n---\n".join(texts)
        return call_llm(prompt, llm_choice=LLM_CHOICE)

    nodes, levels = build_tree(
        chunks=chunks,
        embedder=lambda texts: embedder.encode(texts, show_progress_bar=False),
        summarizer=summarize_list
    )
    save_tree(nodes, levels, book_cache_dir)
    print(f"Built tree with {len(nodes)} nodes across {len(levels)} levels")
    
    # Step 3: Extract entities and SAVE
    # I_e2c, I_c2e, entity_freq = extract_entities_from_chunks(chunks, nlp)
    # save_entities(I_e2c, I_c2e, entity_freq, book_cache_dir)
    # print(f"Extracted {len(I_e2c)} unique entities")
    
    # Step 4: Extract relations and SAVE
    # triples = extract_relations(chunks, I_c2e, nlp)
    # save_triples(triples, book_cache_dir)
    # print(f"Extracted {len(triples)} triples")
    
    # Step 5: Build graph and indexes (TODO)
    # graph = build_graph(triples)
    
    print(f"Book {book_id} complete!")

print("\n✓ All books indexed!")