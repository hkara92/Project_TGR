"""
run_eval_retrieval.py
=====================
Simplified evaluation script focusing ONLY on Step 1: Retrieval.

1. Loads dataset.
2. Initializes Retriever.
3. Retrieves chunks for a few sample questions.
4. Prints the chunks for inspection.

Does NOT generate answers or calculate accuracy yet.
"""

import json
import logging
import numpy as np
from dotenv import load_dotenv
import os
import re

# Project Modules
from retrieval_e2 import load_retriever_from_cache
from dataloader import load_dataset
from llm import get_embeddings, call_llm

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
DATASET_NAME = "InfiniteChoice"

RUN_MODE = "range" # Options: "single", "range", "all"

# "single"
BOOK_ID_TO_EVAL = "0"

# "range" (Inclusive start, Exclusive end)
RANGE_START = 0 
RANGE_END = 10 # Processes 0 to 9

# "all"
TOTAL_BOOKS = 58 

# Paths constructed dynamically
DATASET_PATH = os.path.join("data", "InfiniteBench", "longbook_choice_eng.jsonl")
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "testpassword"

# How many chunks to retrieve per question
MAX_CHUNKS = 25      
SHORTEST_PATH_K = 4

QA_PROMPT = """You are a helpful assistant. You are given a question and evidence. 
Please answer based ONLY on the evidence. 
The answer must be exactly one of the options: "A", "B", "C", or "D".
Do NOT explain your reasoning. Output ONLY the single letter.

Question: {question}
Evidence: {evidence}

Answer: """

def format_options(options_list):
    """Format list of options into A. ... B. ... string."""
    labels = ["A", "B", "C", "D"]
    return "\n".join([f"{labels[i]}. {opt}" for i, opt in enumerate(options_list)])

def embed_wrapper(text):
    return np.array(get_embeddings([text])[0])

def evaluate_book(raw_book_id, dataset):
    """Evaluates a single book given its ID and the loaded dataset."""
    book_id_label = f"{DATASET_NAME}_{raw_book_id}"
    cache_dir = os.path.join("cache", DATASET_NAME, raw_book_id)
    
    if raw_book_id not in dataset:
         print(f"Error: Book ID {raw_book_id} not found in dataset.")
         return

    qa_pairs = dataset[raw_book_id]["qa_pairs"]
    print(f"\n[Evaluate] Book {raw_book_id} ({len(qa_pairs)} Qs)...")

    # Skip if predictions already exist
    pred_file = os.path.join(cache_dir, "predictions.json")
    if os.path.exists(pred_file):
        print(f"  > predictions.json already exists for Book {raw_book_id}. Skipping.")
        return

    # Check cache existence
    if not os.path.exists(cache_dir):
        print(f"  > Cache dir not found: {cache_dir}. Skipping.")
        return

    # 2. INITIALIZE RETRIEVER
    retriever = load_retriever_from_cache(
        cache_dir=cache_dir,
        book_id=book_id_label,
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        embedder_func=embed_wrapper,
        spacy_model="en_core_web_lg",
        max_chunk_setting=MAX_CHUNKS,
        shortest_path_k=SHORTEST_PATH_K
    )
    
    # Start Timer
    import time
    start_time = time.time()
    
    results = []
    
    for i, qa in enumerate(qa_pairs):
        # Construct Input
        q_graph = qa["question"]
        q_dense = f"{qa['question']}\n{format_options(qa['options'])}"
        
        # RUN RETRIEVAL
        result = retriever.query(question=q_graph, full_query=q_dense)
        evidence_text = result['chunks']
        
        # RUN LLM ANSWERING
        question_with_opts = f"{qa['question']}\nOptions:\n{format_options(qa['options'])}"
        prompt = QA_PROMPT.format(question=question_with_opts, evidence=evidence_text)
        
        # Call LLM
        try:
            llm_output = call_llm(prompt, model="gpt", max_tokens=10)
        except Exception as e:
            print(f" LLM Error: {e}")
            llm_output = "Z" # Invalid answer
            
        cleaned = llm_output.strip().upper()
        final_pred = "Z"
        
        # 1. Direct Start match (e.g. "A", "A.", "A)", "Option A")
        start_match = re.match(r"^(?:OPTION\s*)?([A-D])(?:\.|:|\)|$|\s)", cleaned)
        if start_match:
            final_pred = start_match.group(1)
        else:
            # 2. Search for "Answer: X" pattern
            ans_match = re.search(r"ANSWER\s*:\s*([A-D])", cleaned)
            if ans_match:
                final_pred = ans_match.group(1)
            else:
                # 3. Last resort: Look for first isolated [A-D] 
                iso_match = re.search(r"\b([A-D])\b", cleaned)
                if iso_match:
                    final_pred = iso_match.group(1)
        
        # Store Result
        results.append({
            "question_id": i,
            "question": qa["question"],
            "options": qa["options"],
            "ground_truth": qa["answer"],
            "prediction": final_pred,
            "raw_llm_output": llm_output,
            "evidence_used": evidence_text,
            "retrieval_info": {
                "type": result.get("retrieval_type", "Unknown"),
                "entities_found": result.get("entities", []),
                "history": result.get("chunk_counts_history", [])
            }
        })
        print(f"  Q{i}: Pred={final_pred} (GT={qa['answer']}) | Mode: {result.get('retrieval_type', '?')}")

    # Save Results
    out_file = os.path.join(cache_dir, "predictions.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    # Save Time
    total_time = time.time() - start_time
    with open(os.path.join(cache_dir, "eval_time.txt"), "w") as f:
        f.write(f"{total_time:.4f}")
        
    print(f"  > Saved predictions to {out_file}")
    print(f"  > Total Eval Time: {total_time:.2f}s")

def main():
    load_dotenv()
    
    # 1. LOAD DATASET (Once)
    print(f"\n[1/2] Loading Dataset {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, DATASET_PATH)
    
    # Determines IDs to process
    if RUN_MODE == "all":
        ids = range(TOTAL_BOOKS)
    elif RUN_MODE == "range":
        ids = range(RANGE_START, RANGE_END)
    else:
        ids = [int(BOOK_ID_TO_EVAL)]
        
    print(f"\n[2/2] Running Evaluation for {len(ids)} books: {list(ids)}")
    
    for book_idx in ids:
        book_id_str = str(book_idx)
        evaluate_book(book_id_str, dataset)

if __name__ == "__main__":
    main()
