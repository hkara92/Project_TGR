"""
C2_run_eval.py
Evaluation runner for the C2 (Region-Restricted Graph) Retrieval Pipeline.
"""

import os
import json
import time
import re
import traceback
import logging
import numpy as np
from dotenv import load_dotenv

# Local Imports
from C2_retrieval import load_retriever, retrieve
from dataloader import load_dataset
from llm import get_embeddings, call_llm

# --- Configuration ---
DATASET_NAME = "InfiniteChoice"
DATASET_PATH = os.path.join("data", "InfiniteBench", "longbook_choice_eng.jsonl")

# Run Mode: 'single', 'range', or 'all'
RUN_MODE = "range"
BOOK_IDS = range(0, 5)  # For 'range' mode (0 to 9)
TARGET_BOOK_ID = 0       # For 'single' mode

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "testpassword"

# QA Prompt
QA_PROMPT = """You are a helpful assistant. You are given a question and evidence. 
Please answer based ONLY on the evidence. 
The answer must be exactly one of the options: "A", "B", "C", or "D".
Do NOT explain your reasoning. Output ONLY the single letter.

Question: {question}
Evidence: {evidence}

Answer:"""


def get_batch_embeddings(texts):
    """Adapter for C2 retriever to use correct embedding model."""
    return np.array(get_embeddings(texts, model="text-embedding-3-large"))


def extract_answer(llm_output):
    """Extracts A, B, C, or D from LLM response."""
    cleaned = llm_output.strip().upper()
    
    # Priority 1: Starts with letter (e.g., "A", "A.", "A: ")
    match = re.match(r"^(?:OPTION\s*)?([A-D])(?:\.|:|\)|$|\s)", cleaned)
    if match:
        return match.group(1)
        
    # Priority 2: "Answer: A" format
    match = re.search(r"ANSWER\s*:\s*([A-D])", cleaned)
    if match:
        return match.group(1)
        
    # Priority 3: Any standalone letter
    match = re.search(r"\b([A-D])\b", cleaned)
    if match:
        return match.group(1)
        
    return "Z"  # Invalid/Unknown


def evaluate_book(book_idx, dataset):
    """Runs evaluation for a single book."""
    book_id_str = str(book_idx)
    book_label = f"{DATASET_NAME}_{book_id_str}"
    cache_dir = os.path.join("cache", DATASET_NAME, book_id_str)
    
    # Skip checks
    if book_id_str not in dataset:
        print(f"Skipping Book {book_idx}: Not in dataset.")
        return

    output_file = os.path.join(cache_dir, "predictions_C2.json")
    # output_file check removed to force overwrite
    # if os.path.exists(output_file):
    #    print(f"Skipping Book {book_idx}: Predictions exist.")
    #    return

    if not os.path.exists(cache_dir):
        print(f"Skipping Book {book_idx}: No index found (run indexing first).")
        return

    print(f"\nProcessing Book {book_idx}...")

    # Load Resources
    try:
        resources = load_retriever(
            cache_dir=cache_dir,
            book_id=book_label,
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            embedder_func=get_batch_embeddings,
            spacy_model="en_core_web_lg"
        )
    except Exception as e:
        print(f"Failed to load resources for Book {book_idx}: {e}")
        return

    qa_list = dataset[book_id_str]["qa_pairs"]
    results = []
    start_time = time.time()

    for i, qa in enumerate(qa_list):
        print(f"  Q{i+1}/{len(qa_list)}...", end=" ", flush=True)
        
        # 1. Retrieve
        try:
            retrieval_res = retrieve(
                question=qa["question"],
                options=qa["options"],
                resources=resources,
                verbose=False
            )
            context = retrieval_res.get("context", "")
        except Exception:
            traceback.print_exc()
            context = ""
            retrieval_res = {}

        # 2. Generate Answer
        options_text = "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(qa["options"])])
        full_q = f"{qa['question']}\nOptions:\n{options_text}"
        
        prompt = QA_PROMPT.format(question=full_q, evidence=context)
        llm_response = call_llm(prompt, model="gpt", max_tokens=1000)
        prediction = extract_answer(llm_response)

        # 3. Store
        results.append({
            "question_id": i,
            "question": qa["question"],
            "options": qa["options"],
            "ground_truth": qa["answer"],
            "ground_truth_text": qa["options"][ord(qa["answer"]) - 65] if "A" <= qa["answer"] <= "D" else "",
            "prediction": prediction,
            "raw_response": llm_response,
            "evidence": context,
            "retrieval_stats": retrieval_res.get("stats", {})
        })
        print(f"Pred: {prediction} | GT: {qa['answer']}")

    # Save Results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start_time
    print(f"Finished Book {book_idx} in {elapsed:.2f}s. Saved to {output_file}")


def main():
    load_dotenv()
    
    print(f"Loading Dataset: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, DATASET_PATH)
    
    # Select Books
    if RUN_MODE == "all":
        # Assuming max 58 books based on known dataset size
        books_to_run = range(58)
    elif RUN_MODE == "range":
        books_to_run = BOOK_IDS
    else:
        books_to_run = [TARGET_BOOK_ID]

    print(f"Target Books: {list(books_to_run)}")
    
    for book_id in books_to_run:
        evaluate_book(book_id, dataset)

if __name__ == "__main__":
    main()
