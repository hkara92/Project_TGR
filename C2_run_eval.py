"""
C2_run_eval.py

Evaluates the C2 (graph-based) retrieval pipeline. For each book,
retrieves context for every question, generates an answer with the
LLM, and saves predictions to a JSON file.
"""

import os
import json
import time
import re
import traceback
import logging
import numpy as np
from dotenv import load_dotenv

from C2_retrieval import load_retriever, retrieve
from dataloader import load_dataset
from llm import get_embeddings, call_llm, unload_model

# config
DATASET_NAME = "InfiniteChoice"
DATASET_PATH = os.path.join("data", "InfiniteBench", "longbook_choice_eng.jsonl")

RUN_MODE = "range"           # "single", "range", or "all"
BOOK_IDS = range(0, 5)       # used in range mode
TARGET_BOOK_ID = 0           # used in single mode

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "testpassword"

QA_PROMPT = """You are a helpful assistant. You are given a question and evidence. 
Please answer based ONLY on the evidence. 
The answer must be exactly one of the options: "A", "B", "C", or "D".
Do NOT explain your reasoning. Output ONLY the single letter.

Question: {question}
Evidence: {evidence}

Answer:"""


def get_batch_embeddings(texts):
    """Wraps get_embeddings for the retriever."""
    return np.array(get_embeddings(texts, model="bge"))


def extract_answer(llm_output):
    """Pull out A/B/C/D from whatever the LLM returned."""
    cleaned = llm_output.strip().upper()

    # try "A", "A.", "A:" at the start
    match = re.match(r"^(?:OPTION\s*)?([A-D])(?:\.|:|\)|$|\s)", cleaned)
    if match:
        return match.group(1)

    # try "Answer: A"
    match = re.search(r"ANSWER\s*:\s*([A-D])", cleaned)
    if match:
        return match.group(1)

    # any standalone A-D
    match = re.search(r"\b([A-D])\b", cleaned)
    if match:
        return match.group(1)

    return "Z"


def evaluate_book(book_idx, dataset):
    """Run evaluation for one book."""
    book_id_str = str(book_idx)
    book_label = f"{DATASET_NAME}_{book_id_str}"
    cache_dir = os.path.join("cache", DATASET_NAME, book_id_str)

    if book_id_str not in dataset:
        print(f"Skipping book {book_idx}: not in dataset.")
        return

    if not os.path.exists(cache_dir):
        print(f"Skipping book {book_idx}: no index found (run indexing first).")
        return

    output_file = os.path.join(cache_dir, "predictions_C2.json")
    print(f"\nProcessing book {book_idx}...")

    # load retriever resources
    try:
        resources = load_retriever(
            cache_dir=cache_dir,
            book_id=book_label,
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            embedder_func=get_batch_embeddings,
            spacy_model="en_core_web_lg",
        )
    except Exception as e:
        print(f"Failed to load resources for book {book_idx}: {e}")
        return

    qa_list = dataset[book_id_str]["qa_pairs"]
    results = []
    start_time = time.time()

    for i, qa in enumerate(qa_list):
        print(f"  Q{i+1}/{len(qa_list)}...", end=" ", flush=True)

        # retrieve context
        try:
            retrieval_res = retrieve(
                question=qa["question"],
                options=qa["options"],
                resources=resources,
                verbose=False,
            )
            context = retrieval_res.get("context", "")
        except Exception:
            traceback.print_exc()
            context = ""
            retrieval_res = {}

        # build prompt and generate answer
        options_text = "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(qa["options"])])
        full_q = f"{qa['question']}\nOptions:\n{options_text}"

        prompt = QA_PROMPT.format(question=full_q, evidence=context)
        llm_response = call_llm(prompt, model="qwen", max_tokens=1000)
        prediction = extract_answer(llm_response)

        # store result
        results.append({
            "question_id": i,
            "question": qa["question"],
            "options": qa["options"],
            "ground_truth": qa["answer"],
            "ground_truth_text": qa["options"][ord(qa["answer"]) - 65] if "A" <= qa["answer"] <= "D" else "",
            "prediction": prediction,
            "raw_response": llm_response,
            "evidence": context,
            "retrieval_stats": retrieval_res.get("stats", {}),
        })
        print(f"pred={prediction} gt={qa['answer']}")

    # save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start_time
    print(f"Book {book_idx} done in {elapsed:.2f}s. Saved to {output_file}")


def main():
    load_dotenv()

    print(f"Loading dataset: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, DATASET_PATH)

    if RUN_MODE == "all":
        books_to_run = range(58)
    elif RUN_MODE == "range":
        books_to_run = BOOK_IDS
    else:
        books_to_run = [TARGET_BOOK_ID]

    print(f"Books to evaluate: {list(books_to_run)}")

    for book_id in books_to_run:
        evaluate_book(book_id, dataset)
        unload_model()


if __name__ == "__main__":
    main()
