"""
C1_run_eval.py

Evaluates the C1 retrieval pipeline. For each book, retrieves
evidence for every question, generates an answer with the LLM,
and saves predictions to a JSON file.
"""

import json
import logging
import numpy as np
from dotenv import load_dotenv
import os
import re
import time

from C1_retrieval import load_retriever_from_cache
from dataloader import load_dataset
from llm import get_embeddings, call_llm, unload_model
from prompts import PROMPT_CHOICE, PROMPT_OPEN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# config
DATASET_NAME = "InfiniteChoice"
DATASET_PATH = os.path.join("data", "InfiniteBench", "longbook_choice_eng.jsonl")

RUN_MODE = "range"       # "single", "range", or "all"
BOOK_ID_TO_EVAL = "0"    # for single mode
RANGE_START = 0           # for range mode
RANGE_END = 5
TOTAL_BOOKS = 58          # for all mode

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "testpassword"

MAX_CHUNKS = 25
SHORTEST_PATH_K = 4

MAX_CHUNKS = 25
SHORTEST_PATH_K = 4


def embed_wrapper(text):
    return np.array(get_embeddings([text], model="bge")[0])


def evaluate_book(raw_book_id, dataset):
    """Evaluate a single book."""
    book_id_label = f"{DATASET_NAME}_{raw_book_id}"
    cache_dir = os.path.join("cache", DATASET_NAME, raw_book_id)

    if raw_book_id not in dataset:
        print(f"Book {raw_book_id} not found in dataset, skipping.")
        return

    qa_pairs = dataset[raw_book_id]["qa_pairs"]
    print(f"\nEvaluating book {raw_book_id} ({len(qa_pairs)} questions)...")

    pred_file = os.path.join(cache_dir, "predictions.json")
    if os.path.exists(pred_file):
        print(f"  Predictions already exist for book {raw_book_id}, skipping.")
        return

    if not os.path.exists(cache_dir):
        print(f"  Cache not found: {cache_dir}, skipping.")
        return

    # load retriever
    retriever = load_retriever_from_cache(
        cache_dir=cache_dir,
        book_id=book_id_label,
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        embedder_func=embed_wrapper,
        spacy_model="en_core_web_lg",
        max_chunk_setting=MAX_CHUNKS,
        shortest_path_k=SHORTEST_PATH_K,
    )

    start_time = time.time()
    results = []

    for i, qa in enumerate(qa_pairs):
        q_graph = qa["question"]
        q_dense = qa["question"]

        # retrieve evidence
        result = retriever.query(question=q_graph, full_query=q_dense)
        evidence_text = result["chunks"]

        # Select prompt based on whether we have options
        # Select prompt
        prompt_template = PROMPT_CHOICE if qa["options"] else PROMPT_OPEN
        final_prompt = prompt_template.format(question=qa["question"], evidence=evidence_text)

        try:
            llm_output = call_llm(final_prompt, model="qwen", max_tokens=1000)
        except Exception as e:
            print(f"  LLM error: {e}")
            llm_output = "Z"

        if qa["options"]:
            # extract the predicted letter
            cleaned = llm_output.strip().upper()
            final_pred = "Z"

            match = re.match(r"^(?:OPTION\s*)?([A-D])(?:\.|:|\)|$|\s)", cleaned)
            if match:
                final_pred = match.group(1)
            else:
                match = re.search(r"ANSWER\s*:\s*([A-D])", cleaned)
                if match:
                    final_pred = match.group(1)
                else:
                    match = re.search(r"\b([A-D])\b", cleaned)
                    if match:
                        final_pred = match.group(1)
        else:
            # For open QA, just take the raw output
            final_pred = llm_output.strip()

        # ground truth text
        labels = ["A", "B", "C", "D"]
        gt_idx = labels.index(qa["answer"]) if qa["answer"] in labels else -1
        ground_truth_text = qa["options"][gt_idx] if 0 <= gt_idx < len(qa["options"]) else str(qa["answer"])

        results.append({
            "question_id": i,
            "question": qa["question"],
            "options": qa["options"],
            "ground_truth": qa["answer"],
            "ground_truth_text": ground_truth_text,
            "prediction": final_pred,
            "raw_llm_output": llm_output,
            "evidence_used": evidence_text,
            "retrieval_info": {
                "type": result.get("retrieval_type", "unknown"),
                "entities_found": result.get("entities", []),
                "chunk_ids": result.get("chunk_ids", {}),
                "history": result.get("chunk_counts_history", []),
            },
        })
        print(f"  Q{i}: pred={final_pred} gt={qa['answer']} mode={result.get('retrieval_type', '?')}")

    # save predictions
    out_file = os.path.join(cache_dir, "predictions.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - start_time
    with open(os.path.join(cache_dir, "eval_time.txt"), "w") as f:
        f.write(f"{total_time:.4f}")

    print(f"  Saved to {out_file} ({total_time:.2f}s)")


def main():
    load_dotenv()

    print(f"Loading dataset: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, DATASET_PATH)

    if RUN_MODE == "all":
        ids = range(TOTAL_BOOKS)
    elif RUN_MODE == "range":
        ids = range(RANGE_START, RANGE_END)
    else:
        ids = [int(BOOK_ID_TO_EVAL)]

    print(f"Evaluating {len(ids)} books: {list(ids)}")

    for book_idx in ids:
        evaluate_book(str(book_idx), dataset)
        unload_model()


if __name__ == "__main__":
    main()
