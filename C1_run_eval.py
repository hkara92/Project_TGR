"""
Evaluates the C1 (baseline) retrieval pipeline. For each book, retrieves
evidence for every question, generates an answer with the LLM,
and saves predictions to a JSON file.
"""

import os
import re
import json
import time
import logging
import numpy as np
from dotenv import load_dotenv

from C1_retrieval import load_retriever_from_cache
from dataloader import load_dataset
from llm import get_embeddings, call_llm, unload_model, preload_models
from prompts import PROMPT_CHOICE, PROMPT_OPEN

logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Main settings
DATASET_NAME = "InfiniteChoice"
DATASET_PATH = os.path.join("data", "InfiniteBench", "longbook_choice_eng.jsonl")

RUN_MODE = "all"          # "single", "range", or "all"
BOOK_ID_TO_EVAL = "0"     # for single mode
RANGE_START = 0            # for range mode
RANGE_END = 51
TOTAL_BOOKS = 69           # for all mode

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "testpassword"

# Options: "qwen", "gpt", "lmstudio"
LLM_MODEL_NAME = "lmstudio"

MAX_CHUNKS = 25
SHORTEST_PATH_K = 4


def embed_wrapper(text):
    return np.array(get_embeddings([text], model="bge")[0])


def extract_answer(llm_output):
    """
    Extracts the correct A, B, C, or D answer from the LLM's output using regex.
    Falls back to "Z" (incorrect) if no valid letter is found.
    """
    cleaned = llm_output.strip().upper()

    # First check: clean single letter at the start
    match = re.match(r"^(?:OPTION\s*)?([A-D])(?:\.|:|\)|$|\s)", cleaned)
    if match:
        return match.group(1)

    # Second check: explicit "Answer: B" pattern
    match = re.search(r"ANSWER\s*:\s*([A-D])", cleaned)
    if match:
        return match.group(1)

    # Third check: any standalone A-D letter
    match = re.search(r"\b([A-D])\b", cleaned)
    if match:
        return match.group(1)

    return "Z"


def evaluate_book(raw_book_id, dataset):
    """Runs the full C1 retrieval and generation pipeline for a single book."""
    book_id_label = f"{DATASET_NAME}_{raw_book_id}"
    cache_dir = os.path.join("cache", DATASET_NAME, raw_book_id)

    if raw_book_id not in dataset:
        print(f"Book {raw_book_id} not found in dataset, skipping.")
        return

    qa_pairs = dataset[raw_book_id]["qa_pairs"]
    print(f"\n--- Book {raw_book_id} ({len(qa_pairs)} questions) ---")

    if not os.path.exists(cache_dir):
        print(f"  Cache not found: {cache_dir}, skipping.")
        return

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

        # Retrieve evidence chunks
        print(f"  Q{i}/{len(qa_pairs)-1}: Retrieving...")
        t_start_retrieval = time.time()
        result = retriever.query(question=q_graph, full_query=q_dense)
        retrieval_time = time.time() - t_start_retrieval
        evidence_text = result["chunks"]
        rtype = result.get('retrieval_type', '?')
        n_chunks = result.get('len_chunks', 0)
        print(f"  Q{i}/{len(qa_pairs)-1}: {n_chunks} chunks retrieved via [{rtype}] (qtime={retrieval_time:.2f}s)")

        # Generate answer with the LLM
        print(f"  Q{i}/{len(qa_pairs)-1}: LLM generating answer...")
        prompt_template = PROMPT_CHOICE if qa["options"] else PROMPT_OPEN
        final_prompt = prompt_template.format(question=qa["question"], evidence=evidence_text)

        try:
            llm_output = call_llm(final_prompt, model=LLM_MODEL_NAME, max_tokens=1000)
        except Exception as e:
            print(f"  Q{i}/{len(qa_pairs)-1}: [ERROR] LLM failed: {e}")
            llm_output = "Z"

        # Extract the final prediction
        if qa["options"]:
            final_pred = extract_answer(llm_output)
        else:
            final_pred = llm_output.strip()

        # Log the result
        labels = ["A", "B", "C", "D"]
        gt_idx = labels.index(qa["answer"]) if qa["answer"] in labels else -1
        ground_truth_text = qa["options"][gt_idx] if 0 <= gt_idx < len(qa["options"]) else str(qa["answer"])
        status = "OK" if final_pred == qa["answer"] else "WRONG"
        print(f"  Q{i}/{len(qa_pairs)-1}: pred={final_pred} gt={qa['answer']} [{status}]")

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
            "retrieval_time": retrieval_time,
        })

    # Save predictions to disk
    out_file = os.path.join(cache_dir, "predictions.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - start_time
    with open(os.path.join(cache_dir, "eval_time.txt"), "w") as f:
        f.write(f"{total_time:.4f}")

    correct = sum(1 for r in results if r["prediction"] == r["ground_truth"])
    print(f"  Done: {correct}/{len(results)} correct ({total_time:.1f}s)")


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

    print("\nPre-loading models...")
    preload_models(llm_model=LLM_MODEL_NAME)
    print()

    for idx, book_idx in enumerate(ids):
        evaluate_book(str(book_idx), dataset)
        unload_model()


if __name__ == "__main__":
    main()
