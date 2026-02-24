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
from prompts import PROMPT_CHOICE, PROMPT_OPEN

# config
DATASET_NAME = "InfiniteChoice"
DATASET_PATH = os.path.join("data", "InfiniteBench", "longbook_choice_eng.jsonl")

RUN_MODE = "range"           # "single", "range", or "all"
BOOK_IDS = range(0, 20)       # used in range mode
TARGET_BOOK_ID = 0           # used in single mode

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "testpassword"

# Options: "qwen", "gpt", "lmstudio"
LLM_MODEL_NAME = "qwen"


def get_batch_embeddings(texts):
    """Wraps get_embeddings for the retriever."""
    return np.array(get_embeddings(texts, model="bge"))


def extract_answer(llm_output):
    """
    Extract A/B/C/D from LLM output.

    Priority order:
    1. Clean single-letter answer at start: "A", "A.", "A:" with nothing after
       (what the prompt asks for). Does NOT match "A word..." to avoid treating
       the article "A" as an answer choice.
    2. Explicit label: "Answer: B", "The answer is C", "Answer is: D"
    3. Last standalone letter in the text: if the LLM reasoned verbosely before
       concluding, the answer letter is almost always at the END, not the start.
       e.g. "A detailed analysis shows the answer is C" -> last = C (correct)
    4. Fallback: "Z" (counted as wrong in metrics)
    """
    cleaned = llm_output.strip().upper()

    # 1. Pure letter answer at start, optionally with punctuation but nothing else after
    match = re.match(r"^(?:OPTION\s*)?([A-D])(?:\s*$|[.:\)]\s*$|[.:\)]\s+)", cleaned)
    if match:
        return match.group(1)

    # 2. Explicit "Answer: B" or "The answer is C" or "Answer is: D"
    match = re.search(r"(?:ANSWER\s*(?:IS\s*)?[:\-]?\s*|THE\s+ANSWER\s+IS\s+)([A-D])\b", cleaned)
    if match:
        return match.group(1)

    # 3. Last standalone A-D in the text (handles verbose reasoning then conclusion)
    all_matches = re.findall(r"\b([A-D])\b", cleaned)
    if all_matches:
        return all_matches[-1]

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
        t_retrieval = time.time()
        try:
            retrieval_res = retrieve(
                question=qa["question"],
                options=qa["options"],
                resources=resources,
            )
            context = retrieval_res.get("context", "")
        except Exception:
            traceback.print_exc()
            context = ""
            retrieval_res = {}
        retrieval_time = time.time() - t_retrieval

        # print retrieval stage counts
        s = retrieval_res.get("stats", {})
        print(f"  [Stats] region={s.get('region_chunks','?')} chunks | "
              f"graph_edges={s.get('relations_found','?')} | "
              f"after_collection={s.get('candidates_before_mmr','?')} | "
              f"after_MMR={s.get('candidates_after_mmr','?')} | "
              f"after_CrossEncoder={s.get('final_chunks','?')} | "
              f"mode={retrieval_res.get('retrieval_mode','?')} | "
              f"time={retrieval_time:.2f}s")

        # build prompt and generate answer
        # Select prompt
        full_q = qa["question"]
        prompt_template = PROMPT_CHOICE if qa["options"] else PROMPT_OPEN
        prompt = prompt_template.format(question=full_q, evidence=context)
        
        llm_response = call_llm(prompt, model=LLM_MODEL_NAME, max_tokens=1000)

        if qa["options"]:
            prediction = extract_answer(llm_response)
            ground_truth_text = qa["options"][ord(qa["answer"]) - 65] if "A" <= qa["answer"] <= "D" else ""
        else:
            prediction = llm_response.strip()
            ground_truth_text = qa["answer"]

        # store result
        results.append({
            "question_id": i,
            "question": qa["question"],
            "options": qa["options"],
            "ground_truth": qa["answer"],
            "ground_truth_text": ground_truth_text,
            "prediction": prediction,
            "raw_response": llm_response,
            "evidence": context,
            "retrieval_time": retrieval_time,
            "retrieval_stats": retrieval_res.get("stats", {}),
        })
        print(f"pred={prediction} gt={qa['answer']}")

    # save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start_time
    with open(os.path.join(cache_dir, "eval_time.txt"), "w") as f:
        f.write(f"{elapsed:.4f}")

    correct = sum(1 for r in results if r["prediction"] == r["ground_truth"])
    print(f"Book {book_idx} done: {correct}/{len(results)} correct ({elapsed:.1f}s). Saved to {output_file}")


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