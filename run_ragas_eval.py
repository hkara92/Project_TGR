"""
Evaluates predictions from C1/C2 pipelines using the RAGAS framework.
Measures faithfulness, answer relevancy, context recall/precision, and answer correctness.
"""

import os
import json
import logging
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
)

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.run_config import RunConfig
import nest_asyncio

# Main settings
DATASET_NAME = "InfiniteChoice"
CACHE_DIR = "./cache"
PREDICTION_FILE = "predictions_C2.json"
TOTAL_BOOKS = 58

# LM Studio serves an OpenAI-compatible API on this port
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
LM_STUDIO_MODEL = "lm-studio"

load_dotenv()
nest_asyncio.apply()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_book_ragas(book_id):
    """Runs the full suite of RAGAS metrics on a single book's predictions."""

    # Load the predictions we generated earlier
    pred_path = os.path.join(CACHE_DIR, DATASET_NAME, book_id, PREDICTION_FILE)
    if not os.path.exists(pred_path):
        logger.error(f"Prediction file not found: {pred_path}")
        return

    logger.info(f"Loading predictions from {pred_path}...")
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        logger.warning("Empty predictions file.")
        return

    # Pack the data into the HuggingFace Dataset format that RAGAS expects
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in data:
        questions.append(item["question"])

        # Use raw LLM output (key differs between C1 and C2 predictions)
        ans_text = str(item.get("raw_response", item.get("raw_llm_output", "")))
        answers.append(ans_text)

        # C2 stores evidence as a single text block, C1 stores chunk_ids that need lookup
        evidence = item.get("evidence", "")
        if evidence and isinstance(evidence, str):
            ctx_list = [c.strip() for c in evidence.split("\n\n") if c.strip()]
        else:
            ctx_list = []
            nodes_path = os.path.join(CACHE_DIR, DATASET_NAME, book_id, "summary_tree", "nodes.json")
            if os.path.exists(nodes_path):
                try:
                    with open(nodes_path, "r", encoding="utf-8") as f2:
                        nodes_map = json.load(f2)
                    ret_info = item.get("retrieval_info", {})
                    chunk_ids_dict = ret_info.get("chunk_ids", {})
                    flat_ids = set()
                    for c_list in chunk_ids_dict.values():
                        flat_ids.update(c_list)
                    for cid in flat_ids:
                        text = None
                        if cid in nodes_map:
                            text = nodes_map[cid].get("text")
                        elif f"L0_{cid}" in nodes_map:
                            text = nodes_map[f"L0_{cid}"].get("text")
                        if text:
                            ctx_list.append(text)
                except Exception:
                    pass

        if not ctx_list:
            ctx_list = ["No context available."]

        contexts.append(ctx_list)
        ground_truths.append(item.get("ground_truth_text", item.get("ground_truth", "")))

    if not questions:
        logger.warning("No valid data found.")
        return

    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # Set up LLM and embeddings via LM Studio's OpenAI-compatible endpoint
    llm = ChatOpenAI(
        model=LM_STUDIO_MODEL,
        base_url=LM_STUDIO_BASE_URL,
        api_key=LM_STUDIO_API_KEY,
        temperature=0,
    )
    embeddings = OpenAIEmbeddings(
        model="text-embedding-nomic-embed-text-v1.5",
        base_url=LM_STUDIO_BASE_URL,
        api_key=LM_STUDIO_API_KEY,
    )

    # Run the evaluation
    logger.info(f"Running RAGAS evaluation for book {book_id}...")

    metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
        answer_correctness,
    ]

    run_config = RunConfig(timeout=600, max_workers=1, max_wait=300)

    try:
        results = evaluate(
            ragas_dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
            run_config=run_config,
        )
    except Exception as e:
        logger.error(f"RAGAS evaluation failed for book {book_id}: {e}")
        return

    # Save results
    output_dir = os.path.join(CACHE_DIR, DATASET_NAME, book_id)
    out_csv = os.path.join(output_dir, "ragas_results.csv")
    out_txt = os.path.join(output_dir, "ragas_summary.txt")

    df = results.to_pandas()
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved RAGAS details to {out_csv}")

    def safe_mean(col):
        return f"{df[col].mean():.4f}" if col in df.columns else "N/A"

    summary = f"""
    RAGAS Results for Book {book_id}
    ================================
    Faithfulness:        {safe_mean('faithfulness')}
    Answer Relevancy:    {safe_mean('answer_relevancy')}
    Context Recall:      {safe_mean('context_recall')}
    Context Precision:   {safe_mean('context_precision')}
    Answer Correctness:  {safe_mean('answer_correctness')}
    """
    print(summary)

    with open(out_txt, "w") as f:
        f.write(summary)


if __name__ == "__main__":
    for i in range(TOTAL_BOOKS):
        evaluate_book_ragas(str(i))
