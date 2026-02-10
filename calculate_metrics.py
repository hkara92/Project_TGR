import os
import json
import logging
from typing import Tuple
from rouge import Rouge

# --- CONFIGURATION ---
# Options: "InfiniteChoice" (Multiple Choice), "InfiniteQA" (Free Generation)
DATASET_NAME = "InfiniteChoice" 
CACHE_ROOT = "./cache"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def standardize_answer(text):
    return str(text).strip().upper()

def EM_score(pred, truth):
    return 1.0 if standardize_answer(pred) == standardize_answer(truth) else 0.0

def RL_score(pred, gold):
    """Calculate Rouge-L score using rouge library"""
    pred = standardize_answer(pred)
    gold = standardize_answer(gold)
    if not pred or not gold: return 0.0
    
    rouge = Rouge()
    try:
        scores = rouge.get_scores(pred, gold)[0]
        return round(scores['rouge-l']['f'], 4)
    except:
        return 0.0

def calculate_metrics(answer_folder: str, dataset_name: str) -> Tuple[float, float, int]:
    """
    Calculate metrics based on dataset type.
    """
    base_path = os.path.join(answer_folder, dataset_name)
    logger.info(f"Scanning for predictions in: {base_path}")
    
    total_em_score = 0
    total_rl_score = 0
    total_num = 0
    
    if not os.path.exists(base_path):
        logger.error(f"Folder not found: {base_path}")
        return 0.0, 0.0, 0

    for root, _, files in os.walk(base_path):
        for file in files:
            if file in ["predictions.json", "predictions_C2.json"]:
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        answer_list = json.load(f)
                        if len(answer_list) > 0:
                            book_id = os.path.basename(root)
                            print(f"[Metrics] Processing Book ID: {book_id} ({len(answer_list)} qs)")
                            
                except Exception as e:
                    logger.error(f"Error reading {full_path}: {e}")
                    continue
                
                for qa in answer_list:
                    pred = qa.get("prediction", "Z")
                    truth = qa.get("ground_truth", "")
                    
                    if dataset_name == "InfiniteChoice":
                        # Multiple Choice
                        em = EM_score(pred, truth)
                        total_em_score += em
                        total_rl_score += em 
                    elif dataset_name in ["InfiniteQA", "InfiniteQALoader", "NarrativeQA"]:
                        # Free Text
                        rl = RL_score(pred, truth)
                        total_rl_score += rl
                        total_em_score += EM_score(pred, truth)
                    else:
                        total_em_score += EM_score(pred, truth)
                        total_rl_score += RL_score(pred, truth)
                    
                    total_num += 1

    if total_num == 0:
        return 0.0, 0.0, 0
        
    return total_em_score / total_num, total_rl_score / total_num, total_num

if __name__ == "__main__":
    em, rl, count = calculate_metrics(CACHE_ROOT, DATASET_NAME)

    print("\n" + "="*40)
    print(f"EVALUATION RESULTS ({DATASET_NAME})")
    print("="*40)
    print(f"Total Questions: {count}")
    
    if DATASET_NAME == "InfiniteChoice":
        print(f"Accuracy (EM):  {em*100:.2f}%")
    elif DATASET_NAME in ["InfiniteQA"]:
         print(f"Rouge-L (F1):   {rl:.4f}")
