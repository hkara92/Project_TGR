import os
import json
from rouge import Rouge
from sklearn.metrics import confusion_matrix, classification_report, f1_score

DATASET_NAME    = "InfiniteChoice"      # or "InfiniteQA"
CACHE_ROOT      = "./cache"
PREDICTION_FILE = "predictions_C2.json" # "predictions.json" for C1, "predictions_C2.json" for C2

CHOICE_LABELS = ["A", "B", "C", "D"]


def clean(text):
    return str(text).strip().upper()


def em_score(pred, truth):
    return 1.0 if clean(pred) == clean(truth) else 0.0


def rouge_l(pred, gold):
    pred, gold = clean(pred), clean(gold)
    if not pred or not gold:
        return 0.0
    try:
        return round(Rouge().get_scores(pred, gold)[0]["rouge-l"]["f"], 4)
    except:
        return 0.0


def load_predictions(cache_root, dataset_name, pred_file):
    """Walk cache folder and return all prediction records as a flat list."""
    records = []
    base = os.path.join(cache_root, dataset_name)
    for root, _, files in os.walk(base):
        for fname in files:
            if fname == pred_file:
                try:
                    records.extend(json.load(open(os.path.join(root, fname), encoding="utf-8")))
                except Exception as e:
                    print(f"Could not read {os.path.join(root, fname)}: {e}")
    return records


def calculate_accuracy(records):
    if not records:
        return 0.0, 0
    correct = sum(em_score(r.get("prediction", "Z"), r.get("ground_truth", "")) for r in records)
    return correct / len(records), len(records)


def calculate_rouge(records):
    if not records:
        return 0.0, 0
    total = sum(rouge_l(r.get("prediction", ""), r.get("ground_truth", "")) for r in records)
    return total / len(records), len(records)


def calculate_classification_metrics(records):
    """Confusion matrix, per-class P/R/F1, and macro F1 for multiple-choice."""
    y_true, y_pred, skipped = [], [], 0
    for r in records:
        gt = clean(r.get("ground_truth", ""))
        pr = clean(r.get("prediction", "Z"))
        if gt in CHOICE_LABELS and pr in CHOICE_LABELS:
            y_true.append(gt)
            y_pred.append(pr)
        else:
            skipped += 1

    if not y_true:
        return None

    return {
        "count":            len(y_true),
        "skipped":          skipped,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CHOICE_LABELS),
        "macro_f1":         f1_score(y_true, y_pred, labels=CHOICE_LABELS, average="macro"),
        "report":           classification_report(y_true, y_pred, labels=CHOICE_LABELS, digits=4),
    }


def calculate_timing(cache_root, dataset_name, pred_file):
    """Collect tree build times, eval times, and per-question retrieval times."""
    base = os.path.join(cache_root, dataset_name)
    tree_times, eval_times, retrieval_times = [], [], []

    if not os.path.exists(base):
        return tree_times, eval_times, retrieval_times

    for book_dir in sorted(os.listdir(base)):
        bp = os.path.join(base, book_dir)
        if not os.path.isdir(bp):
            continue

        for fname, lst in [("indexing_time_tree.txt", tree_times), ("eval_time.txt", eval_times)]:
            fp = os.path.join(bp, fname)
            if os.path.exists(fp):
                try:
                    lst.append(float(open(fp).read().strip()))
                except:
                    pass

        pp = os.path.join(bp, pred_file)
        if os.path.exists(pp):
            try:
                for qa in json.load(open(pp, encoding="utf-8")):
                    t = qa.get("retrieval_time")
                    if t is not None:
                        retrieval_times.append(float(t))
            except:
                pass

    return tree_times, eval_times, retrieval_times


if __name__ == "__main__":
    records = load_predictions(CACHE_ROOT, DATASET_NAME, PREDICTION_FILE)
    sep = "=" * 45

    print(f"\n{sep}")
    print(f"  RESULTS  -  {DATASET_NAME}  [{PREDICTION_FILE}]  ({len(records)} questions)")
    print(sep)

    if DATASET_NAME == "InfiniteChoice":
        acc, n = calculate_accuracy(records)
        print(f"  Accuracy (EM)  : {acc * 100:.2f}%")

        cls = calculate_classification_metrics(records)
        if cls:
            print(f"  Macro F1       : {cls['macro_f1'] * 100:.2f}%")
            if cls["skipped"]:
                print(f"  Skipped rows   : {cls['skipped']}  (prediction not in A/B/C/D)")

            print(f"\n{sep}")
            print("  CONFUSION MATRIX  (rows=ground truth, cols=predicted)")
            print(sep)
            cm = cls["confusion_matrix"]
            print("  GT / Pred   " + "     ".join(CHOICE_LABELS))
            for i, row in enumerate(cm):
                print(f"  {CHOICE_LABELS[i]}           " + "     ".join(f"{v:>4d}" for v in row))

            print(f"\n{sep}")
            print("  PER-CLASS METRICS")
            print(sep)
            print(cls["report"])

    elif DATASET_NAME == "InfiniteQA":
        rl, n = calculate_rouge(records)
        print(f"  Rouge-L (F1)   : {rl:.4f}")

    # Timing
    tree_times, eval_times, retrieval_times = calculate_timing(CACHE_ROOT, DATASET_NAME, PREDICTION_FILE)
    print(f"{sep}")
    print("  TIMING")
    print(sep)
    for label, vals in [("Tree build  (s/book)    ", tree_times),
                         ("Eval time   (s/book)    ", eval_times),
                         ("Retrieval   (s/question)", retrieval_times)]:
        if vals:
            print(f"  {label}: avg={sum(vals)/len(vals):.2f}   n={len(vals)}")
    if not any([tree_times, eval_times, retrieval_times]):
        print("  No timing data found.")
    print(sep)