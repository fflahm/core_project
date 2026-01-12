import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    classification_report
)
from analysis_module import CognitiveAnalyzer

DATASET_FILES = [
    "data/dataset_v1_definition.json",
    "data/dataset_v3_persona.json"
]

THRESHOLD = 0.5 

def safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except:
        return None

def compute_metrics(y_true, y_pred, y_score):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = safe_auc(y_true, y_score)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": None if auc is None else float(auc),
        "n_samples": int(len(y_true)),
        "positive": int(np.sum(y_true)),
        "negative": int(len(y_true) - np.sum(y_true))
    }

def evaluate_dataset(path, analyzer):
    dataset = json.load(open(path, "r", encoding="utf-8"))
    print(f"\n Evaluating {path} ({len(dataset)} samples)")

    y_true, y_pred, y_score = [], [], []
    predictions, errors = [], []

    groups = {
        "domain": defaultdict(list),
        "persona": defaultdict(list),
        "pattern": defaultdict(list)
    }

    for sample in tqdm(dataset, desc="Evaluating"):
        text = sample["text"]
        gold = int(sample["gold_label"])

        features = analyzer.analyze_text(text)
        is_rum, conf, reasoning = analyzer.detect_rumination(
            features, threshold=THRESHOLD
        )
        pred = int(is_rum)

        record = {
            "id": sample["id"],
            "gold": gold,
            "pred": pred,
            "confidence": conf,
            "reasoning": reasoning,
            "analysis": features,
            "meta": {
                "domain": sample.get("domain"),
                "persona": sample.get("persona"),
                "pattern_id": sample.get("pattern_id"),
                "method": sample.get("method")
            }
        }

        predictions.append(record)

        y_true.append(gold)
        y_pred.append(pred)
        y_score.append(conf)

        if pred != gold:
            errors.append({**record, "text": text, "question": sample.get("question")})

        if sample.get("domain"):
            groups["domain"][sample["domain"]].append(record)
        if sample.get("persona"):
            groups["persona"][sample["persona"]].append(record)
        if sample.get("pattern_id"):
            groups["pattern"][sample["pattern_id"]].append(record)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    overall = compute_metrics(y_true, y_pred, y_score)

    print("\n--- Overall ---")
    print(json.dumps(overall, indent=2))
    print(classification_report(
        y_true, y_pred,
        target_names=["Non-Rumination", "Rumination"],
        zero_division=0
    ))

    group_metrics = {}
    for gname, buckets in groups.items():
        group_metrics[gname] = {}
        for key, recs in buckets.items():
            if len(recs) < 3:
                continue
            yt = np.array([r["gold"] for r in recs])
            yp = np.array([r["pred"] for r in recs])
            ys = np.array([r["confidence"] for r in recs])
            group_metrics[gname][key] = compute_metrics(yt, yp, ys)

    prefix = path.replace(".json", "")
    json.dump(predictions, open(f"results/{prefix}_predictions.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(errors, open(f"results/{prefix}_errors.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(overall, open(f"results/{prefix}_summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(group_metrics, open(f"results/{prefix}_groups.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"Saved: results/{prefix}_*.json")

def main():
    analyzer = CognitiveAnalyzer()
    for path in DATASET_FILES:
        evaluate_dataset(path, analyzer)

if __name__ == "__main__":
    main()
