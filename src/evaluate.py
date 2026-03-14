import json
from pathlib import Path
from typing import Dict

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_classification_metrics(y_true, y_pred) -> Dict[str, float]:
    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0))
    }
    return results


def save_results(results: Dict[str, float], filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]

    results = compute_classification_metrics(y_true, y_pred)

    print("Test metrics:")
    for key, value in results.items():
        print(f"{key}: {value}")

    save_results(results, "results/evaluate_test_results.json")
    print("\nSaved test results to: results/evaluate_test_results.json")
