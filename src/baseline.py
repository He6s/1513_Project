import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


X_TRAIN_PATH = "data/splits/X_train.csv"
X_TEST_PATH = "data/splits/X_test.csv"
Y_TRAIN_PATH = "data/splits/y_train.csv"
Y_TEST_PATH = "data/splits/y_test.csv"

RESULTS_DIR = "results"
RESULTS_PATH = "results/baseline_results.json"


def load_split_data():
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)["target"]
    y_test = pd.read_csv(Y_TEST_PATH)["target"]
    return X_train, X_test, y_train, y_test


def evaluate_baseline(X_train, X_test, y_train, y_test):
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = {
        "model": "naive_baseline_most_frequent",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0))
    }

    return results


def save_results(results, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split_data()
    results = evaluate_baseline(X_train, X_test, y_train, y_test)

    print("Baseline results:")
    for key, value in results.items():
        print(f"{key}: {value}")

    save_results(results, RESULTS_PATH)
    print(f"\nSaved baseline results to: {RESULTS_PATH}")
