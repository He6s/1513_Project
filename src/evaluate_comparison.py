import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

DATA_PATH = "data/dataset_clean.csv"
RESULTS_PATH = "results/comparison_metrics.json"

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
TEST_SIZE = 0.2


def load_data():
    print("Loading full cleaned dataset...")
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["cardio"])
    y = df["cardio"]
    print(f"Dataset shape: {df.shape}")
    return X, y


def get_models(seed):
    baseline = DummyClassifier(strategy="most_frequent")

    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1,
            penalty="l1",
            solver="saga",
            max_iter=5000,
            random_state=seed
        ))
    ])

    svm_linear = LinearSVC(
        C=1,
        penalty="l2",
        dual=False,
        random_state=seed,
        max_iter=2000
    )
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(svm_linear, cv=5))
    ])

    xgboost = XGBClassifier(
        colsample_bytree=0.8,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        n_estimators=200,
        subsample=1.0,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1
    )

    return {
        "Baseline": baseline,
        "Logistic Regression": logreg,
        "SVM (Linear)": svm,
        "XGBoost": xgboost
    }


def evaluate_over_seeds(X, y):
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]

    all_results = {
        "Baseline": {m: [] for m in metric_names},
        "Logistic Regression": {m: [] for m in metric_names},
        "SVM (Linear)": {m: [] for m in metric_names},
        "XGBoost": {m: [] for m in metric_names},
    }

    for seed in SEEDS:
        print(f"\nRunning seed {seed}...")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=y
        )

        models = get_models(seed)

        for name, model in models.items():
            print(f"  Evaluating {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            all_results[name]["Accuracy"].append(float(accuracy_score(y_test, y_pred)))
            all_results[name]["Precision"].append(float(precision_score(y_test, y_pred, zero_division=0)))
            all_results[name]["Recall"].append(float(recall_score(y_test, y_pred, zero_division=0)))
            all_results[name]["F1-Score"].append(float(f1_score(y_test, y_pred, zero_division=0)))

    summary = []

    print("\n" + "=" * 110)
    print(f"{'Model':<20} | {'Acc Mean':<10} | {'Acc Std':<10} | {'Prec Mean':<10} | {'Rec Mean':<10} | {'F1 Mean':<10}")
    print("-" * 110)

    for name, metrics in all_results.items():
        row = {
            "Model": name,
            "Accuracy Mean": float(np.mean(metrics["Accuracy"])),
            "Accuracy Std": float(np.std(metrics["Accuracy"], ddof=1)),
            "Precision Mean": float(np.mean(metrics["Precision"])),
            "Precision Std": float(np.std(metrics["Precision"], ddof=1)),
            "Recall Mean": float(np.mean(metrics["Recall"])),
            "Recall Std": float(np.std(metrics["Recall"], ddof=1)),
            "F1-Score Mean": float(np.mean(metrics["F1-Score"])),
            "F1-Score Std": float(np.std(metrics["F1-Score"], ddof=1)),
        }
        summary.append(row)

        print(
            f"{name:<20} | "
            f"{row['Accuracy Mean']:.4f}     | {row['Accuracy Std']:.4f}     | "
            f"{row['Precision Mean']:.4f}     | {row['Recall Mean']:.4f}     | "
            f"{row['F1-Score Mean']:.4f}"
        )

    print("=" * 110)
    return summary


def save_results(summary):
    Path("results").mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved comparison metrics to {RESULTS_PATH}")


if __name__ == "__main__":
    X, y = load_data()
    summary = evaluate_over_seeds(X, y)
    save_results(summary)
