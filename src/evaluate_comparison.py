import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)

# Constants
X_TRAIN_PATH = "data/splits/X_train.csv"
X_TEST_PATH = "data/splits/X_test.csv"
Y_TRAIN_PATH = "data/splits/y_train.csv"
Y_TEST_PATH = "data/splits/y_test.csv"
RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"

def load_data():
    """Load train and test datasets."""
    print("Loading data...")
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)["target"]
    y_test = pd.read_csv(Y_TEST_PATH)["target"]
    return X_train, X_test, y_train, y_test

def get_best_models():
    """Define models with best hyperparameters found in validation."""
    
    # 1. Baseline
    baseline = DummyClassifier(strategy="most_frequent")
    
    # 2. Logistic Regression
    # Best: C=1, penalty='l1'
    logreg = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1, penalty='l1', solver='saga', max_iter=5000, random_state=42))
    ])
    
    # 3. SVM (LinearSVC wrapped in CalibratedClassifierCV)
    # Best: C=1, penalty='l2'
    # Wrap in CalibratedClassifierCV to get predict_proba
    svm_linear = LinearSVC(C=1, penalty='l2', dual=False, random_state=42, max_iter=2000)
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', CalibratedClassifierCV(svm_linear, cv=5)) 
    ])
    
    # 4. XGBoost
    # Best: colsample_bytree=0.8, learning_rate=0.05, max_depth=4, min_child_weight=3, n_estimators=200, subsample=1.0
    xgboost = XGBClassifier(
        colsample_bytree=0.8,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        n_estimators=200,
        subsample=1.0,
        eval_metric='logloss',
        random_state=42
    )

    return {
        "Baseline": baseline,
        "Logistic Regression": logreg,
        "SVM (Linear)": svm,
        "XGBoost": xgboost
    }

def evaluate_and_plot(models, X_train, X_test, y_train, y_test):
    """Train models, compute metrics, and generate plots."""
    
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    
    metrics_summary = []
    roc_data = [] # Store FPR, TPR, AUC, label for final plot

    for name, model in models.items():
        print(f"Evaluating {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0)
        }
        metrics_summary.append(metrics)
        
        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
        
        plt.figure()
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix: {name}")
        plt.savefig(f"{PLOTS_DIR}/cm_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png")
        plt.close() 
        
        # ROC Data Collection
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            roc_data.append({"fpr": fpr, "tpr": tpr, "auc": roc_auc, "name": name})
        else:
             print(f"Warning: {name} does not support predict_proba, skipping ROC curve.")

    # Combined ROC Plot
    plt.figure(figsize=(10, 8))
    for item in roc_data:
        plt.plot(item['fpr'], item['tpr'], label=f"{item['name']} (AUC = {item['auc']:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label="Chance (AUC = 0.50)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(f"{PLOTS_DIR}/roc_comparison.png")
    plt.close()
    
    return metrics_summary


def save_summary(metrics_summary):
    """Save metrics to JSON."""
    filepath = f"{RESULTS_DIR}/comparison_metrics.json"
    with open(filepath, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"\nSummary metrics saved to {filepath}")
    
    # Also print as text table
    print("\n" + "="*60)
    print(f"{'Model':<20} | {'acc':<8} | {'prec':<8} | {'rec':<8} | {'f1':<8}")
    print("-" * 60)
    for m in metrics_summary:
        print(f"{m['Model']:<20} | {m['Accuracy']:.4f}   | {m['Precision']:.4f}   | {m['Recall']:.4f}   | {m['F1-Score']:.4f}")
    print("="*60)


if __name__ == "__main__":
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_data()
    
    # 2. Get Models
    models = get_best_models()
    
    # 3. Evaluate and Plot
    metrics_summary = evaluate_and_plot(models, X_train, X_test, y_train, y_test)
    
    # 4. Save Results
    save_summary(metrics_summary)
    print(f"\nPlots saved to {PLOTS_DIR}/")
