import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_validate, StratifiedKFold
from xgboost import XGBClassifier

# Constants
X_TRAIN_PATH = "data/splits/X_train.csv"
Y_TRAIN_PATH = "data/splits/y_train.csv"
RESULTS_PATH = "results/cv_results.json"

def load_train_data():
    print("Loading training data...")
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)["target"]
    return X_train, y_train

def get_models():
    """Return models with best parameters."""
    
    # 1. Logistic Regression
    # Best: C=1, penalty='l1'
    logreg = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1, penalty='l1', solver='saga', max_iter=5000, random_state=42))
    ])

    # 2. Linear SVM (Calibrated)
    # Best: C=1, penalty='l2'
    svm_linear = LinearSVC(C=1, penalty='l2', dual=False, random_state=42, max_iter=2000)
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', CalibratedClassifierCV(svm_linear, cv=5)) 
    ])

    # 3. XGBoost
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
        "Logistic Regression": logreg,
        "SVM (Linear)": svm,
        "XGBoost": xgboost
    }

def run_cv_evaluation(X, y, models):
    """Run 5-Fold Cross-Validation."""
    results = {}
    
    # Stratified K-Fold ensures class distribution is preserved
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = ['accuracy', 'precision', 'recall', 'f1']

    print("\nRunning 5-Fold Cross-Validation...")
    print(f"{'Model':<20} | {'Metric':<10} | {'Mean':<8} | {'Std Dev':<8}")
    print("-" * 60)

    for name, model in models.items():
        # cross_validate returns a dict with 'test_score', 'fit_time', etc.
        # But since we passed a list of metrics, it returns 'test_accuracy', 'test_precision', etc.
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        model_results = {}
        for metric in scoring:
            key = f"test_{metric}"
            mean_score = scores[key].mean()
            std_score = scores[key].std()
            
            model_results[metric] = {
                "mean": mean_score,
                "std": std_score
            }
            
            print(f"{name:<20} | {metric:<10} | {mean_score:.4f}   | {std_score:.4f}")
        
        results[name] = model_results
        print("-" * 60)
            
    return results

def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nCV Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    X_train, y_train = load_train_data()
    models = get_models()
    results = run_cv_evaluation(X_train, y_train, models)
    save_results(results)