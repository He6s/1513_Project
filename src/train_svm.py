import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV


# Constants
X_TRAIN_PATH = "data/splits/X_train.csv"
X_TEST_PATH = "data/splits/X_test.csv"
Y_TRAIN_PATH = "data/splits/y_train.csv"
Y_TEST_PATH = "data/splits/y_test.csv"
RESULTS_PATH = "results/svm_results.json"

def load_data():
    """Load train and test datasets."""
    print("Loading data...")
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)["target"]
    y_test = pd.read_csv(Y_TEST_PATH)["target"]
    print(f"Data loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_svm(X_train, y_train):
    """
    Train a LinearSVC model with GridSearchCV for hyperparameter tuning.
    LinearSVC is much faster than SVC for large datasets.
    Includes StandardScaler in a pipeline.
    """
    print("Training Linear SVM...")
    
    # Create a pipeline with scaling and LinearSVC
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        # dual=False is preferred when n_samples > n_features
        ('svm', LinearSVC(dual=False, random_state=42, max_iter=2000))
    ])

    # Define hyperparameters to tune
    # We tune 'C' (regularization strength) and 'penalty' (L1 or L2)
    param_grid = {
        'svm__C': [0.01, 0.1, 1, 10],
        'svm__penalty': ['l1', 'l2']
    }

    # extensive search over specified parameter values for an estimator.
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='f1', # Optimize for f1 weighted balance
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation recall: {grid_search.best_score_:.4f}")
    
    # Wrap in CalibratedClassifierCV to enable probability output if needed later
    # (Optional, but good for roc_auc_score)
    best_estimator = grid_search.best_estimator_
    calibrated_svc = CalibratedClassifierCV(best_estimator, cv='prefit')
    calibrated_svc.fit(X_train, y_train) # Fit calibration on training data
    
    return calibrated_svc, grid_search.best_params_

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set."""
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0))
    }
    
    return metrics

def save_results(metrics, best_params, filepath):
    """Save metrics and params to JSON."""
    
    output_data = {
        "model": "svm_tuned",
        "best_params": best_params,
        "metrics": metrics
    }
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"\nResults saved to {filepath}")


if __name__ == "__main__":
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_data()
    
    # 2. Train Model
    best_pipeline, best_params_dict = train_svm(X_train, y_train)
    
    # 3. Evaluate
    metrics = evaluate_model(best_pipeline, X_test, y_test)
    print("\nTest Set Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # 4. Save Results
    save_results(metrics, best_params_dict, RESULTS_PATH)
