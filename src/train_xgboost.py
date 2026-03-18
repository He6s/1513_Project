import json
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Constants
X_TRAIN_PATH = "data/splits/X_train.csv"
X_TEST_PATH = "data/splits/X_test.csv"
Y_TRAIN_PATH = "data/splits/y_train.csv"
Y_TEST_PATH = "data/splits/y_test.csv"
RESULTS_PATH = "results/xgboost_results.json"

def load_data():
    """Load train and test datasets."""
    print("Loading data...")
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)["target"]
    y_test = pd.read_csv(Y_TEST_PATH)["target"]
    print(f"Data loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_xgboost(X_train, y_train):
    """
    Train an XGBoost model with GridSearchCV for hyperparameter tuning.
    Note: Tree-based models like XGBoost generally don't require feature scaling.
    """
    print("Training XGBoost...")
    
    # Initialize XGBClassifier
    # eval_metric='logloss' is standard for binary classification
    xgb = XGBClassifier(
        eval_metric='logloss', 
        random_state=42
    )

    # Define hyperparameters to tune
    # n_estimators: Number of gradient boosted trees.
    # max_depth: Maximum tree depth for base learners.
    # learning_rate: Boosting learning rate (eta).
    # subsample: Subsample ratio of the training instances (prevents overfitting).
    # colsample_bytree: Subsample ratio of columns when constructing each tree (good for correlated features).
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4],
        'learning_rate': [0.03, 0.05],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Grid search optimization
    grid_search = GridSearchCV(
        xgb, 
        param_grid, 
        cv=5, 
        scoring='f1', # Optimize for f1 weighted balance
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


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

def interpret_model(model, feature_names):
    """Extract and print top feature importances from XGBoost."""
    importances = model.feature_importances_
    
    # Create a dataframe for visualization
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance value
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head(5))
    
    return feature_importance

def save_results(metrics, feature_importance, best_params, filepath):
    """Save metrics, top features, and params to JSON."""
    
    # Convert feature importance to dict
    top_features = feature_importance[['feature', 'importance']].head(10).to_dict(orient='records')
    
    # Convert numpy floats to native python floats for JSON serialization if needed
    # (Though logic above uses float() casting already)
    
    output_data = {
        "model": "xgboost_tuned",
        "best_params": best_params,
        "metrics": metrics,
        "top_features": top_features
    }
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"\nResults saved to {filepath}")

if __name__ == "__main__":
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_data()
    
    # 2. Train Model
    best_model = train_xgboost(X_train, y_train)
    
    # 3. Evaluate
    metrics = evaluate_model(best_model, X_test, y_test)
    print("\nTest Set Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # 4. Interpret
    feature_names = X_train.columns.tolist()
    feature_importance = interpret_model(best_model, feature_names)
    
    # 5. Save Results
    best_params = best_model.get_params()
    # Filter only the relevant tuned params for clarity in JSON
    relevant_params = {k: best_params[k] for k in ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree']}
    
    save_results(metrics, feature_importance, relevant_params, RESULTS_PATH)
