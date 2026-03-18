import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Constants
X_TRAIN_PATH = "data/splits/X_train.csv"
X_TEST_PATH = "data/splits/X_test.csv"
Y_TRAIN_PATH = "data/splits/y_train.csv"
Y_TEST_PATH = "data/splits/y_test.csv"
RESULTS_PATH = "results/logreg_results.json"

def load_data():
    """Load train and test datasets."""
    print("Loading data...")
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)["target"]
    y_test = pd.read_csv(Y_TEST_PATH)["target"]
    print(f"Data loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_logreg(X_train, y_train):
    """
    Train a Logistic Regression model with GridSearchCV for hyperparameter tuning.
    Includes StandardScaler in a pipeline.
    """
    print("Training Logistic Regression...")
    
    # Create a pipeline with scaling and logistic regression
    # Scaling is important for regularization to work correctly
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        # Increased max_iter to ensure convergence with interaction terms
        ('logreg', LogisticRegression(solver='saga', max_iter=5000, random_state=42)) 
    ])

    # Define hyperparameters to tune
    # C is the inverse of regularization strength
    param_grid = {
        'logreg__C': [0.01, 0.1, 1, 10],
        'logreg__penalty': ['l1', 'l2']
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
    """Extract and print top coefficients."""
    # Access the logistic regression step from the pipeline
    logreg = model.named_steps['logreg']
    coefs = logreg.coef_[0]
    
    # Create a dataframe for visualization
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefs,
        'abs_coefficient': np.abs(coefs)
    })
    
    # Sort by absolute coefficient value
    feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
    
    print("\nTop 5 Most Influential Features:")
    print(feature_importance[['feature', 'coefficient']].head(5))
    
    return feature_importance

def save_results(metrics, feature_importance, filepath):
    """Save metrics and top features to JSON."""
    
    # Convert feature importance to dict for JSON serialization
    top_features = feature_importance[['feature', 'coefficient']].head(10).to_dict(orient='records')
    
    output_data = {
        "model": "logistic_regression_tuned",
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
    best_model = train_logreg(X_train, y_train)
    
    # 3. Evaluate
    metrics = evaluate_model(best_model, X_test, y_test)
    print("\nTest Set Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # 4. Interpret
    feature_names = X_train.columns.tolist()
    feature_importance = interpret_model(best_model, feature_names)
    
    # 5. Save Results
    save_results(metrics, feature_importance, RESULTS_PATH)
