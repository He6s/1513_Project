import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier

# Constants
X_TRAIN_PATH = "data/splits/X_train.csv"
X_TEST_PATH = "data/splits/X_test.csv"
Y_TRAIN_PATH = "data/splits/y_train.csv"
PLOTS_DIR = "results/plots"

def load_data():
    """Load train and test datasets."""
    print("Loading data...")
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)["target"]
    return X_train, X_test, y_train

def train_best_model(X_train, y_train):
    """Train the best XGBoost model found during tuning."""
    print("Training XGBoost model for SHAP analysis...")
    # Best params: colsample_bytree=0.8, learning_rate=0.05, max_depth=4, min_child_weight=3, n_estimators=200, subsample=1.0
    model = XGBClassifier(
        colsample_bytree=0.8,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        n_estimators=200,
        subsample=1.0,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def run_shap_analysis(model, X_train, X_test):
    """Perform SHAP analysis and generate plots."""
    print("Calculating SHAP values...")
    
    # Use a sample of test data if it's too large
    # For TreeExplainer 1000-2000 is fine. For PermutationExplainer, we need fewer (e.g. 100-200)
    shap_data = X_test.sample(n=500, random_state=42) if len(X_test) > 500 else X_test
    
    # Try different explainers to handle XGBoost/SHAP version mismatches
    try:
        print("Attempting to use TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(shap_data)
        explainer_type = "tree"
    except Exception as e:
        print(f"TreeExplainer failed: {e}")
        print("Falling back to PermutationExplainer (Model Agnostic)...")
        # PermutationExplainer is slower, so use fewer samples
        shap_data_small = shap_data.sample(100, random_state=42) if len(shap_data) > 100 else shap_data
        
        # pass the prediction function
        explainer = shap.Explainer(model.predict_proba, shap_data_small)
        shap_values = explainer(shap_data_small)
        # For binary classification, this returns (N, 2) explanation. We focus on class 1.
        shap_values = shap_values[..., 1]
        shap_data = shap_data_small
        explainer_type = "permutation"

    
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # 1. Summary Plot (Bar) - Feature Importance
    print("Generating Summary Bar Plot...")
    plt.figure()
    shap.summary_plot(shap_values, shap_data, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/shap_summary_bar.png")
    plt.close()

    # 2. Summary Plot (Dot) - Directionality
    print("Generating Summary Dot Plot...")
    plt.figure()
    shap.summary_plot(shap_values, shap_data, show=False)
    plt.title("SHAP Summary (Directionality)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/shap_summary_dot.png")
    plt.close()
    
    # 3. Dependence Plots for Top Features
    # Dependence plot works best with simple arrays from TreeExplainer, 
    # but let's try to make it work for Permutation output too
    top_features = ["ap_hi", "age", "cholesterol", "weight"]
    
    for feature in top_features:
        if feature in shap_data.columns:
            print(f"Generating Dependence Plot for {feature}...")
            plt.figure()
            
            try:
                # Handle Explanation vs numpy array
                if hasattr(shap_values, "values"):
                    vals = shap_values.values
                else:
                    vals = shap_values

                shap.dependence_plot(feature, vals, shap_data, show=False)
                plt.title(f"SHAP Dependence: {feature}")
                plt.tight_layout()
                plt.savefig(f"{PLOTS_DIR}/shap_dependence_{feature}.png")
            except Exception as e:
                print(f"Could not generate dependence plot for {feature}: {e}")
            plt.close()

if __name__ == "__main__":
    # 1. Load
    X_train, X_test, y_train = load_data()
    
    # 2. Train
    model = train_best_model(X_train, y_train)
    
    # 3. Explain
    run_shap_analysis(model, X_train, X_test)
    
    print(f"\nSHAP analysis complete. Plots saved to {PLOTS_DIR}")