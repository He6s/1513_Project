import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from evaluate import compute_classification_metrics, save_results


X_TRAIN_PATH = "data/splits/X_train.csv"
X_TEST_PATH = "data/splits/X_test.csv"
Y_TRAIN_PATH = "data/splits/y_train.csv"
Y_TEST_PATH = "data/splits/y_test.csv"

RESULTS_PATH = "results/logistic_regression_results.json"


def load_split_data():
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)["target"]
    y_test = pd.read_csv(Y_TEST_PATH)["target"]
    return X_train, X_test, y_train, y_test


def train_and_evaluate(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    metrics = compute_classification_metrics(y_test, y_pred)
    results = {"model": "logistic_regression", **metrics}
    return results


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split_data()
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("Logistic regression results:")
    for key, value in results.items():
        print(f"{key}: {value}")

    save_results(results, RESULTS_PATH)
    print(f"\nSaved results to: {RESULTS_PATH}")
