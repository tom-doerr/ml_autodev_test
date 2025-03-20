import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle
import json
from sklearn.utils import Bunch


def run_training_pipeline() -> None:
    """End-to-end ML training pipeline
    
    Saves outputs:
    - model.pkl: Trained Decision Tree classifier
    - metrics.json: Training and test accuracy metrics
    """
    # Load data
    iris: Bunch = load_iris()
    X = pd.DataFrame(iris["data"], columns=iris["feature_names"])
    y = iris["target"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X_train, y_train)

    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    # Save artifacts with error handling
    try:
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        metrics = {"train_accuracy": float(train_acc), "test_accuracy": float(test_acc)}
        with open("metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
            
    except IOError as e:
        raise RuntimeError(f"Failed to save pipeline artifacts: {str(e)}") from e
