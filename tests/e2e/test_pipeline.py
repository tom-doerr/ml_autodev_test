import os
import sys
import json
import pickle
from pathlib import Path

# Add project root to Python path (matches conftest.py setup)
root_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(root_dir))

from pipeline import run_training_pipeline


def test_pipeline_execution():
    """Test full pipeline execution from raw data to trained model"""
    # Setup - ensure clean start
    for filename in ["model.pkl", "metrics.json"]:
        if os.path.exists(filename):
            os.remove(filename)

    # Execute pipeline
    run_training_pipeline()

    # Verify outputs
    assert os.path.exists("model.pkl"), "Model file not created"
    assert os.path.exists("metrics.json"), "Metrics file not created"
    assert os.path.getsize("model.pkl") > 1024, "Model file seems too small"
    assert os.path.getsize("metrics.json") > 10, "Metrics file seems empty"

    # Verify metric contents
    with open("metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    
    assert "train_accuracy" in metrics, "Missing train accuracy in metrics"
    assert "test_accuracy" in metrics, "Missing test accuracy in metrics"
    assert 0 <= metrics["train_accuracy"] <= 1, "Invalid train accuracy value"
    assert 0 <= metrics["test_accuracy"] <= 1, "Invalid test accuracy value"
    
    # Verify model can predict
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    
    sample_input = [[5.8, 3.0, 3.8, 1.2]]  # Sample Iris features
    prediction = model.predict(sample_input)
    assert prediction.shape == (1,), "Invalid prediction shape"
