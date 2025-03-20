import os
from pipeline.train import run_training_pipeline


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
