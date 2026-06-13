import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
import torch

# mock model loading before importing app
with patch('torch.load', return_value={}), \
     patch('training_pipeline.model.BrainTumorClassifier.load_state_dict'):
    from fastapi.testclient import TestClient
    from inference_service.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["model"] == "EfficientNet-B0"