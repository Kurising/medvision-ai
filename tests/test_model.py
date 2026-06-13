import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from training_pipeline.model import BrainTumorClassifier

def test_model_output_shape():
    model = BrainTumorClassifier(num_classes=4)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (1, 4)

def test_model_classes():
    model = BrainTumorClassifier(num_classes=4)
    assert model is not None