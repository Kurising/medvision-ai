import torch
from training_pipeline.model import BrainTumorClassifier

def load_model(model_path: str, device: str = 'cpu'):
    model = BrainTumorClassifier(num_classes=4)
    model.load_state_dict(torch.load(model_path, 
                                      map_location=device,
                                      weights_only=True))
    model.eval()
    return model