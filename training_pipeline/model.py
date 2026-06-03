import torch.nn as nn
from torchvision import models

class BrainTumorClassifier(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()
        self.model = models.efficientnet_b0(weights ='DEFAULT')
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.classifier[1].in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
