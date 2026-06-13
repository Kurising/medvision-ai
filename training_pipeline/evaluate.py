import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
import numpy as np
from training_pipeline.dataset import BrainDataset
from training_pipeline.model import BrainTumorClassifier

def evaluate(model, data_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []  # store softmax probabilities

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
    
    # compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    cm = confusion_matrix(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'auc_roc': auc
    }

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = BrainDataset(root_dir='data/Testing', transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = BrainTumorClassifier(num_classes=4)
    model.load_state_dict(torch.load('models/best_model.pth',
                                      map_location=device,
                                      weights_only=True))
    model = model.to(device)
    
    results = evaluate(model, val_loader, device)
    
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"AUC-ROC:   {results['auc_roc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(results['confusion_matrix'])