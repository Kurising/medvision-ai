import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from training_pipeline.dataset import BrainDataset
from training_pipeline.model import BrainTumorClassifier
import mlflow

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = BrainDataset(root_dir='data/Training', transform=train_transforms)
val_dataset = BrainDataset(root_dir='data/Testing', transform=val_transforms)

train_Loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_Loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = BrainTumorClassifier(num_classes=4)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

with mlflow.start_run():
    mlflow.log_param("epochs", num_epochs)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("learning_rate", 0.001)

    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_Loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in val_Loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        epoch_loss = val_loss / len(val_Loader)
        epoch_accuracy = total_correct / total_samples

        mlflow.log_metric("val_loss", epoch_loss, step=epoch)
        mlflow.log_metric("val_accuracy", epoch_accuracy, step=epoch)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")

        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f"Best model saved with accuracy: {best_accuracy:.4f}")

    mlflow.log_artifact('models/best_model.pth')
    print(f"Training complete. Best accuracy: {best_accuracy:.4f}")