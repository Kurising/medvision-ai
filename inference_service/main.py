import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
from torchvision import transforms
from training_pipeline.model import BrainTumorClassifier

app = FastAPI(
    title = "Medvision AI",
    description = "Brain tumor MRI classification with explainability",
    version="1.0.0"
)

# load model at startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BrainTumorClassifier(num_classes=4)
model.load_state_dict(torch.load('models/best_model.pth',
                                  map_location=device,
                                  weights_only=True))
model = model.to(device)
model.eval()

# class names
class_names = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])



@app.get("/health")
def health():
    return {"status": "ok", "service": "MedVision AI"}
    

@app.get("/model-info")
def model_info():
    return {"model": "EfficientNet-B0", "num_classes": 4, "input_size": (224,224)}
    

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # transform
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted.item()]
    confidence_score = round(confidence.item() * 100, 2)
    
    return {
        "prediction": predicted_class,
        "confidence": f"{confidence_score}%",
        "all_probabilities": {
            class_names[i]: round(probabilities[0][i].item() * 100, 2)
            for i in range(4)
        }
    }