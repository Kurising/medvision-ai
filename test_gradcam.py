import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from PIL import Image
from torchvision import transforms
from training_pipeline.model import BrainTumorClassifier
from explainability.gradcam import GradCAM, overlay_heatmap
import cv2

# load model
model = BrainTumorClassifier(num_classes=4)
model.load_state_dict(torch.load('models/best_model.pth', map_location='cpu', weights_only=True))
model.eval()

# target last conv layer
target_layer = model.model.features[-1]

# load one test image
image_path = 'data/Testing/glioma/Te-gl_1.jpg'
original_image = Image.open(image_path).convert('RGB')

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

input_tensor = transform(original_image).unsqueeze(0)

# generate heatmap
gradcam = GradCAM(model, target_layer)
heatmap = gradcam(input_tensor)

# overlay on original image
result = overlay_heatmap(heatmap, original_image)

# save result
cv2.imwrite('docs/gradcam_output.png', result)
print("GradCAM saved to docs/gradcam_output.png")