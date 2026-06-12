import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        # captures activations during forward pass
        self.target_layer.register_forward_hook(self.save_activation)
        # captures gradients during backward pass  
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, input, output):
        self.gradients = output[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        # step 1 — forward pass
        self.model.eval()
        output = self.model(input_tensor)
    
        # step 2 — get predicted class if not provided
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
    
        # step 3 — backward pass on predicted class
        self.model.zero_grad()
        output[0, class_idx].backward()
    
        # step 4 — compute heatmap
        # gradients shape: [1, channels, h, w]
        # activations shape: [1, channels, h, w]
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap = heatmap / (heatmap.max() + 1e-8)
    
        # step 5 — return as numpy
        return heatmap.cpu().numpy()
    
def overlay_heatmap(heatmap, original_image, alpha=0.4):
    # resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (original_image.width, original_image.height))
    
    # convert to colour heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # convert original image to numpy
    original = np.array(original_image)
    original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    
    # overlay
    overlaid = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)
    return overlaid