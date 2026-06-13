import shap
import torch
import numpy as np
from PIL import Image

class ShapExplainer:
    def __init__(self, model, background_data):
        self.model = model
        self.background_data = background_data
        self.shap_explainer = shap.GradientExplainer(self.model, self.background_data)

    def explain(self, input_image):
        self.shap_values = self.shap_explainer.shap_values(input_image)
        return self.shap_values