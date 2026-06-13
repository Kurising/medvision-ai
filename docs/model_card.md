# Model Card - MedVision AI

## Model Description
EfficientNet-B0 fine-tuned for brain tumor MRI classification. 4 classes: glioma, meningioma, pituitary, no tumor. Trained on Kaggle Brain Tumor MRI dataset (7,200 images).

## Intended Use
Decision-support tool for radiologists. NOT a replacement for clinical diagnosis. Final decision must be made by a qualified medical professional.

## Performance
| Metric | Score |
|--------|-------|
| Accuracy | 95.37% |
| Precision | 95.60% |
| Recall | 95.37% |
| F1 Score | 95.31% |
| AUC-ROC | 99.14% |

## Limitations
- Trained on Kaggle dataset — may not generalise to all 
  clinical MRI equipment
- Glioma/meningioma confusion observed in 41 cases
- Not validated in a real clinical setting
- Performance may degrade on out-of-distribution images

## EU AI Act Classification
HIGH RISK — Article 6, Annex III (medical devices)

## Compliance Measures
- GradCAM explainability on every prediction
- SHAP feature importance scores
- Human oversight required - system is advisory only
- Audit logging planned for production deployment

## Training Data
- Dataset: Kaggle Brain Tumor MRI Dataset
- Size: 7,200 images (5,600 train / 1,600 test)
- Classes: glioma, meningioma, pituitary, notumor
- Class balance: 1,400 images per class (balanced)