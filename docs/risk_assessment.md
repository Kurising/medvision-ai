# Risk Assessment — MedVision AI

## Risk Classification
HIGH RISK under EU AI Act Article 6, Annex III

## Potential Harms
| Risk | Severity | Likelihood |
|------|----------|------------|
| False negative: missed tumor | Critical | Low (4.63%) |
| False positive: unnecessary procedures | Moderate | Low |
| Over-reliance on AI | High | Medium |
| Distribution shift in deployment | High | Medium |

## Mitigation Measures
- GradCAM explainability shows decision basis
- Confidence scores flag uncertain predictions
- Human oversight mandatory - radiologist must review
- System labelled as decision-support only

## Intended Safeguards
This system must never be used as the sole basis for 
clinical diagnosis. It is a decision-support tool only.

## Deployment Constraints
- Must be used by qualified medical professionals only
- Not approved for direct patient-facing use
- Requires validation on local clinical data before deployment