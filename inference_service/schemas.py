from pydantic import BaseModel

class PredictionResponse(BaseModel):
    prediction: str
    confidence: str
    all_probabilities: dict