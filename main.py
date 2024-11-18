from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd

# Cargar el modelo entrenado
model = load("model.pkl")

app = FastAPI()

class PredictionRequest(BaseModel):
    data: list

@app.post("/predict")
def predict(request: PredictionRequest):
    df = pd.DataFrame(request.data)
    predictions = model.predict_proba(df)
    return {"predictions": [dict(enumerate(probs)) for probs in predictions]}
