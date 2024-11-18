from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd
import os
import glob

# Cargar el modelo entrenado
model = load("model.pkl")

app = FastAPI()

class PredictionRequest(BaseModel):
    data: list

@app.post("/predict")
def batch_predict():
    input_folder = os.getenv("INPUT_FOLDER")
    output_folder = os.getenv("OUTPUT_FOLDER")
    parquet_files = glob.glob(f"{input_folder}/*.parquet")

    for file in parquet_files:
        df = pd.read_parquet(file)
        predictions = model.predict_proba(df)
        output_path = os.path.join(output_folder, os.path.basename(file))
        pd.DataFrame(predictions).to_parquet(output_path)

    return {"message": "Batch predictions completed!"}
