from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("titanic_rf_model.pkl")

class Passenger(BaseModel):
    features: list

@app.get("/")
def root():
    return {"message": "🚢 Titanic Survival Prediction API is up!"}

@app.post("/predict")
def predict(passenger: Passenger):
    prediction = model.predict([passenger.features])
    return {"survived": int(prediction[0])}
