import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Load the trained model
model = joblib.load("titanic_rf_model.pkl")  # Make sure this file exists

# ✅ REMOVE the line below if you didn’t save the scaler
# scaler = joblib.load("scaler.pkl")

app = FastAPI()

class PassengerData(BaseModel):
    features: list  # should be 11 input features

@app.get("/")
def read_root():
    return {"message": "🚢 Titanic Survival Prediction API is running!"}

@app.post("/predict")
def predict(data: PassengerData):
    input_array = np.array(data.features).reshape(1, -1)

    # ✅ Skip scaling
    input_scaled = input_array

    prediction = model.predict(input_scaled)[0]
    return {"prediction": int(prediction)}
