#!/usr/bin/env python3
import pickle
import os
from fastapi import FastAPI

MODEL_PATH = os.getenv("MODEL_PATH", "model/model.pkl")
model = pickle.load(open(MODEL_PATH, "rb"))

app = FastAPI()

@app.post("/predict/")
def predict(payload: dict) -> dict:
    print(type(payload))
    pred = model.predict(payload["features"])
    return {"prediction": int(pred), "status": 200}
