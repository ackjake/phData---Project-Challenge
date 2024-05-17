#!/usr/bin/env python3
import pickle
import os
from fastapi import FastAPI
import pandas as pd
import json

MODEL_PATH = os.getenv("MODEL_PATH", "model/model.pkl")
model = pickle.load(open(MODEL_PATH, "rb"))
demographics = (
    pd.read_csv("data/zipcode_demographics.csv", dtype={"zipcode": str})
    .set_index("zipcode")
    .to_dict("index")
)
features = json.load(open("model/model_features.json", 'rb'))


app = FastAPI()


@app.post("/predict/")
def predict(payload: dict) -> dict:
    """Method to predict
    
    required input format: {zipcode: x, features: {feature1: y, feature2: z, ...}}

    other docs... 
    """
    demographic_features = demographics[payload["zipcode"]]
    features = {**demographic_features, **payload["features"]}
    pred = model.predict([[features[feature] for feature in features]])

    return {"prediction": int(pred), "status": 200}
