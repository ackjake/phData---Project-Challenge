#!/usr/bin/env python3
import pickle
import os
from fastapi import FastAPI
import pandas as pd
import json

MODEL_PATH = os.getenv("MODEL_PATH", "model/model.pkl")
MODEL_FEATURES_PATH = os.getenv("MODEL_FEATURES_PATH", "model/model_features.json")
DEMOGRAPHICS_DATA_PATH = os.getenv(
    "ZIPCODE_DEMOGRAPHICS_PATH", "data/zipcode_demographics.csv"
)

model = pickle.load(open(MODEL_PATH, "rb"))
zipcode_demographics_features = (
    pd.read_csv(DEMOGRAPHICS_DATA_PATH, dtype={"zipcode": str})
    .set_index("zipcode")
    .to_dict("index")
)
ordered_features = json.load(open(MODEL_FEATURES_PATH, "rb"))


app = FastAPI()


@app.post("/predict/")
def predict(payload: dict) -> dict:
    """Method to predict

    required input format: {zipcode: x, features: {feature1: y, feature2: z, ...}}

    other docs...
    """
    features = zipcode_demographics_features[payload["zipcode"]]
    features.update(payload["features"])
    pred = model.predict([[features[feature] for feature in ordered_features]])

    return {"property_value_estimate": int(pred), "status": 200}
