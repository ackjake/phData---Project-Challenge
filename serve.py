#!/usr/bin/env python3
import json
import pickle

import pandas as pd
from fastapi import FastAPI

model = pickle.load(open("model.pkl", "rb"))
zipcode_demographics_features = (
    pd.read_csv("zipcode_demographics.csv", dtype={"zipcode": str})
    .set_index("zipcode")
    .to_dict("index")
)
ordered_features = json.load(open("model_features.json", "rb"))


app = FastAPI()


@app.post("/predict/")
def predict(payload: dict) -> dict:
    """Post method to predict property value.

    Joins demographics data to input features on zipcode. Assumes that input
    zipcode exists within zipcode_demographics.csv.

    Args:
        payload: json input, requires zipcode and features as keys.
            Expected input: {zipcode: x, features: {feature1: x, feature2: y, ...}}

    Returns:
        Paylod containing property value estimate.
    """
    features = zipcode_demographics_features[payload["zipcode"]]
    features.update(payload["features"])
    pred = model.predict([[features[feature] for feature in ordered_features]])

    return {"property_value_estimate": int(pred), "status": 200}
