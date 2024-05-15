#!/usr/bin/env python3

import pickle
import os
import fastapi

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
model = pickle.load(MODEL_PATH, "r")

