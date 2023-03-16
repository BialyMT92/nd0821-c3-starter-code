import pandas as pd
import joblib
from pipeline.ml.data import process_data
from pipeline.ml.model import compute_model_metrics, inference
import numpy as np

df = pd.read_csv("../../data/clean_census.csv")
model = joblib.load("../../model/rfc_model.pkl")
lb = joblib.load("../../model/lb.pkl")
encoder = joblib.load("../../model/encoder.pkl")

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

""" Function for calculating descriptive stats on slices of the Census dataset."""

over = {
    "age": 31,
    "workclass": "Private",
    "fnlgt": 45781,
    "education": "Masters",
    "education_num": 14,
    "marital-status": "Never-married",
    "occupation": "Prof-specialty",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital-gain": 14084,
    "capital-loss": 0,
    "hours-per-week": 50,
    "native-country": "United-States"
}

under = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 284582,
    "education": "Masters",
    "education_num": 14,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Wife",
    "race": "White",
    "sex": "Female",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

df_temp = pd.DataFrame([over])


X_test, _, _, _ = process_data(df_temp, categorical_features=cat_features,
                                      training=False,
                                      encoder=encoder, lb=lb)
preds = inference(model, X_test)
print(preds)
value = lb.inverse_transform(preds)[0]
print(value)

df_temp = pd.DataFrame([under])


X_test, _, _, _ = process_data(df_temp, categorical_features=cat_features,
                                      training=False,
                                      encoder=encoder, lb=lb)
preds = inference(model, X_test)
print(preds)
value = lb.inverse_transform(preds)[0]
print(value)