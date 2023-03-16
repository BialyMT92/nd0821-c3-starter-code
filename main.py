import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pipeline.ml.data import process_data
from pipeline.ml.model import compute_model_metrics, inference


# Declare the data object with its components and their type. Columns: age,workclass,fnlgt,education,education-num,
# marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,
# salary Default person: 37, Private,284582, Masters,14, Married-civ-spouse, Exec-managerial, Wife, White, Female,0,
# 0,40, United-States, <=50K


class TaggedItem(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


'''
data = TaggedItem().dict()

df = pd.DataFrame([data])
df.columns = df.columns.str.replace("_", "-")

print(df)
'''
app = FastAPI(
    title="Udacity CI/CD course",
    description="An API that is running ML model for prediction of possible yearly income",
    version="1.0.0")


@app.get("/")
async def frontend():
    return {"Welcome to Udacity CI/CD ML Deployment"}


@app.post("/predict")
async def predict(item: TaggedItem):
    model = joblib.load("model/rfc_model.pkl")
    lb = joblib.load("model/lb.pkl")
    encoder = joblib.load("model/encoder.pkl")

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

    data = item.dict()

    df = pd.DataFrame([data])
    df.columns = df.columns.str.replace("_", "-")

    X_test, _, _, _ = process_data(df, categorical_features=cat_features,
                                        training=False,
                                        encoder=encoder, lb=lb)

    pred_array = inference(model, X_test)
    value = lb.inverse_transform(pred_array)[0]
    return {"Salary prediction is": value}
