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
    age: int = 37
    workclass: str = 'Private'
    fnlgt: int = 284582
    education: str = 'Masters'
    education_num: int = 14
    marital_status: str = 'Married-civ-spouse'
    occupation: str = 'Exec-managerial'
    relationship: str = 'Wife'
    race: str = 'White'
    sex: str = 'Female'
    capital_gain: int = 0
    capital_loss: int = 0
    hours_per_week: int = 40
    native_country: str = 'United-States'


'''
data = TaggedItem().dict()

df = pd.DataFrame([data])
df.columns = df.columns.str.replace("_", "-")

print(df)
'''
app = FastAPI()


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

    X_test, y_test, _, _ = process_data(df, categorical_features=cat_features,
                                        training=False,
                                        encoder=encoder, lb=lb)

    pred_array = inference(model, X_test)

    pred_value = np.argmax(pred_array)

    if pred_value == 1:
        salary = '>50k$'
    else:
        salary = '<= 50k$'
    return {"Salary prediction is": salary}
