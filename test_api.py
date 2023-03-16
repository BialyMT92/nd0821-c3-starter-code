from fastapi.testclient import TestClient
import json
# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)


def test_api_get_basic():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == ["Welcome to Udacity CI/CD ML Deployment"]


def test_api_post_data_success_under():
    data = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 284582,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }

    r = client.post("/predict", json=data)
    assert r.status_code == 200
    assert r.json() == {"Salary prediction is": ' <=50K'}


def test_api_post_data_success_over():

    data = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 14084,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    }
    r = client.post("/predict", json=data)
    assert r.status_code == 200
    assert r.json() == {"Salary prediction is": ' >50K'}


def test_api_post_data_fail():
    data = {
        "age": "abc",
        "workclass": "Private",
        "fnlgt": 284582,
        "education": 'Masters',
        "education_num": 14,
        "marital_status": 'Married-civ-spouse',
        "occupation": 'Exec-managerial',
        "relationship": 'Wife',
        "race": 'White',
        "sex": 'Female',
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": 'United-States'
    }

    r = client.post("/predict", json=data)
    assert r.status_code == 422
