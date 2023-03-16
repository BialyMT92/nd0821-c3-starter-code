import requests

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


response = requests.post('https://udacity-mlops-course.onrender.com/predict', json=data)

print(response.status_code)
print(response.json())