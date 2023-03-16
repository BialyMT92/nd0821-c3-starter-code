# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model was created for the Udacity course "Deploying a ML Model to Cloud Application Platform with FastAPI". 

The project was made by a student - Mateusz Białek.

Model Version - V1.0.0.

Model Type - Random Forest Classifier from the scikit-learn library.
## Intended Use
The model, based on personal data, predicts a person's earnings in two ranges: <=$50k per year or >$50k per year.
## Training Data
Data are available on site: https://archive.ics.uci.edu/ml/datasets/census+income.
```
Donor:

Ronny Kohavi and Barry Becker
Data Mining and Visualization
Silicon Graphics.
e-mail: ronnyk '@' sgi.com for questions.


Data Set Information:

Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

Prediction task is to determine whether a person makes over 50K a year.


Attribute Information:

Listing of attributes:

>50K, <=50K.

age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
```

## Evaluation Data
For the evaluation data I used 20% of data 
```
train, test = train_test_split(data, test_size=0.20)
```
## Metrics
Model is running with following metrics:
```
Precision: 0.7245508982035929
Recall: 0.6111111111111112
Fbeta: 0.663013698630137
```
## Ethical Considerations
For some people, the set of data may be discriminatory especially when we include race, sex, nationality etc.
## Caveats and Recommendations
To improve this ML Pipeline it should use additional software for tracking and tagging the artefacts like wandb. 
Use mlflow to increase level of automatization, parser and hydra for parametrization and pushing multiple hyperparameter.
Result of each run for training the model should be checked, and picked up for the best model as a standard. 
Better model could not be able to push into github which could cause problem with pytest in github action.