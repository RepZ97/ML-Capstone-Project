import numpy as np
import pandas as pd

# Classifier algorithms
from sklearn.ensemble import RandomForestClassifier

from flask import Flask, jsonify, request

import joblib
import json

from sklearn.preprocessing import StandardScaler

import pickle


def pre_processing(data):
    # Apply one hot encoding to binary categorical columns
    data = data[data.columns].replace(
        {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 'No, borderline diabetes': 0, 'Yes (during pregnancy)': 1})

    # Apply one hot encoding to other categorical columns
    data = data.join(pd.get_dummies(data['GenHealth'], prefix='GenHealth'))
    data = data.join(pd.get_dummies(data['AgeCategory'], prefix='AgeCategory'))
    data = data.join(pd.get_dummies(data['Race'], prefix='Race'))

    # Then dropping the un-necessary categorical columns
    data.drop(columns=['AgeCategory', 'Race', 'GenHealth'], inplace=True)

    # Standardization of numerical columns
    numerical_columns = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
    Scaler = StandardScaler()
    data[numerical_columns] = Scaler.fit_transform(data[numerical_columns])

    # Re-assign data types of binary columnsfor the memory optimization
    data['Diabetic'] = data['Diabetic'].astype('int8')
    data['HeartDisease'] = data['HeartDisease'].astype('int8')
    data['Smoking'] = data['Smoking'].astype('int8')
    data['AlcoholDrinking'] = data['AlcoholDrinking'].astype('int8')
    data['Stroke'] = data['Stroke'].astype('int8')
    data['DiffWalking'] = data['DiffWalking'].astype('int8')
    data['Sex'] = data['Sex'].astype('int8')
    data['PhysicalActivity'] = data['PhysicalActivity'].astype('int8')
    data['Asthma'] = data['Asthma'].astype('int8')
    data['KidneyDisease'] = data['KidneyDisease'].astype('int8')
    data['SkinCancer'] = data['SkinCancer'].astype('int8')

    # X variable selection
    X_variables = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'Diabetic', 'PhysicalActivity', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer', 'AgeCategory_18-24', 'AgeCategory_25-29', 'AgeCategory_30-34', 'AgeCategory_35-39', 'AgeCategory_40-44', 'AgeCategory_45-49', 'AgeCategory_50-54',
                   'AgeCategory_55-59', 'AgeCategory_60-64', 'AgeCategory_65-69', 'AgeCategory_70-74', 'AgeCategory_75-79', 'AgeCategory_80 or older', 'Race_American Indian/Alaskan Native', 'Race_Asian', 'Race_Black', 'Race_Hispanic', 'Race_Other', 'Race_White', 'GenHealth_Excellent', 'GenHealth_Fair', 'GenHealth_Good', 'GenHealth_Poor', 'GenHealth_Very good']
    # Assign 0 to missing columns
    for x in list(set(X_variables) - set(data.columns)):
        data[x] = 0

    return data[X_variables]


def score(input_data, model):
    return model.predict_proba(input_data)


def post_processing(prediction):
    if len(prediction) == 1:
        return prediction[:, 1][0]
    else:
        return prediction[:, 1]


def app_prediction_function(input_data, model):
    likelihood = post_processing(
        score(input_data=pre_processing(input_data), model=model))
    perc = ' %'
    likelihood_as_perc = str(likelihood*100) + perc
    return likelihood_as_perc


app = Flask(__name__)

# Load model
model = pickle.load(open('model_rf3_best.pickle', 'rb'))
print(model)


@app.route("/")
def index():
    return "Greetings from Prediction API"


@app.route("/classifier", methods=['POST'])
def classifier():
    if request.method == 'POST':
        input_data = request.form.to_dict()
        print(input_data)
        print(type(input_data))
        input_data = pd.DataFrame([input_data])
        print(input_data)
        prediction = app_prediction_function(input_data, model)
        return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)
