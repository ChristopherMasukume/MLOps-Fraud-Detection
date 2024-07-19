import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import requests
import json
import os
import mlflow.pyfunc

# Create or set the desired experiment
experiment = mlflow.set_experiment("fraud_detection")
print("Experiment ID:", experiment.experiment_id)
print("Experiment Name:", experiment.name)

# Set MLFLOW_TRACKING_URI to the desired directory
mlflow_tracking_uri = "file:///C:/Fraud_Detection/mlruns"
os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri.replace('\\', '/')

# Set the artifact location to a shorter path
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Read the dataset
credit_card_data = pd.read_csv('C:\Fraud_Detection\creditcard.csv')

# Display basic information
print(credit_card_data.head())
print(credit_card_data.tail())
credit_card_data.info()
print(credit_card_data.isnull().sum())
print(credit_card_data['Class'].value_counts())

# Data separation
acceptable = credit_card_data[credit_card_data.Class == 0]
fraudulent = credit_card_data[credit_card_data.Class == 1]

print(acceptable.shape)
print(fraudulent.shape)

# Statistical measures
print(acceptable.Amount.describe())
print(fraudulent.Amount.describe())

print(credit_card_data.groupby('Class').mean())

# Data Sampling
acceptable_sample = acceptable.sample(n=492)
new_dataset = pd.concat([acceptable_sample, fraudulent], axis=0)

# Splitting dataset
x = new_dataset.drop(columns='Class', axis=1)
y = new_dataset['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
print(x.shape, x_train.shape, x_test.shape)

# Training the Model
model = LogisticRegression(max_iter=1000000)
model.fit(x_train, y_train)

# Accuracy
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print('Accuracy on the Training data : ', training_data_accuracy * 100)

x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print('Accuracy on the Test data : ', test_data_accuracy * 100)

# Manually start and end the MLflow run
run = mlflow.start_run()
try:
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_metric("training_data_accuracy", training_data_accuracy)
    mlflow.log_metric("test_data_accuracy", test_data_accuracy)
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"
finally:
    mlflow.end_run()

print(f"Model logged to MLflow with run_id: {run_id}")

# Serve the model
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Example of making predictions using the loaded model
sample_input = x_test.iloc[:5].values.tolist()
predictions = loaded_model.predict(sample_input)
print("Predictions:", predictions)

# Alternatively, you can use Python's built-in HTTP server to serve the model
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the model serving endpoint!"

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    predictions = loaded_model.predict(data)
    return jsonify(predictions.tolist())

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# activate a vertual environment ---C:\Fraud_Detection\Scripts\activate
# to display the mlflow dashboard ---mlflow ui --backend-store-uri file:///C:/Fraud_Detection/mlruns