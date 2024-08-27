import argparse
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import schedule
import time

def main(data_path):
    print(f"Reading data from: {data_path}")
    # Set the tracking URI to your MLflow server
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Replace with your MLflow tracking server URI

    # Load data
    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error reading the data file: {e}")
        return

    # Preprocess and split data
    X = data.drop(columns='Class')
    y = data['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate model
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    # Log model with MLflow
    with mlflow.start_run() as run:
        mlflow.log_param('random_state', 42)
        mlflow.log_metric('train_accuracy', train_accuracy)
        mlflow.log_metric('test_accuracy', test_accuracy)
        mlflow.sklearn.log_model(model, 'model')

        # Register the model
        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name="LogisticRegressionModel"
        )

    print(f"Train Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")

def job():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help="Path to the training data", required=True)
    args = parser.parse_args()

    # Print args.data for debugging
    print(f"Data path provided: {args.data}")

    # Check if the file exists
    if not os.path.isfile(args.data):
        print(f"Error: The file {args.data} does not exist.")
    else:
        main(args.data)

if __name__ == "__main__":
    # Schedule the job to run every 30 days
    schedule.every(30).days.do(job)

    # Run the scheduling loop
    while True:
        schedule.run_pending()
        time.sleep(1)

# run:
# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
# then run:
# c:/Fraud_Detection/Scripts/python.exe c:/Fraud_Detection/train_log.py --data "c:/Fraud_Detection/creditcard.csv"