@echo off
REM Start MLflow server in a new window
start "MLflow Server" cmd /c "mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000"

REM Wait for a few seconds to ensure the server is up
timeout /t 10 /nobreak

REM Run the training script
c:/Fraud_Detection/Scripts/python.exe c:/Fraud_Detection/train_log.py --data "c:/Fraud_Detection/creditcard.csv"

REM Close the MLflow server
taskkill /FI "WINDOWTITLE eq MLflow Server*"

echo Training process completed.
pause
