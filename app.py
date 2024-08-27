from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flasgger import Swagger, swag_from

app = Flask(__name__)
swagger = Swagger(app)

# Load the saved model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return "Welcome to the Fraud Detection API!"

@app.route('/predict', methods=['POST'])
@swag_from({
    'responses': {
        200: {
            'description': 'Prediction of transaction acceptability',
            'examples': {
                'application/json': {
                    'prediction': 'Acceptable transaction'
                }
            }
        }
    },
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'Time': {'type': 'number'},
                    'V1': {'type': 'number'},
                    'V2': {'type': 'number'},
                    'V3': {'type': 'number'},
                    'V4': {'type': 'number'},
                    'V5': {'type': 'number'},
                    'V6': {'type': 'number'},
                    'V7': {'type': 'number'},
                    'V8': {'type': 'number'},
                    'V9': {'type': 'number'},
                    'V10': {'type': 'number'},
                    'V11': {'type': 'number'},
                    'V12': {'type': 'number'},
                    'V13': {'type': 'number'},
                    'V14': {'type': 'number'},
                    'V15': {'type': 'number'},
                    'V16': {'type': 'number'},
                    'V17': {'type': 'number'},
                    'V18': {'type': 'number'},
                    'V19': {'type': 'number'},
                    'V20': {'type': 'number'},
                    'V21': {'type': 'number'},
                    'V22': {'type': 'number'},
                    'V23': {'type': 'number'},
                    'V24': {'type': 'number'},
                    'V25': {'type': 'number'},
                    'V26': {'type': 'number'},
                    'V27': {'type': 'number'},
                    'V28': {'type': 'number'},
                    'Amount': {'type': 'number'}
                },
                'required': ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                             'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                             'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
            }
        }
    ]
})
def predict():
    try:
        # Get data from POST request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Create a DataFrame from the received data
        transaction_data = pd.DataFrame(data, index=[0])

        # Perform prediction using the loaded model
        prediction = model.predict(transaction_data)

        # Prepare response
        if prediction[0] == 0:
            result = 'Acceptable transaction'
        else:
            result = 'Fraudulent transaction'

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    app.run(debug=os.environ.get('DEBUG', 'False') == 'True')