import joblib
from flask import Flask, request, jsonify
import numpy as np
import os
import pandas as pd
app = Flask(__name__)

current_directory = os.path.dirname(os.path.abspath(__file__))  # Get the current script directory
model_path = os.path.join(current_directory, 'model', 'balancedrandomforest_final_model.joblib')

# Load the model
model = joblib.load(model_path)
selected_features = ['GP', 'MIN', 'PTS', 'FG%', '3P Made', '3P%', 'FT%', 'REB', 'AST', 'STL', 'BLK', 'TOV']

@app.route('/')
def index():
    return "ML Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    feature_names = ['GP', 'MIN', 'PTS', 'FG%', '3P Made', '3P%', 'FT%', 'REB', 'AST', 'STL', 'BLK', 'TOV']
    input_df = pd.DataFrame([data], columns=feature_names)
    # Make predictions using the loaded model
    prediction = model.predict(input_df)
    
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
