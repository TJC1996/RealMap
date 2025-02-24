from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import boto3
import json
import joblib
import numpy as np
import datetime
import pandas as pd
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder="templates")
CORS(app)

endpoint_name = os.getenv("SAGEMAKER_ENDPOINT")
preprocessor = joblib.load("preprocessor.joblib")

runtime_client = boto3.client('sagemaker-runtime')

def preprocess_input(data):
    try:
        sq_ft = float(data.get("sq_ft", 0))
        acreage = float(data.get("acreage", 0))
        yr_built = int(data.get("yr_built", 0))
        sale_date_str = data.get("sale_date", None)
        
        if sale_date_str:
            sale_date = datetime.datetime.strptime(sale_date_str, "%Y-%m-%d")
            sale_year = sale_date.year
            sale_month = sale_date.month
        else:
            sale_year = 0
            sale_month = 0

        current_year = datetime.datetime.now().year
        building_age = current_year - yr_built if yr_built > 0 else 0
        sale_month_sine = np.sin(2 * np.pi * sale_month / 12)
        sale_month_cosine = np.cos(2 * np.pi * sale_month / 12)
        latitude = float(data.get("latitude", 0))
        longitude = float(data.get("longitude", 0))
        total_assmnt = float(data.get("total_assmnt", 0))
        taxes_1 = float(data.get("taxes_1", 0))

        municipality = data.get("municipality", "")
        property_class = data.get("property_class", "")
        type_use = data.get("type_use", "")
        neigh = data.get("neigh", "")

        input_df = pd.DataFrame([{
            "Sq. Ft.": sq_ft,
            "Acreage": acreage,
            "Building Age": building_age,
            "Sale Year": sale_year,
            "Sale Month Sine": sale_month_sine,
            "Sale Month Cosine": sale_month_cosine,
            "Latitude": latitude,
            "Longitude": longitude,
            "Total Assmnt": total_assmnt,
            "Taxes 1": taxes_1,
            "Municipality": municipality,
            "Property Class": property_class,
            "Type/Use": type_use,
            "Neigh": neigh
        }])

        transformed = preprocessor.transform(input_df)
        features = transformed.tolist()[0]
        return features
    except Exception as e:
        logging.error(f"Error in preprocessing input: {e}")
        raise ValueError(f"Error in preprocessing input: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        features = preprocess_input(data)
        csv_input = ",".join(map(str, features))

        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="text/csv",
            Body=csv_input
        )

        raw_prediction = json.loads(response['Body'].read().decode())
        predicted_sale_price = np.expm1(raw_prediction[0]) if isinstance(raw_prediction, list) else np.expm1(raw_prediction)

        return jsonify({"predicted_sale_price": predicted_sale_price})
    
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
