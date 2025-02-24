# Monmouth County Real Estate Price Prediction

## Project Overview
This project utilizes machine learning to predict commercial property sale prices in Monmouth County. The model is built using Python and various libraries such as pandas, numpy, scikit-learn, xgboost, and pyspark. The trained model is deployed as a real-time endpoint using Amazon SageMaker. The repository contains all necessary code for data preprocessing, feature engineering, model training, evaluation, and deployment.

## Folder Structure
```
realestate-tool-final/
├── data/
│   └── commericalnj.csv
├── notebooks/
│   ├── submission.ipynb
│   ├── scalingdata.ipynb
│   └── variousmodels.ipynb
├── src/
│   ├── train.py
│   ├── deployment.py
│   ├── app.py
├── templates/
│   ├── index.html
│   ├── result.html
├── requirements.txt
├── README.md
├── model.tar.gz
├── preprocessor.joblib
└── deployed_endpoint.txt
```

- **data/**: Contains raw CSV file with property data.
- **notebooks/**: Jupyter notebooks for data processing, model training, scaling, and evaluation.
- **src/**: Python scripts for model training (`train.py`), deployment (`deployment.py`), and API handling (`app.py`).
- **templates/**: HTML files for web-based input and result display.
- **model.tar.gz**: Trained model packaged for deployment.
- **preprocessor.joblib**: Preprocessing pipeline used for data transformation before inference.
- **deployed_endpoint.txt**: Stores the deployed endpoint name for API reference.

## Environment Setup
- **Programming Language**: Python 3.11
- **Required Packages**:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - pyspark
  - matplotlib
  - Flask
  - boto3
  - joblib
  - python-dotenv
  - git-lfs (if handling large files)

### Installation
Install dependencies using:
```bash
pip install -r requirements.txt
```

## Running the Project

### Running Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/TJC1996/Monmouth-County-Final.git
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the Flask API locally:
   ```bash
   python src/app.py
   ```
4. Use Postman or `curl` to test predictions:
   ```bash
   curl -X POST "http://127.0.0.1:5000/predict" \
     -H "Content-Type: application/json" \
     -d '{"sq_ft": 2000, "acreage": 0.5, "yr_built": 1990, "sale_date": "2024-01-01", "latitude": 40.2, "longitude": -74.0, "total_assmnt": 500000, "taxes_1": 12000, "municipality": "Town A", "property_class": "Class B", "type_use": "Retail", "neigh": "Downtown"}'
   ```

### Running in Amazon SageMaker
1. Open **SageMaker Studio** and connect to your environment.
2. Clone the repository into your SageMaker Studio workspace.
3. Run the training and deployment scripts:
   ```bash
   python src/train.py  # Train and save model
   python src/deployment.py  # Deploy model to SageMaker
   ```
4. Retrieve the deployed endpoint name from `deployed_endpoint.txt` and use it for API calls.

## Deployment Details
The model is deployed using Amazon SageMaker endpoints with the following pipeline:
- **Data Pipeline**: Data is preprocessed in the Jupyter notebooks.
- **Model Training**: Trained using XGBoost and evaluated using RMSE and R² scores.
- **Deployment**: Packaged and deployed using the SageMaker Python SDK.

Example SageMaker Deployment Code:
```python
import sagemaker
from sagemaker.model import Model
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sagemaker_session = sagemaker.Session()
role = os.getenv("AWS_SAGEMAKER_ROLE")
model_uri = os.getenv("S3_MODEL_URI")
endpoint_name = os.getenv("SAGEMAKER_ENDPOINT")

image_uri = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region=sagemaker_session.boto_region_name,
    version="1.5-1"
)

model = Model(
    model_data=model_uri,
    image_uri=image_uri,
    role=role,
    sagemaker_session=sagemaker_session,
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name=endpoint_name
)
```

## API Integration
For querying the deployed model from an external application:
```python
import requests
import json

endpoint_url = "https://<your_endpoint>.execute-api.<region>.amazonaws.com/prod"
data = {"sq_ft": 2000, "acreage": 0.5, "yr_built": 1990, "sale_date": "2024-01-01", "latitude": 40.2, "longitude": -74.0, "total_assmnt": 500000, "taxes_1": 12000, "municipality": "Town A", "property_class": "Class B", "type_use": "Retail", "neigh": "Downtown"}
response = requests.post(endpoint_url, json=data)
print(response.json())
```

## Troubleshooting & FAQ
- **Environment Variables Not Loading**:
  - Ensure `.env` file exists with correct values.
  - Run `source .env` before running scripts.
- **Endpoint Not Found Error**:
  - Verify the endpoint name stored in `deployed_endpoint.txt`.
  - Check the SageMaker dashboard to ensure the endpoint is active.
- **CSV Header Issues**:
  - Ensure correct column names in the dataset and preprocessing pipeline.

## Final Notes
- Test the entire pipeline before submission to confirm accuracy.
- Ensure all environment variables are properly configured.
- If deploying updates, delete the old endpoint before creating a new one.
- For questions or issues, contact me at [tonyclark1996@gmail.com](mailto:tonyclark1996@gmail.com).

