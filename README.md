# Monmouth County Real Estate Price Prediction

## Project Overview
This project aims to predict commercial property sale prices in Monmouth County using machine learning. The model is built using Python with libraries such as Pandas, NumPy, Scikit-learn, XGBoost, and PySpark. The final model is deployed using Amazon SageMaker, making it accessible through a real-time API endpoint. 

## Live Demo
The deployed model can be accessed via a live demo using Ngrok. Users can either interact through a web-based form or send a JSON request using cURL.

### Web UI for Predictions
![Web UI Screenshot](./path_to_image.png)

URL: [https://10e5-52-4-240-77.ngrok-free.app](https://10e5-52-4-240-77.ngrok-free.app)

### API Prediction via cURL
To send a prediction request, use the following command:

```sh
curl -X POST "https://10e5-52-4-240-77.ngrok-free.app/predict" \
  -H "Content-Type: application/json" \
  -d '{"sq_ft": 2000, "acreage": 0.5, "yr_built": 1990, "sale_date": "2024-01-01", "latitude": 40.2, "longitude": -74.0, "total_assmnt": 500000, "taxes_1": 12000, "municipality": "Town A", "property_class": "Class B", "type_use": "Retail", "neigh": "Downtown"}'
```

Expected response:
```json
{
  "predicted_sale_price": 827027.2550148179
}
```

## Repository Structure
```
realestate-tool-final/
├── data/
│   └── commercialnj.csv
├── notebooks/
│   ├── submission.ipynb
│   ├── scalingdata.ipynb
│   └── variousmodels.ipynb
├── src/
│   ├── train.py
│   ├── deployment.py
│   └── app.py
├── templates/
│   ├── index.html
│   ├── result.html
├── requirements.txt
├── README.md
└── .gitignore
```

- **data/**: Raw CSV file with property data.
- **notebooks/**: Jupyter notebooks for data exploration, feature engineering, and training.
- **src/**: Contains Python scripts for training, deployment, and API handling.
- **templates/**: HTML files for the frontend UI.
- **requirements.txt**: Required dependencies for running the project.
- **README.md**: This documentation.

## Setup Instructions

### Environment Setup
#### Locally
1. Clone the repository:
   ```sh
   git clone https://github.com/TJC1996/Monmouth-County-Final.git
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Flask app:
   ```sh
   python src/app.py
   ```

#### Amazon SageMaker
1. Open SageMaker Studio.
2. Clone the repository.
3. Execute `notebooks/submission.ipynb` to train and deploy the model.

## Model Deployment
The model is deployed using Amazon SageMaker as a real-time endpoint. The deployment process includes:
- **Data Preparation:** Feature engineering and scaling.
- **Model Training:** Using XGBoost with hyperparameter tuning.
- **Deployment:** SageMaker endpoint hosting the trained model.

### Deployment Code Example
```python
import sagemaker
from sagemaker.model import Model
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
role = os.getenv("AWS_SAGEMAKER_ROLE")
model_uri = os.getenv("S3_MODEL_URI")
image_uri = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region=sagemaker.Session().boto_region_name,
    version="1.5-1"
)

model = Model(
    model_data=model_uri,
    image_uri=image_uri,
    role=role,
    sagemaker_session=sagemaker.Session(),
)

predictor = model.deploy(initial_instance_count=1, instance_type="ml.g4dn.xlarge")
```

## Logging and Troubleshooting
- **Logs:** Console output for key model metrics (RMSE, R², inference speed).
- **Monitoring:** Use Amazon CloudWatch for endpoint health monitoring.
- **Common Issues:**
  - If the endpoint is not responsive, check SageMaker logs.
  - Ensure AWS credentials and environment variables are correctly set.

## Contact
For inquiries, contact: [tonyclark1996@gmail.com](mailto:tonyclark1996@gmail.com)

---
