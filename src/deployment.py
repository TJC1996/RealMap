import sagemaker
from sagemaker.model import Model
from sagemaker import get_execution_role
import uuid
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sagemaker_session = sagemaker.Session()
role = os.getenv("AWS_SAGEMAKER_ROLE")
model_uri = os.getenv("S3_MODEL_URI")

image_uri = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region=sagemaker_session.boto_region_name,
    version="1.5-1"
)

unique_id = uuid.uuid4().hex[:8]
model_name = f"realestate-xgboost-model-{unique_id}"
endpoint_name = f"realestate-xgboost-endpoint-{unique_id}"

print(f"🚀 Deploying Model as {model_name} with Endpoint {endpoint_name}...")

sm_client = sagemaker_session.sagemaker_client

def cleanup_resources(ep_name, mod_name):
    try:
        print(f"🗑 Deleting endpoint '{ep_name}' if it exists...")
        sm_client.delete_endpoint(EndpointName=ep_name)
        time.sleep(5)  
    except Exception:
        print("ℹ️ No existing endpoint to delete.")

    try:
        print(f"🗑 Deleting endpoint config '{ep_name}' if it exists...")
        sm_client.delete_endpoint_config(EndpointConfigName=ep_name)
    except Exception:
        print("ℹ️ No existing endpoint config to delete.")

    try:
        print(f"🗑 Deleting model '{mod_name}' if it exists...")
        sm_client.delete_model(ModelName=mod_name)
    except Exception:
        print("ℹ️ No existing model to delete.")

cleanup_resources(endpoint_name, model_name)

model = Model(
    model_data=model_uri,
    image_uri=image_uri,
    role=role,
    sagemaker_session=sagemaker_session,
    name=model_name
)

model.create(instance_type="ml.g4dn.xlarge")

print(f"✅ Model {model_name} successfully registered in SageMaker!")

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge",
    endpoint_name=endpoint_name,
    wait=True
)

with open("deployed_endpoint.txt", "w") as f:
    f.write(endpoint_name)

print(f"✅ Model successfully deployed! Endpoint Name: {endpoint_name}")
