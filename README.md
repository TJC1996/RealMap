# 🏡 Monmouth County Real Estate Price Prediction

## 📌 Project Overview
This project aims to predict **commercial property sale prices** in Monmouth County using machine learning. The model is built using Python with libraries such as **Pandas, NumPy, Scikit-learn, XGBoost, and PySpark**. The final model is deployed using **Amazon SageMaker**, making it accessible through a real-time API endpoint.

---

## 🚀 Live Demo
The deployed model can be accessed via a live demo using **Ngrok**. Users can either interact through a **web-based form** or send a **JSON request** using `cURL`.

### 🌐 **Web UI for Predictions**
#### **User Inputs in Web Form:**
The interactive web interface allows users to enter property details and obtain a **predicted sale price**.

<img width="838" alt="Screenshot 2025-02-24 at 7 43 48 PM" src="https://github.com/user-attachments/assets/4c953b53-1983-4322-9e6b-1a2d05a3b74f" />


📌 **How It Works**:
1. Users fill out property details (square footage, acreage, year built, etc.).
2. The form submits data to the Flask API.
3. The API processes the input, sends it to SageMaker, and returns a prediction.

🔗 **Live Demo URL:**  
[https://10e5-52-4-240-77.ngrok-free.app](https://10e5-52-4-240-77.ngrok-free.app)

---

### 🖼 **Example Prediction and Map Visualization**
Once a user enters property details, the predicted price is displayed. Below is an example:

#### **Predicted Price Output:**
<img width="303" alt="Screenshot 2025-02-24 at 7 44 00 PM" src="https://github.com/user-attachments/assets/6d725aba-05b2-479e-ab19-511adda52760" />

📌 **What This Shows**:
- The model predicted a sale price of **$787,103.71** for the inputted property.
- The result page displays the estimated price based on **historical trends and property characteristics**.

---

### 🖼 **Interactive Map with Historical Sales**
The application also includes a **map visualization** that displays historical property sales.

<img width="850" alt="Screenshot 2025-02-24 at 7 48 36 PM" src="https://github.com/user-attachments/assets/e3babed1-9b64-431d-b9fa-6ac62f56afcc" />


📌 **Key Features**:
- Displays **historical sale prices** with detailed property information.
- Users can compare the **model’s predictions** with real-world sale prices.
- Provides a **geospatial analysis** of property values.

---

## 📡 API Prediction via cURL
Users can also send property data via **cURL** to obtain a prediction.

```sh
curl -X POST "https://10e5-52-4-240-77.ngrok-free.app/predict" \
  -H "Content-Type: application/json" \
  -d '{"sq_ft": 2000, "acreage": 0.5, "yr_built": 1990, "sale_date": "2024-01-01", "latitude": 40.2, "longitude": -74.0, "total_assmnt": 500000, "taxes_1": 12000, "municipality": "Town A", "property_class": "Class B", "type_use": "Retail", "neigh": "Downtown"}'
