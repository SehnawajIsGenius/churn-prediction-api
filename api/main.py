from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Auto-detect project root
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
os.chdir(project_root)

# Create FastAPI app FIRST
app = FastAPI(title="Customer Churn Prediction API", version="1.0")

# Train model if it doesn't exist
if not os.path.exists('models/model.pkl'):
    print("⚠️ Model not found! Training now...")
    from train_on_startup import train_model
    train_model()

# Load model and encoders
model = joblib.load('models/model.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')
feature_names = joblib.load('models/feature_names.pkl')

class CustomerData(BaseModel):
    tenure: float
    monthly_charges: float
    total_charges: float
    contract_type: str
    payment_method: str
    internet_service: str
    online_security: str
    tech_support: str
    streaming_tv: str
    paperless_billing: str
    senior_citizen: int
    partner: str
    dependents: str
    phone_service: str
    multiple_lines: str

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str

@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API",
        "status": "active",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs",
            "health": "/health"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    try:
        # Convert to DataFrame
        data = {
            'tenure': [customer.tenure],
            'monthly_charges': [customer.monthly_charges],
            'total_charges': [customer.total_charges],
            'contract_type': [customer.contract_type],
            'payment_method': [customer.payment_method],
            'internet_service': [customer.internet_service],
            'online_security': [customer.online_security],
            'tech_support': [customer.tech_support],
            'streaming_tv': [customer.streaming_tv],
            'paperless_billing': [customer.paperless_billing],
            'senior_citizen': [customer.senior_citizen],
            'partner': [customer.partner],
            'dependents': [customer.dependents],
            'phone_service': [customer.phone_service],
            'multiple_lines': [customer.multiple_lines]
        }
        
        df = pd.DataFrame(data)
        
        # Encode categorical variables
        for col in label_encoders.keys():
            df[col] = label_encoders[col].transform(df[col])
        
        # Predict
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        # Determine risk level
        if probability < 0.3:
            risk = "Low"
        elif probability < 0.7:
            risk = "Medium"
        else:
            risk = "High"
        
        return {
            "churn_probability": float(probability),
            "churn_prediction": int(prediction),
            "risk_level": risk
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}
