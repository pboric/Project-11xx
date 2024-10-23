from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any

app = FastAPI()

# Load models and features
try:
    feature_columns = joblib.load('feature_columns_20241022_215059.pkl')
    stacking_classifier = joblib.load('stacking_classifier_20241022_215059.pkl')
    print("Loaded features:", feature_columns)
    print("\nModel and features loaded successfully!")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

class PredictionRequest(BaseModel):
    data: Dict[str, Any]

def clean_input_data(df):
    """Clean input data before prediction"""
    # Replace inf values with nan
    df = df.replace([np.inf, -np.inf], np.nan)
    # Fill nan values with 0 (or another appropriate value)
    df = df.fillna(0)
    return df

@app.get("/")
def read_root():
    return {
        "message": "Model API is running",
        "features_required": list(feature_columns)
    }

@app.get("/features")
def get_features():
    return {
        "features_required": list(feature_columns),
        "total_features": len(feature_columns)
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([request.data])
        
        # Clean the input data
        input_data = clean_input_data(input_data)
        
        # Check for missing features
        missing_features = set(feature_columns) - set(input_data.columns)
        if missing_features:
            return {
                "error": "Missing features",
                "missing_features": list(missing_features),
                "required_features": list(feature_columns)
            }
        
        # Select only the required features in the correct order
        input_data = input_data[feature_columns]
        
        # Make prediction
        prediction = stacking_classifier.predict(input_data)[0]
        probabilities = stacking_classifier.predict_proba(input_data)[0]
        
        return {
            "prediction": int(prediction),
            "probability": {
                "class_0": float(probabilities[0]),
                "class_1": float(probabilities[1])
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/batch-predict")
def batch_predict(requests: Dict[str, list]):
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame(requests['data'])
        
        # Clean the input data
        input_data = clean_input_data(input_data)
        
        # Select only the required features in the correct order
        input_data = input_data[feature_columns]
        
        # Make predictions
        predictions = stacking_classifier.predict(input_data)
        probabilities = stacking_classifier.predict_proba(input_data)
        
        # Format results
        results = [
            {
                "prediction": int(pred),
                "probability": {
                    "class_0": float(prob[0]),
                    "class_1": float(prob[1])
                }
            }
            for pred, prob in zip(predictions, probabilities)
        ]
        
        return {
            "predictions": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )