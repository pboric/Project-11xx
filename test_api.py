import pandas as pd
import numpy as np
import requests
import json

def clean_for_json(data):
    """Clean data to make it JSON compatible"""
    if isinstance(data, dict):
        return {k: clean_for_json(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [clean_for_json(x) for x in data]
    elif isinstance(data, (int, float)):
        # Handle inf, -inf, nan
        if np.isnan(data) or np.isinf(data):
            return None
        return data
    else:
        return data

def load_test_data(file_path):
    df = pd.read_csv(file_path)
    # Replace inf and -inf with nan
    df = df.replace([np.inf, -np.inf], np.nan)
    # Fill nan values with 0 (or another appropriate value)
    df = df.fillna(0)
    return df

def test_single_prediction(data):
    """Test a single prediction with the first row of data"""
    # Get the first row as a dictionary
    sample = data.iloc[0].to_dict()
    
    # Clean the data for JSON
    sample = clean_for_json(sample)
    
    try:
        # Make prediction request
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"data": sample}
        )
        
        print("\n=== Single Prediction Test ===")
        print("\nResponse:")
        print(json.dumps(response.json(), indent=2))
        
    except Exception as e:
        print(f"Error in single prediction: {str(e)}")

def test_batch_prediction(data, batch_size=5):
    """Test batch prediction with multiple rows"""
    # Take first batch_size rows
    batch = data.head(batch_size).to_dict('records')
    
    # Clean the data for JSON
    batch = clean_for_json(batch)
    
    try:
        # Make batch prediction request
        response = requests.post(
            "http://127.0.0.1:8000/batch-predict",
            json={"data": batch}
        )
        
        print("\n=== Batch Prediction Test ===")
        print(f"Testing with {batch_size} samples")
        print("\nResponse:")
        print(json.dumps(response.json(), indent=2))
        
    except Exception as e:
        print(f"Error in batch prediction: {str(e)}")

def check_features():
    """Check what features the model expects"""
    response = requests.get("http://127.0.0.1:8000/features")
    print("\n=== Required Features ===")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    # Replace with your actual file path
    file_path = r'C:\Users\kresi\OneDrive\Desktop\Turing college\Project11xx\processed_test.csv'  # ← CHANGE THIS to your test data path
    
    try:
        # Check required features first
        check_features()
        
        # Load and clean test data
        print("\nLoading and cleaning test data...")
        test_data = load_test_data(file_path)
        print(f"Loaded test data shape: {test_data.shape}")
        
        # Run tests
        test_single_prediction(test_data)
        test_batch_prediction(test_data, batch_size=3)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        
    print("\nTesting completed!")