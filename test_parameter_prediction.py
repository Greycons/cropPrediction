#!/usr/bin/env python3
"""
Test parameter prediction functionality
"""

import pandas as pd
import numpy as np
import pickle
import os

def test_parameter_prediction():
    """Test the parameter prediction with sample data"""
    
    print("🧪 Testing Parameter Prediction...")
    
    # Check if models exist
    model_files = [
        'notebooks/rf_model.pkl',
        'notebooks/ridge_model.pkl',
        'notebooks/gbr_model.pkl'
    ]
    
    models = {}
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                with open(model_file, 'rb') as f:
                    models[os.path.basename(model_file).replace('_model.pkl', '')] = pickle.load(f)
                print(f"✅ Loaded {model_file}")
            except Exception as e:
                print(f"❌ Error loading {model_file}: {str(e)}")
    
    if not models:
        print("❌ No models available for testing")
        return
    
    # Create sample input data (same format as Streamlit app)
    input_data = {
        'state': 'Andhra Pradesh',
        'district': 'Anantapur',
        'year': 2024,
        'crop': 'Rice',
        'groundwater_ph': 7.5,
        'ec_groundwater_(µs/cm)': 1200,
        'hardness_groundwater_(mg/l)': 400,
        'nitrate_groundwater_(mg/l)': 40,
        'rainfall_mm': 650,
        'soil_ph': 7.2,
        'soil_organic_carbon': 0.7,
        'soil_nitrogen': 200,
        'soil_phosphorus': 20,
        'soil_potassium': 250
    }
    
    print(f"📊 Testing with sample input:")
    for key, value in input_data.items():
        print(f"   {key}: {value}")
    
    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    
    # One-hot encode categorical variables (same as in app)
    input_encoded = pd.get_dummies(input_df, columns=['state', 'district', 'crop'], drop_first=True)
    
    print(f"📊 After one-hot encoding: {input_encoded.shape[1]} features")
    print(f"   Features: {list(input_encoded.columns)}")
    
    # Load feature info if available
    feature_info = None
    try:
        with open('notebooks/feature_info.pkl', 'rb') as f:
            feature_info = pickle.load(f)
        print(f"✅ Loaded feature info: {feature_info['n_features']} expected features")
    except Exception as e:
        print(f"⚠️ Could not load feature info: {str(e)}")
    
    if feature_info:
        # Use the exact feature columns from training
        expected_columns = feature_info['feature_columns']
        
        # Add missing columns with 0
        for col in expected_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Reorder columns to match training data exactly
        input_encoded = input_encoded[expected_columns]
        
        print(f"📊 After feature alignment: {input_encoded.shape[1]} features")
        print(f"   Expected: {len(expected_columns)}")
        print(f"   Actual: {input_encoded.shape[1]}")
    
    # Test predictions
    predictions = {}
    for model_name, model in models.items():
        try:
            if isinstance(model, dict) and 'model' in model:
                # Handle models with scalers
                scaler = model.get('scaler')
                model_obj = model['model']
                if scaler and model_obj:
                    input_scaled = scaler.transform(input_encoded)
                    pred = model_obj.predict(input_scaled)[0]
                else:
                    pred = model_obj.predict(input_encoded)[0]
            else:
                pred = model.predict(input_encoded)[0]
            predictions[model_name] = pred
            print(f"✅ {model_name}: {pred:.2f}")
        except Exception as e:
            print(f"❌ Error with {model_name}: {str(e)}")
    
    if predictions:
        avg_prediction = np.mean(list(predictions.values()))
        print(f"\n🎯 Average Prediction: {avg_prediction:.2f}")
        print(f"📊 Individual Predictions:")
        for name, pred in predictions.items():
            print(f"   {name}: {pred:.2f}")
        print("\n✅ Parameter prediction is working!")
    else:
        print("\n❌ No successful predictions")

if __name__ == "__main__":
    test_parameter_prediction()
