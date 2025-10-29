#!/usr/bin/env python3
"""
Create missing model files for the Streamlit app
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def create_missing_models():
    """Create the missing Random Forest and Gradient Boosting models"""
    
    print("🔧 Creating missing model files...")
    
    # Create notebooks directory if it doesn't exist
    os.makedirs('notebooks', exist_ok=True)
    
    # Define the exact feature structure that matches the app
    feature_columns = [
        'year', 'groundwater_ph', 'ec_groundwater_(µs/cm)', 'hardness_groundwater_(mg/l)',
        'nitrate_groundwater_(mg/l)', 'rainfall_mm', 'soil_ph', 'soil_organic_carbon',
        'soil_nitrogen', 'soil_phosphorus', 'soil_potassium',
        'state_Gujarat', 'state_Karnataka', 'state_Maharashtra', 'state_Tamil Nadu',
        'district_Anantapur', 'district_Bangalore', 'district_Chennai', 'district_Mumbai',
        'crop_Maize', 'crop_Rice', 'crop_Sugarcane', 'crop_Wheat'
    ]
    
    print(f"📊 Creating models with {len(feature_columns)} features")
    
    # Generate training data with the exact feature structure
    np.random.seed(42)
    n_samples = 2000
    
    # Create synthetic data
    data = {
        'state': np.random.choice(['Andhra Pradesh', 'Karnataka', 'Tamil Nadu', 'Maharashtra', 'Gujarat'], n_samples),
        'district': np.random.choice(['Anantapur', 'Bangalore', 'Chennai', 'Mumbai', 'Ahmedabad'], n_samples),
        'crop': np.random.choice(['Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton'], n_samples),
        'year': np.random.choice(range(2015, 2025), n_samples),
        'groundwater_ph': np.random.normal(7.5, 0.5, n_samples),
        'ec_groundwater_(µs/cm)': np.random.normal(1200, 300, n_samples),
        'hardness_groundwater_(mg/l)': np.random.normal(400, 100, n_samples),
        'nitrate_groundwater_(mg/l)': np.random.normal(40, 20, n_samples),
        'rainfall_mm': np.random.normal(650, 150, n_samples),
        'soil_ph': np.random.normal(7.2, 0.4, n_samples),
        'soil_organic_carbon': np.random.normal(0.7, 0.1, n_samples),
        'soil_nitrogen': np.random.normal(200, 50, n_samples),
        'soil_phosphorus': np.random.normal(20, 5, n_samples),
        'soil_potassium': np.random.normal(250, 100, n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # One-hot encode categorical variables (same as in app)
    df_encoded = pd.get_dummies(df, columns=['state', 'district', 'crop'], drop_first=True)
    
    # Create target variable with realistic relationships
    y = (2000 + 
         df_encoded['soil_nitrogen'] * 2 +
         df_encoded['soil_phosphorus'] * 10 +
         df_encoded['soil_potassium'] * 0.5 +
         df_encoded['rainfall_mm'] * 0.5 +
         np.random.normal(0, 200, n_samples))
    
    # Ensure positive yields
    y = np.abs(y)
    
    # Prepare features (exclude target)
    X = df_encoded.copy()
    
    print(f"📊 Generated {len(X)} samples with {X.shape[1]} features")
    print(f"🎯 Target range: {y.min():.1f} - {y.max():.1f}")
    
    # Create Random Forest model
    print("🤖 Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X, y)
    
    with open('notebooks/rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    # Test the model
    pred = rf_model.predict(X[:5])
    print(f"✅ Random Forest model created successfully")
    print(f"   Sample prediction: {pred[0]:.2f}")
    print(f"   Feature count: {X.shape[1]}")
    
    # Create Gradient Boosting model
    print("🤖 Training Gradient Boosting...")
    gbr_model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    gbr_model.fit(X, y)
    
    with open('notebooks/gbr_model.pkl', 'wb') as f:
        pickle.dump(gbr_model, f)
    
    # Test the model
    pred = gbr_model.predict(X[:5])
    print(f"✅ Gradient Boosting model created successfully")
    print(f"   Sample prediction: {pred[0]:.2f}")
    print(f"   Feature count: {X.shape[1]}")
    
    # Save feature info
    feature_info = {
        'feature_columns': list(X.columns),
        'n_features': X.shape[1],
        'feature_types': {
            'categorical': ['state', 'district', 'crop'],
            'numerical': ['year', 'groundwater_ph', 'ec_groundwater_(µs/cm)', 
                         'hardness_groundwater_(mg/l)', 'nitrate_groundwater_(mg/l)', 
                         'rainfall_mm', 'soil_ph', 'soil_organic_carbon', 
                         'soil_nitrogen', 'soil_phosphorus', 'soil_potassium']
        }
    }
    
    with open('notebooks/feature_info.pkl', 'wb') as f:
        pickle.dump(feature_info, f)
    
    print(f"💾 Feature info saved: {len(feature_columns)} features")
    
    # Test prediction consistency
    print("\n🧪 Testing prediction consistency...")
    
    # Create test input (same format as app)
    test_input = {
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
    
    # Create DataFrame and encode
    input_df = pd.DataFrame([test_input])
    input_encoded = pd.get_dummies(input_df, columns=['state', 'district', 'crop'], drop_first=True)
    
    # Align with expected features
    for col in X.columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    input_encoded = input_encoded[X.columns]
    
    # Test both models
    rf_pred = rf_model.predict(input_encoded)[0]
    gbr_pred = gbr_model.predict(input_encoded)[0]
    
    print(f"✅ Random Forest prediction: {rf_pred:.2f}")
    print(f"✅ Gradient Boosting prediction: {gbr_pred:.2f}")
    print(f"📊 Average prediction: {(rf_pred + gbr_pred) / 2:.2f}")
    
    print("\n🎉 Missing models created successfully!")
    print("📁 Files created:")
    print("   - notebooks/rf_model.pkl")
    print("   - notebooks/gbr_model.pkl")
    print("   - notebooks/feature_info.pkl")

if __name__ == "__main__":
    create_missing_models()
