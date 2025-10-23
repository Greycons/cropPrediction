#!/usr/bin/env python3
"""
Generate demo data for the Streamlit application
This creates sample data files if the notebooks haven't been run yet
"""

import pandas as pd
import numpy as np
import os

def generate_demo_data():
    """Generate demo data for testing the Streamlit app"""
    
    # Create notebooks directory if it doesn't exist
    os.makedirs('notebooks', exist_ok=True)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic agricultural data
    data = {
        'state': np.random.choice(['Andhra Pradesh', 'Karnataka', 'Tamil Nadu', 'Maharashtra', 'Gujarat'], n_samples),
        'district': np.random.choice(['Anantapur', 'Bangalore', 'Chennai', 'Mumbai', 'Ahmedabad'], n_samples),
        'year': np.random.choice(range(2015, 2025), n_samples),
        'crop': np.random.choice(['Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton'], n_samples),
        'groundwater_ph': np.random.normal(7.5, 0.5, n_samples),
        'ec_groundwater_(µs/cm)': np.random.normal(1200, 300, n_samples),
        'hardness_groundwater_(mg/l)': np.random.normal(400, 100, n_samples),
        'nitrate_groundwater_(mg/l)': np.random.normal(40, 20, n_samples),
        'rainfall_mm': np.random.normal(650, 150, n_samples),
        'soil_ph': np.random.normal(7.2, 0.4, n_samples),
        'soil_organic_carbon': np.random.normal(0.7, 0.1, n_samples),
        'soil_nitrogen': np.random.normal(200, 50, n_samples),
        'soil_phosphorus': np.random.normal(20, 5, n_samples),
        'soil_potassium': np.random.normal(250, 100, n_samples),
        'crop_yield': np.random.normal(3000, 500, n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure positive values for yield
    df['crop_yield'] = np.abs(df['crop_yield'])
    
    # Clip values to realistic ranges
    df['groundwater_ph'] = df['groundwater_ph'].clip(6.4, 8.8)
    df['soil_ph'] = df['soil_ph'].clip(6.0, 8.5)
    df['rainfall_mm'] = df['rainfall_mm'].clip(400, 1000)
    df['soil_organic_carbon'] = df['soil_organic_carbon'].clip(0.4, 1.1)
    df['soil_nitrogen'] = df['soil_nitrogen'].clip(161, 416)
    df['soil_phosphorus'] = df['soil_phosphorus'].clip(11, 33)
    df['soil_potassium'] = df['soil_potassium'].clip(100, 500)
    
    # Save as both cleaned and normalized datasets
    df.to_csv('notebooks/Cleaned_Dataset.csv', index=False)
    
    # Create normalized version
    df_normalized = df.copy()
    numeric_cols = ['groundwater_ph', 'ec_groundwater_(µs/cm)', 'hardness_groundwater_(mg/l)',
                   'nitrate_groundwater_(mg/l)', 'rainfall_mm', 'soil_ph',
                   'soil_organic_carbon', 'soil_nitrogen', 'soil_phosphorus', 
                   'soil_potassium', 'crop_yield']
    
    # Simple min-max normalization
    for col in numeric_cols:
        if col in df_normalized.columns:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    
    df_normalized.to_csv('notebooks/Normalized_Dataset.csv', index=False)
    
    print("✅ Demo data generated successfully!")
    print(f"📊 Created {len(df)} records")
    print("📁 Files created:")
    print("   - notebooks/Cleaned_Dataset.csv")
    print("   - notebooks/Normalized_Dataset.csv")
    print("\n🚀 You can now run the Streamlit app!")

def create_demo_models():
    """Create demo model files (placeholder)"""
    import pickle
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    import numpy as np
    
    # Create dummy models for demo
    np.random.seed(42)
    X_dummy = np.random.rand(100, 10)
    y_dummy = np.random.rand(100)
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
    rf_model.fit(X_dummy, y_dummy)
    
    # Ridge Regression
    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model.fit(X_dummy, y_dummy)
    
    # Save models
    with open('notebooks/rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    with open('notebooks/ridge_model.pkl', 'wb') as f:
        pickle.dump(ridge_model, f)
    
    print("✅ Demo models created!")
    print("📁 Files created:")
    print("   - notebooks/rf_model.pkl")
    print("   - notebooks/ridge_model.pkl")

if __name__ == "__main__":
    print("🌾 Generating demo data for Crop Prediction AI...")
    print("=" * 50)
    
    generate_demo_data()
    create_demo_models()
    
    print("\n🎉 Demo setup complete!")
    print("Run the app with: python run_app.py")
