#!/usr/bin/env python3
"""
Crop Prediction AI - Easy Startup Script
This script handles everything needed to run the Streamlit application
"""

import os
import sys
import subprocess

def main():
    print("🌾 Crop Prediction AI - Easy Startup")
    print("=" * 50)
    
    # Check if data files exist
    data_files = [
        'notebooks/Cleaned_Dataset.csv',
        'notebooks/Normalized_Dataset.csv'
    ]
    
    missing_files = [f for f in data_files if not os.path.exists(f)]
    
    if missing_files:
        print("📊 Data files not found. Generating demo data...")
        try:
            subprocess.run([sys.executable, "generate_demo_data.py"], check=True)
            print("✅ Demo data generated successfully!")
        except subprocess.CalledProcessError:
            print("❌ Error generating demo data. Please run manually:")
            print("   python generate_demo_data.py")
            return
        except FileNotFoundError:
            print("❌ generate_demo_data.py not found. Creating basic data...")
            create_basic_data()
    
    # Check if models exist and are valid
    model_files = [
        'notebooks/rf_model.pkl',
        'notebooks/ridge_model.pkl'
    ]
    
    missing_models = [f for f in model_files if not os.path.exists(f)]
    corrupted_models = []
    
    # Check for corrupted models
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                import pickle
                with open(model_file, 'rb') as f:
                    pickle.load(f)
            except (pickle.UnpicklingError, EOFError, ValueError):
                corrupted_models.append(model_file)
    
    if missing_models or corrupted_models:
        print("🤖 Model files missing or corrupted. Creating new models...")
        try:
            subprocess.run([sys.executable, "fix_models.py"], check=True)
        except:
            print("⚠️ Could not run fix_models.py, creating basic models...")
            create_basic_models()
    
    # Install requirements if needed
    print("📦 Checking dependencies...")
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        print("✅ All required packages are available!")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install requirements: pip install -r requirements_streamlit.txt")
        return
    
    # Run the Streamlit app
    print("\n🚀 Starting Streamlit application...")
    print("The app will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
    except Exception as e:
        print(f"❌ Error: {e}")

def create_basic_data():
    """Create basic data files if demo script fails"""
    import pandas as pd
    import numpy as np
    
    os.makedirs('notebooks', exist_ok=True)
    
    # Create minimal dataset
    data = {
        'state': ['Andhra Pradesh', 'Karnataka', 'Tamil Nadu'] * 10,
        'district': ['Anantapur', 'Bangalore', 'Chennai'] * 10,
        'year': [2020, 2021, 2022] * 10,
        'crop': ['Rice', 'Wheat', 'Maize'] * 10,
        'groundwater_ph': np.random.normal(7.5, 0.5, 30),
        'ec_groundwater_(µs/cm)': np.random.normal(1200, 300, 30),
        'hardness_groundwater_(mg/l)': np.random.normal(400, 100, 30),
        'nitrate_groundwater_(mg/l)': np.random.normal(40, 20, 30),
        'rainfall_mm': np.random.normal(650, 150, 30),
        'soil_ph': np.random.normal(7.2, 0.4, 30),
        'soil_organic_carbon': np.random.normal(0.7, 0.1, 30),
        'soil_nitrogen': np.random.normal(200, 50, 30),
        'soil_phosphorus': np.random.normal(20, 5, 30),
        'soil_potassium': np.random.normal(250, 100, 30),
        'crop_yield': np.random.normal(3000, 500, 30)
    }
    
    df = pd.DataFrame(data)
    df.to_csv('notebooks/Cleaned_Dataset.csv', index=False)
    df.to_csv('notebooks/Normalized_Dataset.csv', index=False)
    print("✅ Basic data files created!")

def create_basic_models():
    """Create basic model files if demo script fails"""
    import pickle
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    import numpy as np
    
    os.makedirs('notebooks', exist_ok=True)
    
    # Create dummy models
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    rf.fit(X, y)
    
    # Ridge
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X, y)
    
    # Save models
    with open('notebooks/rf_model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    
    with open('notebooks/ridge_model.pkl', 'wb') as f:
        pickle.dump(ridge, f)
    
    print("✅ Basic model files created!")

if __name__ == "__main__":
    main()
