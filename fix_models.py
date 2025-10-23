#!/usr/bin/env python3
"""
Fix corrupted model files by generating new demo models
"""

import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def create_demo_models():
    """Create demo model files to replace corrupted ones"""
    
    # Create notebooks directory if it doesn't exist
    os.makedirs('notebooks', exist_ok=True)
    
    print("🔧 Creating demo models to replace corrupted files...")
    
    # Generate dummy training data
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
    # Create synthetic features (similar to the real dataset)
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples) * 1000 + 2000  # Yield between 2000-3000
    
    # Add some realistic relationships
    y += X[:, 0] * 500  # First feature has positive impact
    y += X[:, 1] * 300  # Second feature has positive impact
    y -= X[:, 2] * 200  # Third feature has negative impact
    
    # Ensure positive yields
    y = np.abs(y)
    
    print(f"📊 Generated {n_samples} training samples with {n_features} features")
    
    # Create and save models
    models_to_create = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Bayesian Ridge': BayesianRidge(),
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    for name, model in models_to_create.items():
        try:
            print(f"🤖 Training {name}...")
            
            if name == 'MLP':
                # MLP needs scaling
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                model.fit(X_scaled, y)
                
                # Save model with scaler
                model_data = {'model': model, 'scaler': scaler}
                with open(f'notebooks/{name.lower().replace(" ", "_")}_model.pkl', 'wb') as f:
                    pickle.dump(model_data, f)
            else:
                model.fit(X, y)
                with open(f'notebooks/{name.lower().replace(" ", "_")}_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
            
            # Test the model
            if name == 'MLP':
                X_test_scaled = scaler.transform(X[:5])
                pred = model.predict(X_test_scaled)
            else:
                pred = model.predict(X[:5])
            
            print(f"✅ {name} model created and tested successfully")
            print(f"   Sample prediction: {pred[0]:.2f}")
            
        except Exception as e:
            print(f"❌ Error creating {name}: {str(e)}")
    
    # Create XGBoost model (if available)
    try:
        import xgboost as xgb
        print("🤖 Training XGBoost...")
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=5)
        xgb_model.fit(X, y)
        
        with open('notebooks/xgb_model.pkl', 'wb') as f:
            pickle.dump(xgb_model, f)
        
        pred = xgb_model.predict(X[:5])
        print(f"✅ XGBoost model created successfully")
        print(f"   Sample prediction: {pred[0]:.2f}")
        
    except ImportError:
        print("⚠️ XGBoost not available, skipping...")
    
    # Create CatBoost model (if available)
    try:
        import catboost as cb
        print("🤖 Training CatBoost...")
        cat_model = cb.CatBoostRegressor(iterations=100, random_seed=42, verbose=False)
        cat_model.fit(X, y)
        
        with open('notebooks/catboost_model.pkl', 'wb') as f:
            pickle.dump(cat_model, f)
        
        pred = cat_model.predict(X[:5])
        print(f"✅ CatBoost model created successfully")
        print(f"   Sample prediction: {pred[0]:.2f}")
        
    except ImportError:
        print("⚠️ CatBoost not available, skipping...")
    
    # Create LightGBM model (if available)
    try:
        import lightgbm as lgb
        print("🤖 Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        lgb_model.fit(X, y)
        
        with open('notebooks/lgbm_model.pkl', 'wb') as f:
            pickle.dump(lgb_model, f)
        
        pred = lgb_model.predict(X[:5])
        print(f"✅ LightGBM model created successfully")
        print(f"   Sample prediction: {pred[0]:.2f}")
        
    except ImportError:
        print("⚠️ LightGBM not available, skipping...")
    
    print("\n🎉 Model creation complete!")
    print("📁 Created model files:")
    for file in os.listdir('notebooks'):
        if file.endswith('_model.pkl'):
            print(f"   - {file}")

def verify_models():
    """Verify that the created models can be loaded"""
    print("\n🔍 Verifying model files...")
    
    model_files = [
        'rf_model.pkl',
        'gbr_model.pkl', 
        'ridge_model.pkl',
        'bayesian_ridge_model.pkl',
        'mlp_model.pkl'
    ]
    
    for model_file in model_files:
        file_path = f'notebooks/{model_file}'
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
                print(f"✅ {model_file} - OK")
            except Exception as e:
                print(f"❌ {model_file} - Error: {str(e)}")
        else:
            print(f"⚠️ {model_file} - Not found")

if __name__ == "__main__":
    print("🔧 Fixing Corrupted Model Files")
    print("=" * 40)
    
    create_demo_models()
    verify_models()
    
    print("\n🚀 You can now run the Streamlit app!")
    print("Run: streamlit run app.py")
