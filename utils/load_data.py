"""Data loading and model utility functions."""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_models():
    """Load all trained models with robust error handling."""
    models = {}
    messages = []
    model_files = {
        'XGBoost': 'notebooks/xgb_model.pkl',
        'Random Forest': 'notebooks/rf_model.pkl',
        'Gradient Boosting': 'notebooks/gbr_model.pkl',
        'CatBoost': 'notebooks/catboost_model.pkl',
        'LightGBM': 'notebooks/lgbm_model.pkl',
        'Ridge': 'notebooks/ridge_model.pkl',
        'MLP': 'notebooks/mlp_model.pkl',
        'Bayesian Ridge': 'notebooks/bayesian_ridge_model.pkl'
    }
    
    for name, file_path in model_files.items():
        try:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    models[name] = pickle.load(f)
                messages.append({"level": "success", "text": f"✅ Loaded {name} model"})
            else:
                messages.append({"level": "warning", "text": f"⚠️ Model {name} not found at {file_path}"})
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            messages.append({"level": "error", "text": f"❌ Corrupted model file {name}: {str(e)}"})
        except Exception as e:
            messages.append({"level": "error", "text": f"❌ Error loading {name}: {str(e)}"})
    
    if not models:
        messages.append({"level": "error", "text": "🚨 No models loaded! Please run the model training notebooks first."})
        messages.append({"level": "info", "text": "📝 To fix this: Run the notebooks or generate demo models."})
    
    return models, messages

def prepare_input_data(input_params, feature_info=None):
    """Prepare input data for model prediction."""
    input_df = pd.DataFrame([input_params])
    
    # One-hot encode categorical variables
    cat_cols = ['state', 'district', 'crop']
    input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
    
    # Handle feature alignment if feature_info provided
    if feature_info and 'feature_columns' in feature_info:
        expected_columns = feature_info['feature_columns']
        for col in expected_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[expected_columns]
    
    return input_encoded

def make_prediction(model, input_data, scaler=None):
    """Make prediction using the model."""
    if scaler:
        input_scaled = scaler.transform(input_data)
        return model.predict(input_scaled)[0]
    return model.predict(input_data)[0]

def get_feature_importance(model, feature_names):
    """Get feature importance from the model if available."""
    if hasattr(model, 'feature_importances_'):
        return dict(zip(feature_names, model.feature_importances_))
    return None

def get_shap_explanation(model, input_data, background_data=None):
    """Get SHAP explanation for the prediction."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        return shap_values
    except ImportError:
        return None

def create_shap_plot(shap_values, feature_names, plot_type='bar'):
    """Create SHAP plot for visualization."""
    try:
        import shap
        if plot_type == 'bar':
            plt.figure()
            shap.summary_plot(shap_values, feature_names=feature_names, plot_type='bar')
            return plt.gcf()
    except ImportError:
        return None
    
def explain_crop_prediction(model, input_data, feature_names, actual_yield=None):
    """Generate comprehensive explanation for crop prediction."""
    prediction = make_prediction(model, input_data)
    
    explanation = {
        'predicted_yield': prediction,
        'actual_yield': actual_yield,
        'feature_importance': get_feature_importance(model, feature_names),
        'shap_values': get_shap_explanation(model, input_data)
    }
    
    return explanation