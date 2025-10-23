import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def load_models():
    """Load all trained models with error handling"""
    models = {}
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
            with open(file_path, 'rb') as f:
                models[name] = pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: Model {name} not found at {file_path}")
        except Exception as e:
            print(f"Error loading {name}: {str(e)}")
    
    return models

def prepare_input_data(input_dict, training_columns):
    """Prepare input data for prediction"""
    # Create DataFrame from input
    input_df = pd.DataFrame([input_dict])
    
    # One-hot encode categorical variables
    categorical_cols = ['state', 'district', 'crop']
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    
    # Ensure all required columns are present
    for col in training_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Reorder columns to match training data
    input_encoded = input_encoded[training_columns]
    
    return input_encoded

def make_prediction(input_data, models, model_name=None):
    """Make prediction using specified model or ensemble"""
    predictions = {}
    
    if model_name and model_name in models:
        # Single model prediction
        try:
            model = models[model_name]
            if isinstance(model, dict) and 'model' in model:
                # Handle models with scalers
                scaler = model.get('scaler')
                model_obj = model['model']
                if scaler:
                    input_scaled = scaler.transform(input_data)
                    pred = model_obj.predict(input_scaled)[0]
                else:
                    pred = model_obj.predict(input_data)[0]
            else:
                pred = model.predict(input_data)[0]
            return pred
        except Exception as e:
            print(f"Error with {model_name}: {str(e)}")
            return None
    else:
        # Ensemble prediction
        for name, model in models.items():
            try:
                if isinstance(model, dict) and 'model' in model:
                    # Handle models with scalers
                    scaler = model.get('scaler')
                    model_obj = model['model']
                    if scaler:
                        input_scaled = scaler.transform(input_data)
                        pred = model_obj.predict(input_scaled)[0]
                    else:
                        pred = model_obj.predict(input_data)[0]
                else:
                    pred = model.predict(input_data)[0]
                predictions[name] = pred
            except Exception as e:
                print(f"Error with {name}: {str(e)}")
        
        return predictions

def get_shap_explanation(model, input_data, model_name):
    """Get SHAP explanation for model prediction"""
    try:
        if model_name in ['Random Forest', 'XGBoost', 'Gradient Boosting', 'CatBoost', 'LightGBM']:
            # Tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)
            return shap_values
        elif model_name in ['Ridge', 'Bayesian Ridge']:
            # Linear models
            if model_name == 'Ridge':
                explainer = shap.LinearExplainer(model, input_data, feature_perturbation="interventional")
            else:
                explainer = shap.Explainer(model, input_data)
            shap_values = explainer.shap_values(input_data)
            return shap_values
        else:
            # Other models - use KernelExplainer (slower)
            background = input_data.sample(min(100, len(input_data)), random_state=42)
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(input_data.iloc[:50])  # Limit for performance
            return shap_values
    except Exception as e:
        print(f"SHAP explanation error: {str(e)}")
        return None

def create_shap_plot(shap_values, input_data, feature_names=None):
    """Create SHAP visualization"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, input_data, feature_names=feature_names, show=False)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"SHAP plot error: {str(e)}")
        return None

def get_location_recommendations(df, state, district, year=None, crop_preference=None):
    """Get crop recommendations based on location"""
    # Filter data for location
    location_data = df[(df['state'] == state) & (df['district'] == district)]
    
    if len(location_data) == 0:
        return None
    
    # Filter by year if specified
    if year:
        location_data = location_data[location_data['year'] == year]
    
    if len(location_data) == 0:
        return None
    
    # Get crop performance
    crop_performance = location_data.groupby('crop')['crop_yield'].agg(['mean', 'count', 'std']).reset_index()
    crop_performance = crop_performance.sort_values('mean', ascending=False)
    
    # Filter by crop preference if specified
    if crop_preference:
        crop_performance = crop_performance[crop_performance['crop'].isin(crop_preference)]
    
    return crop_performance

def calculate_confidence_interval(predictions):
    """Calculate confidence interval for ensemble predictions"""
    if not predictions:
        return None, None, None
    
    pred_values = list(predictions.values())
    mean_pred = np.mean(pred_values)
    std_pred = np.std(pred_values)
    
    # 95% confidence interval
    ci_lower = mean_pred - 1.96 * std_pred
    ci_upper = mean_pred + 1.96 * std_pred
    
    return mean_pred, ci_lower, ci_upper

def get_feature_importance(model, input_data, model_name):
    """Get feature importance for model"""
    try:
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = model.feature_importances_
            return importance
        elif hasattr(model, 'coef_'):
            # Linear models
            importance = np.abs(model.coef_)
            return importance
        else:
            return None
    except Exception as e:
        print(f"Feature importance error: {str(e)}")
        return None

def validate_input_parameters(params):
    """Validate input parameters for prediction"""
    errors = []
    
    # Check required parameters
    required_params = ['soil_ph', 'soil_organic_carbon', 'soil_nitrogen', 
                      'soil_phosphorus', 'soil_potassium', 'rainfall_mm']
    
    for param in required_params:
        if param not in params:
            errors.append(f"Missing required parameter: {param}")
        elif not isinstance(params[param], (int, float)):
            errors.append(f"Invalid type for {param}: must be numeric")
    
    # Check parameter ranges
    ranges = {
        'soil_ph': (4.5, 9.0),
        'soil_organic_carbon': (0.4, 1.1),
        'soil_nitrogen': (161, 416),
        'soil_phosphorus': (11, 33),
        'soil_potassium': (100, 500),
        'rainfall_mm': (400, 1000)
    }
    
    for param, (min_val, max_val) in ranges.items():
        if param in params:
            if not (min_val <= params[param] <= max_val):
                errors.append(f"{param} out of range: {min_val}-{max_val}")
    
    return errors

def format_prediction_result(prediction, confidence_level="High"):
    """Format prediction result for display"""
    if isinstance(prediction, dict):
        # Ensemble prediction
        mean_pred = np.mean(list(prediction.values()))
        return {
            'predicted_yield': mean_pred,
            'confidence': confidence_level,
            'model_count': len(prediction),
            'individual_predictions': prediction
        }
    else:
        # Single model prediction
        return {
            'predicted_yield': prediction,
            'confidence': confidence_level,
            'model_count': 1,
            'individual_predictions': {'Single Model': prediction}
        }
