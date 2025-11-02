import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
import os
import re
import json
import matplotlib.pyplot as plt
from utils import (load_models, prepare_input_data, make_prediction, get_feature_importance,
                  get_shap_explanation, create_shap_plot, explain_crop_prediction,
                  get_crop_specific_advice, get_fertilizer_recommendations)
warnings.filterwarnings('ignore')

# Try to import dotenv for .env file support
try:
    from dotenv import load_dotenv
    # Load .env file
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Try to import and configure Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    GEMINI_REASON = None
    
    # Configure Gemini API - Check both environment variable and .env file
    api_key = os.environ.get('GOOGLE_API_KEY')
    if api_key:
        # Configure with simplified settings
        genai.configure(api_key=api_key)
        
        # Verify the configuration by testing model creation with model list
        try:
            # List available models first
            available_models = [model.name for model in genai.list_models()]
            # Try to find best available model (Gemini 2.5 Flash variants only)
            model_name = None
            candidates = [
                'gemini-2.5-flash',
                'gemini-2.5-flash-latest',
                'gemini-2.5-flash-preview-09-2025',
                'gemini-2.5-flash-preview-05-20',
                'gemini-2.5-flash-lite',
                'gemini-2.5-flash-lite-preview-09-2025',
                'gemini-flash-latest',
                'gemini-flash'
            ]
            for candidate in candidates:
                if f"models/{candidate}" in available_models:
                    model_name = candidate
                    break
            
            if not model_name:
                GEMINI_AVAILABLE = False
                GEMINI_REASON = f"No supported Gemini model found. Available models: {', '.join(available_models)}"
            else:
                # Try a test generation with the found model
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("Test connection")
                if not response:
                    GEMINI_AVAILABLE = False
                    GEMINI_REASON = f"Could not generate response from Gemini model {model_name}"
        except Exception as e:
            GEMINI_AVAILABLE = False
            GEMINI_REASON = f"Error connecting to Gemini: {str(e)}"
    else:
        GEMINI_AVAILABLE = False
        if not DOTENV_AVAILABLE:
            GEMINI_REASON = "GOOGLE_API_KEY not found. Install python-dotenv and create .env file"
        else:
            GEMINI_REASON = "GOOGLE_API_KEY not found in environment or .env file"
except ImportError:
    GEMINI_AVAILABLE = False
    GEMINI_REASON = "google-generativeai package not installed. Run: pip install google-generativeai"
    genai = None

# Try to import shap, but handle if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("SHAP not available. Some features will be limited.")

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Helper functions
def generate_ai_response(prompt, location=None, soil_params=None, df_clean=None):
    """Generate AI response using Google's Gemini Pro with context from ML models and dataset.

    The function builds a short, structured context from provided location/soil/dataset
    information and sends a single generation request to the Gemini Pro model. It
    returns the model text on success or a friendly error message on failure.
    """

    if not GEMINI_AVAILABLE:
        return (
            "🔑 **Gemini API Setup Required**\n\n"
            "Please set up Google's Gemini API:\n"
            "1. Get an API key from Google AI Studio\n"
            "2. Add to your `.env` file:\n"
            "   GOOGLE_API_KEY=your_key_here\n"
            "3. Install required package: `pip install google-generativeai`\n"
            "4. Restart the Streamlit app\n\n"
            f"Current Status: {GEMINI_REASON or 'Unknown error'}"
        )

    # Build a concise context
    context_parts = [
        "You are an expert agricultural AI assistant helping farmers with crop predictions, soil health, and farming best practices."
        " Be concise, actionable, and use emojis where appropriate."
    ]

    if location and df_clean is not None:
        try:
            location_data = df_clean[
                (df_clean['state'] == location['state']) & (df_clean['district'] == location['district'])
            ]
            if len(location_data) > 0:
                avg_yield = location_data['crop_yield'].mean()
                top_crops = location_data.groupby('crop')['crop_yield'].mean().sort_values(ascending=False).head(3)
                context_parts.append(f"📍 Location: {location['state']}, {location['district']}")
                context_parts.append(f"- Avg yield: {avg_yield:.1f} kg/acre")
                context_parts.append(f"- Top crops: {', '.join(top_crops.index.tolist())}")
            else:
                context_parts.append(f"📍 Location: {location['state']}, {location['district']} (no historical rows)")
        except Exception:
            # Non-fatal — continue without location-specific stats
            context_parts.append(f"📍 Location: {location.get('state')}, {location.get('district')}")

    if soil_params:
        soil_lines = ["🌱 Soil Parameters:"]
        for param, value in soil_params.items():
            pretty = param.replace('soil_', '').replace('_', ' ').title()
            soil_lines.append(f"- {pretty}: {value}")
        context_parts.append('\n'.join(soil_lines))

    if df_clean is not None:
        context_parts.append(f"📊 Historical records: {len(df_clean)} rows; {df_clean['crop'].nunique()} crops; {df_clean['state'].nunique()} states")

    # Final prompt
    full_prompt = "\n\n".join(context_parts) + "\n\nUser Question: " + prompt

    # Send to Gemini with best available model
    try:
        # Use same model discovery logic as in configuration
        available_models = [model.name for model in genai.list_models()]
        model_name = None
        # Prefer Gemini 2.5 Flash family only
        candidates = [
            'gemini-2.5-flash',
            'gemini-2.5-flash-latest',
            'gemini-2.5-flash-preview-09-2025',
            'gemini-2.5-flash-preview-05-20',
            'gemini-2.5-flash-lite',
            'gemini-2.5-flash-lite-preview-09-2025',
            'gemini-flash-latest',
            'gemini-flash'
        ]
        for candidate in candidates:
            if f"models/{candidate}" in available_models:
                model_name = candidate
                break
                
        if not model_name:
            return "No supported Gemini model currently available"
            
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            full_prompt,
            generation_config={
                'temperature': 0.7,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 800,
            }
        )        # Some SDK responses put text in .text, others in .content; handle both
        if hasattr(response, 'text') and response.text:
            return response.text
        if hasattr(response, 'content'):
            # content may be a dict or list depending on SDK version
            try:
                if isinstance(response.content, dict):
                    return response.content.get('output', '') or str(response.content)
                return str(response.content)
            except Exception:
                return str(response.content)

        return str(response)

    except Exception as e:
        return (
            "🤖 **AI Assistant Error**\n\n"
            "I encountered an error while processing your request:\n"
            f"{str(e)}\n\n"
            "Please try again or check:\n"
            "- Your internet connection\n"
            "- API key configuration\n"
            "- Gemini API status"
        )

def get_crop_recommendations_ml_only(input_data, models, df_clean, state_override=None, district_override=None):
    """Get crop recommendations using only ML models.
    If state_override and district_override are provided, use that location for all crops.
    Returns (crop_scores, crop_predictions) where crop_scores is a sorted list of tuples
    and crop_predictions is a dict of per-crop prediction details.
    """
    if df_clean is None or not models:
        return None

    available_crops = df_clean['crop'].unique()
    crop_predictions = {}

    for crop in available_crops:
        if state_override is not None and district_override is not None:
            state = state_override
            district = district_override
        else:
            crop_data = df_clean[df_clean['crop'] == crop]
            if len(crop_data) > 0:
                location_counts = crop_data.groupby(['state', 'district']).size().reset_index(name='count')
                most_common_location = location_counts.loc[location_counts['count'].idxmax()]
                state = most_common_location['state']
                district = most_common_location['district']
            else:
                state = df_clean['state'].iloc[0]
                district = df_clean['district'].iloc[0]

        test_input = input_data.copy()
        test_input['state'] = state
        test_input['district'] = district
        test_input['crop'] = crop

        input_df = pd.DataFrame([test_input])
        input_encoded = pd.get_dummies(input_df, columns=['state', 'district', 'crop'], drop_first=True)

        feature_info = None
        try:
            with open('notebooks/feature_info.pkl', 'rb') as f:
                feature_info = pickle.load(f)
        except Exception:
            feature_info = None

        if feature_info:
            expected_columns = feature_info.get('feature_columns', [])
            for col in expected_columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[expected_columns]

        crop_model_predictions = []
        working_models = 0

        for model_name, model in models.items():
            if model_name != 'Random Forest':
                continue
            try:
                if isinstance(model, dict) and 'model' in model:
                    scaler = model.get('scaler')
                    model_obj = model['model']
                    if scaler and model_obj:
                        input_scaled = scaler.transform(input_encoded)
                        pred = model_obj.predict(input_scaled)[0]
                    else:
                        pred = model_obj.predict(input_encoded)[0]
                else:
                    pred = model.predict(input_encoded)[0]
                crop_model_predictions.append(pred)
                working_models += 1
            except Exception:
                continue

        if crop_model_predictions:
            avg_prediction = np.mean(crop_model_predictions)
            crop_predictions[crop] = {
                'avg_yield': avg_prediction,
                'model_count': working_models,
                'individual_predictions': crop_model_predictions,
                'location_used': f"{state}, {district}"
            }

    if not crop_predictions:
        return None

    yields = [data['avg_yield'] for data in crop_predictions.values()]
    min_yield = min(yields)
    max_yield = max(yields)

    crop_scores = []
    for crop, data in crop_predictions.items():
        if max_yield > min_yield:
            score = ((data['avg_yield'] - min_yield) / (max_yield - min_yield)) * 100
        else:
            score = 50

        if data['model_count'] >= 6:
            confidence = "High"
        elif data['model_count'] >= 4:
            confidence = "Medium"
        else:
            confidence = "Low"

        crop_scores.append((crop, score, confidence, data['avg_yield'], data['model_count']))

    crop_scores.sort(key=lambda x: x[1], reverse=True)
    return crop_scores, crop_predictions

def get_crop_recommendations_combined(input_data, models, df_clean):
    """Combine ML and historical recommendations (weighted)."""
    ml_result = get_crop_recommendations_ml_only(input_data, models, df_clean)
    if ml_result is None:
        return None
    ml_recommendations, _ = ml_result

    historical_recommendations = get_crop_recommendations_historical(input_data, df_clean)
    if historical_recommendations is None:
        return ml_recommendations

    combined_scores = []
    for crop, ml_score, ml_conf, ml_yield, ml_count in ml_recommendations:
        hist = next((h for h in historical_recommendations if h[0] == crop), None)
        if hist:
            hist_score, hist_yield = hist[1], hist[3]
            combined_score = (ml_score * 0.7) + (hist_score * 0.3)
            combined_yield = (ml_yield * 0.7) + (hist_yield * 0.3)
        else:
            combined_score = ml_score * 0.8
            combined_yield = ml_yield

        if ml_count >= 6 and hist:
            confidence = "High"
        elif ml_count >= 4:
            confidence = "Medium"
        else:
            confidence = "Low"

        combined_scores.append((crop, combined_score, confidence, combined_yield, ml_count))

    combined_scores.sort(key=lambda x: x[1], reverse=True)
    return combined_scores

def get_crop_recommendations_historical(input_data, df_clean):
    """Get crop recommendations using historical data analysis."""
    if df_clean is None:
        return None

    filtered_data = df_clean.copy()
    tolerance = 0.1
    try:
        filtered_data = filtered_data[
            (filtered_data['soil_ph'] >= input_data['soil_ph'] * (1 - tolerance)) &
            (filtered_data['soil_ph'] <= input_data['soil_ph'] * (1 + tolerance)) &
            (filtered_data['rainfall_mm'] >= input_data['rainfall_mm'] * (1 - tolerance)) &
            (filtered_data['rainfall_mm'] <= input_data['rainfall_mm'] * (1 + tolerance)) &
            (filtered_data['soil_nitrogen'] >= input_data['soil_nitrogen'] * (1 - tolerance)) &
            (filtered_data['soil_nitrogen'] <= input_data['soil_nitrogen'] * (1 + tolerance))
        ]
    except Exception:
        # If filtering fails because columns missing, fallback to full data
        filtered_data = df_clean

    if len(filtered_data) == 0:
        filtered_data = df_clean

    crop_performance = filtered_data.groupby('crop')['crop_yield'].agg(['mean', 'count']).reset_index()
    crop_performance = crop_performance.sort_values('mean', ascending=False)

    yields = crop_performance['mean'].values
    min_yield = float(min(yields)) if len(yields) > 0 else 0.0
    max_yield = float(max(yields)) if len(yields) > 0 else 0.0

    historical_scores = []
    for _, row in crop_performance.iterrows():
        crop = row['crop']
        avg_yield = row['mean']
        count = int(row['count'])
        if max_yield > min_yield:
            score = ((avg_yield - min_yield) / (max_yield - min_yield)) * 100
        else:
            score = 50

        if count >= 20:
            confidence = "High"
        elif count >= 10:
            confidence = "Medium"
        else:
            confidence = "Low"

        historical_scores.append((crop, score, confidence, avg_yield, count))

    return historical_scores
    # Test each crop with the given parameters
    for crop in available_crops:
        # Use explicit location if provided, otherwise find the most common location for this crop
        if state_override is not None and district_override is not None:
            state = state_override
            district = district_override
        else:
            crop_data = df_clean[df_clean['crop'] == crop]
            if len(crop_data) > 0:
                # Get the most frequent state-district combination for this crop
                location_counts = crop_data.groupby(['state', 'district']).size().reset_index(name='count')
                most_common_location = location_counts.loc[location_counts['count'].idxmax()]
                state = most_common_location['state']
                district = most_common_location['district']
            else:
                # Fallback to first available location
                state = df_clean['state'].iloc[0]
                district = df_clean['district'].iloc[0]
        
        # Create input data with this crop and its most common location
        test_input = input_data.copy()
        test_input['state'] = state
        test_input['district'] = district
        test_input['crop'] = crop
        
        # Create DataFrame
        input_df = pd.DataFrame([test_input])
        
        # One-hot encode categorical variables
        input_encoded = pd.get_dummies(input_df, columns=['state', 'district', 'crop'], drop_first=True)
        
        # Load feature info if available
        feature_info = None
        try:
            with open('notebooks/feature_info.pkl', 'rb') as f:
                feature_info = pickle.load(f)
        except:
            pass
        
        if feature_info:
            # Use the exact feature columns from training
            expected_columns = feature_info['feature_columns']
            
            # Add missing columns with 0
            for col in expected_columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            # Reorder columns to match training data exactly
            input_encoded = input_encoded[expected_columns]
        
        # Get predictions from all models for this crop
        crop_model_predictions = []
        working_models = 0
        
        for model_name, model in models.items():
            # Use only Random Forest model
            if model_name != 'Random Forest':
                continue
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
                
                crop_model_predictions.append(pred)
                working_models += 1
            except Exception as e:
                continue
        
        if crop_model_predictions:
            # Calculate average prediction for this crop
            avg_prediction = np.mean(crop_model_predictions)
            crop_predictions[crop] = {
                'avg_yield': avg_prediction,
                'model_count': working_models,
                'individual_predictions': crop_model_predictions,
                'location_used': f"{state}, {district}"
            }
    
    if not crop_predictions:
        return None
    
    # Calculate ML scores based on yield predictions
    yields = [data['avg_yield'] for data in crop_predictions.values()]
    min_yield = min(yields)
    max_yield = max(yields)
    
    # Calculate scores and confidence
    crop_scores = []
    for crop, data in crop_predictions.items():
        # Normalize yield to 0-100 score
        if max_yield > min_yield:
            score = ((data['avg_yield'] - min_yield) / (max_yield - min_yield)) * 100
        else:
            score = 50  # Default score if all yields are same
        
        # Confidence based on number of working models
        if data['model_count'] >= 6:
            confidence = "High"
        elif data['model_count'] >= 4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        crop_scores.append((crop, score, confidence, data['avg_yield'], data['model_count']))
    
    # Sort by score (descending)
    crop_scores.sort(key=lambda x: x[1], reverse=True)
    
    return crop_scores, crop_predictions

def get_crop_recommendations_combined(input_data, models, df_clean):
    """Get crop recommendations using both historical data and ML models"""
    
    # Get ML recommendations
    ml_recommendations, _ = get_crop_recommendations_ml_only(input_data, models, df_clean)
    
    if ml_recommendations is None:
        return None
    
    # Get historical recommendations
    historical_recommendations = get_crop_recommendations_historical(input_data, df_clean)
    
    if historical_recommendations is None:
        return ml_recommendations
    
    # Combine both methods (weighted average)
    combined_scores = []
    
    for crop, ml_score, ml_conf, ml_yield, ml_count in ml_recommendations:
        # Find historical data for this crop
        hist_data = next((item for item in historical_recommendations if item[0] == crop), None)
        
        if hist_data:
            hist_score, hist_yield = hist_data[1], hist_data[3]  # hist_data[3] is avg_yield
            # Weighted combination: 70% ML, 30% Historical
            combined_score = (ml_score * 0.7) + (hist_score * 0.3)
            combined_yield = (ml_yield * 0.7) + (hist_yield * 0.3)
        else:
            combined_score = ml_score * 0.8  # Reduce score if no historical data
            combined_yield = ml_yield
        
        # Determine confidence
        if ml_count >= 6 and hist_data:
            confidence = "High"
        elif ml_count >= 4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        combined_scores.append((crop, combined_score, confidence, combined_yield, ml_count))
    
    # Sort by combined score
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    
    return combined_scores

def get_crop_recommendations_historical(input_data, df_clean):
    """Get crop recommendations using only historical data analysis"""
    
    if df_clean is None:
        return None
    
    # Filter data based on input parameters (with some tolerance)
    filtered_data = df_clean.copy()
    
    # Apply filters with tolerance ranges
    tolerance = 0.1  # 10% tolerance
    
    filtered_data = filtered_data[
        (filtered_data['soil_ph'] >= input_data['soil_ph'] * (1 - tolerance)) &
        (filtered_data['soil_ph'] <= input_data['soil_ph'] * (1 + tolerance)) &
        (filtered_data['rainfall_mm'] >= input_data['rainfall_mm'] * (1 - tolerance)) &
        (filtered_data['rainfall_mm'] <= input_data['rainfall_mm'] * (1 + tolerance)) &
        (filtered_data['soil_nitrogen'] >= input_data['soil_nitrogen'] * (1 - tolerance)) &
        (filtered_data['soil_nitrogen'] <= input_data['soil_nitrogen'] * (1 + tolerance))
    ]
    
    if len(filtered_data) == 0:
        # If no matching data, use all data
        filtered_data = df_clean
    
    # Get crop performance
    crop_performance = filtered_data.groupby('crop')['crop_yield'].agg(['mean', 'count']).reset_index()
    crop_performance = crop_performance.sort_values('mean', ascending=False)
    
    # Calculate scores
    yields = crop_performance['mean'].values
    min_yield = min(yields)
    max_yield = max(yields)
    
    historical_scores = []
    for _, row in crop_performance.iterrows():
        crop = row['crop']
        avg_yield = row['mean']
        count = row['count']
        
        # Normalize yield to 0-100 score
        if max_yield > min_yield:
            score = ((avg_yield - min_yield) / (max_yield - min_yield)) * 100
        else:
            score = 50
        
        # Confidence based on data count
        if count >= 20:
            confidence = "High"
        elif count >= 10:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        historical_scores.append((crop, score, confidence, avg_yield, count))
    
    return historical_scores

# Page configuration
st.set_page_config(
    page_title="Crop Prediction AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Global text color - white for most elements */
    body, .main, .stApp {
        color: #ffffff !important;
    }
    
    /* Main header */
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Cards with white background - keep black text */
    .metric-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        color: #000000 !important;
    }
    .metric-card h3, .metric-card p {
        color: #000000 !important;
    }
    
    .prediction-card {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #2E8B57;
        margin: 1rem 0;
        color: #000000 !important;
    }
    .prediction-card h4, .prediction-card p {
        color: #000000 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f8ff;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
        color: #000000 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E8B57;
        color: white !important;
    }
    
    /* Input textboxes - keep black text on white background */
    .stTextInput input, .stTextArea textarea, .stSelectbox select, 
    .stSlider input, .stNumberInput input, .stDateInput input {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Text in input containers */
    .stTextInput label, .stTextArea label, .stSelectbox label, 
    .stSlider label, .stNumberInput label, .stDateInput label {
        color: #ffffff !important;
    }
    
    /* All other text - make white */
    .stMarkdown, .stMarkdown * {
        color: #ffffff !important;
    }
    
    /* Headers and text */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    p, span, div {
        color: #ffffff !important;
    }
    
    /* Sidebar text */
    .css-1d391kg, .css-1d391kg * {
        color: #ffffff !important;
    }
    
    /* Metrics and values */
    .metric-container, .metric-container * {
        color: #ffffff !important;
    }
    
    /* Buttons - keep original colors */
    .stButton button {
        color: #ffffff !important;
    }
    
    /* Alerts and info boxes - set text to white */
    .stAlert, .stSuccess, .stWarning, .stError, .stInfo {
        color: #ffffff !important;
    }
    .stAlert *, .stSuccess *, .stWarning *, .stError *, .stInfo * {
        color: #ffffff !important;
    }
    
    /* DataFrames and tables - keep black text for readability */
    .stDataFrame, .stDataFrame * {
        color: #000000 !important;
    }
    
    /* Charts and plots - keep original colors */
    .plotly, .plotly * {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_data
def load_models():
    """Load all trained models with robust error handling"""
    import os
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

@st.cache_data
def load_data():
    """Load datasets"""
    try:
        # Load normalized dataset for predictions
        df = pd.read_csv('notebooks/Normalized_Dataset.csv')
        
        # Load cleaned dataset for analytics
        df_clean = pd.read_csv('notebooks/Cleaned_Dataset.csv')
        
        return df, df_clean
    except FileNotFoundError:
        st.error("Dataset files not found. Please ensure the notebooks have been run to generate the required CSV files.")
        return None, None

@st.cache_data
def load_model_metrics_file():
    """Load precomputed model metrics from notebooks/model_metrics.json if present."""
    import json
    paths = [
        'notebooks/model_metrics.json',
        'notebooks/models_metrics.json'
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    data = json.load(f)
                    # Expecting dict of model_name -> metrics
                    if isinstance(data, dict):
                        return data
            except Exception:
                continue
    return None

@st.cache_data
def evaluate_selected_model(_models, df_clean, selected_model, sample_size=5000):
    """Compute metrics for only the selected model; optionally sample rows for speed."""
    if df_clean is None or not _models or selected_model not in _models:
        return None
    try:
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    except Exception:
        return None
    models = _models
    data = df_clean.dropna(subset=['crop_yield']).copy()
    if len(data) == 0:
        return None
    if sample_size and len(data) > sample_size:
        data = data.sample(sample_size, random_state=42)
    base_cols = [
        'groundwater_ph', 'ec_groundwater_(µs/cm)', 'hardness_groundwater_(mg/l)',
        'nitrate_groundwater_(mg/l)', 'rainfall_mm', 'soil_ph',
        'soil_organic_carbon', 'soil_nitrogen', 'soil_phosphorus', 'soil_potassium',
        'state', 'district', 'crop'
    ]
    available_cols = [c for c in base_cols if c in data.columns]
    if not available_cols:
        return None
    X_raw = data[available_cols].copy()
    y = data['crop_yield'].values
    cat_cols = [c for c in ['state', 'district', 'crop'] if c in X_raw.columns]
    if cat_cols:
        X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
    else:
        X_encoded = X_raw
    feature_names_ref = None
    try:
        with open('notebooks/feature_info.pkl', 'rb') as f:
            fi = pickle.load(f)
            feature_names_ref = fi.get('feature_columns')
    except Exception:
        pass
    if feature_names_ref is not None:
        for col in feature_names_ref:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        X_encoded = X_encoded[feature_names_ref]
    model = models[selected_model]
    # Handle both wrapped and direct model formats
    if isinstance(model, dict) and 'model' in model:
        core_model = model['model']
        scaler = model.get('scaler')
    else:
        core_model = model
        scaler = None
    X_eval = X_encoded.values
    if scaler is not None:
        try:
            X_eval = scaler.transform(X_eval)
        except Exception:
            pass
    try:
        y_pred = core_model.predict(X_eval)
        mae = float(mean_absolute_error(y, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        with np.errstate(divide='ignore', invalid='ignore'):
            mape_arr = np.abs((y - y_pred) / np.where(y == 0, np.nan, y))
        mape = float(np.nanmean(mape_arr) * 100)
        r2 = float(r2_score(y, y_pred))
        return {'r2': r2, 'mae': mae, 'rmse': rmse, 'mape': mape, 'n_samples': int(len(y))}
    except Exception as e:
        st.error(f"Evaluation error for {selected_model}: {str(e)}")
        st.error(f"X_eval shape: {X_eval.shape}, y shape: {y.shape}")
        return None

@st.cache_data
def evaluate_all_models(_models, df_clean):
    """Compute evaluation metrics for all loaded models against df_clean.
    Returns dict: model_name -> {r2, mae, rmse, mape, n_samples}
    """
    results = {}
    models = _models
    if df_clean is None or not models:
        return results
    try:
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    except Exception:
        return results
    
    # Build features
    base_cols = [
        'groundwater_ph', 'ec_groundwater_(µs/cm)', 'hardness_groundwater_(mg/l)',
        'nitrate_groundwater_(mg/l)', 'rainfall_mm', 'soil_ph',
        'soil_organic_carbon', 'soil_nitrogen', 'soil_phosphorus', 'soil_potassium',
        'state', 'district', 'crop'
    ]
    data = df_clean.dropna(subset=['crop_yield']).copy()
    available_cols = [c for c in base_cols if c in data.columns]
    if not available_cols:
        return results
    X_raw = data[available_cols].copy()
    y = data['crop_yield'].values
    
    cat_cols = [c for c in ['state', 'district', 'crop'] if c in X_raw.columns]
    if cat_cols:
        X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
    else:
        X_encoded = X_raw
    
    feature_names_ref = None
    try:
        with open('notebooks/feature_info.pkl', 'rb') as f:
            fi = pickle.load(f)
            feature_names_ref = fi.get('feature_columns')
    except Exception:
        pass
    
    if feature_names_ref is not None:
        for col in feature_names_ref:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        X_encoded = X_encoded[feature_names_ref]
    
    for model_name, model in models.items():
        try:
            # Handle both wrapped and direct model formats
            if isinstance(model, dict) and 'model' in model:
                core_model = model['model']
                scaler = model.get('scaler')
            else:
                core_model = model
                scaler = None
            X_eval = X_encoded.values
            if scaler is not None:
                try:
                    X_eval = scaler.transform(X_eval)
                except Exception:
                    pass
            y_pred = core_model.predict(X_eval)
            mae = float(mean_absolute_error(y, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
            with np.errstate(divide='ignore', invalid='ignore'):
                mape_arr = np.abs((y - y_pred) / np.where(y == 0, np.nan, y))
            mape = float(np.nanmean(mape_arr) * 100)
            r2 = float(r2_score(y, y_pred))
            results[model_name] = {
                'r2': r2,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'n_samples': int(len(y))
            }
        except Exception as e:
            st.error(f"Evaluation error for {model_name}: {str(e)}")
            continue
    return results

# Initialize session state
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

# Load data and models
with st.spinner("Loading models and data..."):
    models, model_load_messages = load_models()
    df, df_clean = load_data()

if df is None:
    st.stop()

# Sidebar navigation
with st.sidebar:
    st.image("https://via.placeholder.com/200x100/2E8B57/FFFFFF?text=Crop+AI", width=200)
    st.markdown("---")
    
    page = st.selectbox(
        "🌾 Navigate",
        ["🏠 Home", "📍 Location Prediction", "🔬 Parameter Prediction", 
         "📊 Analytics Dashboard", "💬 AI Assistant", "📚 Help", "🔍 Model Details"],
        key="navigation_page"
    )
    
    # Allow Quick Start buttons to navigate without modifying the widget state
    if 'navigate_to_page' in st.session_state:
        page = st.session_state.navigate_to_page
        del st.session_state['navigate_to_page']
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### 📈 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Models Available", len(models))

# Main content based on selected page
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🌾 Crop Prediction AI</h1>', unsafe_allow_html=True)
    
    # Welcome section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <style>
            .metric-card {
                color: black; /* text color */
            }
        </style>
        <div class="metric-card">
            <h3>Welcome to Crop Prediction AI!</h3>
            <p>Get intelligent crop recommendations based on your location and soil conditions. 
            Our AI models analyze environmental factors to suggest the best crops for your farm.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model load status (visible only on Home)
    st.markdown("### 🤖 Models Loaded")
    cols = st.columns(2)
    idx = 0
    for msg in model_load_messages:
        if msg.get("level") == "success" and msg.get("text", "").startswith("✅ Loaded"):
            with cols[idx % 2]:
                label = msg["text"].replace("✅ ", "")  # cleaner label on button
                if st.button(label, key=f"model_btn_{idx}", use_container_width=True):
                    # Extract model name from text 'Loaded {Model} model'
                    try:
                        model_name = label.replace("Loaded ", "").replace(" model", "").strip()
                    except Exception:
                        model_name = label
                    st.session_state.selected_model = model_name
                    st.session_state.navigate_to_page = "🔍 Model Details"
                    st.rerun()
            idx += 1
    
    # Recent predictions
    if st.session_state.predictions_history:
        st.markdown("### 📋 Recent Predictions")
        for i, pred in enumerate(st.session_state.predictions_history[-3:]):
            prediction_num = len(st.session_state.predictions_history) - i
            with st.expander(f"Prediction #{prediction_num}"):
                if pred['type'] == 'location_rf_top3':
                    st.markdown(f"""
                    **Location-Based Prediction**
                    📍 Location: {pred['state']}, {pred['district']}
                    
                    **Top Recommended Crops:**
                    1. {pred['top3'][0]['crop']} — {pred['top3'][0]['predicted_yield']:.1f} kg/acre
                    2. {pred['top3'][1]['crop']} — {pred['top3'][1]['predicted_yield']:.1f} kg/acre
                    3. {pred['top3'][2]['crop']} — {pred['top3'][2]['predicted_yield']:.1f} kg/acre
                    """)
                elif pred['type'] == 'parameter_rf_top3':
                    params = pred['parameters']
                    st.markdown(f"""
                    **Parameter-Based Prediction**
                    
                    🌱 **Soil Parameters:**
                    - pH: {params['soil_ph']:.1f}
                    - Nitrogen: {params['soil_nitrogen']:.0f} kg/acre
                    - Phosphorus: {params['soil_phosphorus']:.0f} kg/acre
                    - Potassium: {params['soil_potassium']:.0f} kg/acre
                    - Organic Carbon: {params['soil_organic_carbon']:.2f}%
                    
                    💧 **Water Conditions:**
                    - Groundwater pH: {params['groundwater_ph']:.1f}
                    - Water Hardness: {params['hardness_groundwater_(mg/l)']:.0f} mg/l
                    - Nitrate Level: {params['nitrate_groundwater_(mg/l)']:.0f} mg/l
                    - EC Level: {params['ec_groundwater_(µs/cm)']:.0f} µS/cm
                    
                    🌧️ **Environmental:**
                    - Rainfall: {params['rainfall_mm']:.0f} mm
                    
                    **Top Recommended Crops:**
                    1. {pred['top3'][0]['crop']} — {pred['top3'][0]['predicted_yield']:.1f} kg/acre
                    2. {pred['top3'][1]['crop']} — {pred['top3'][1]['predicted_yield']:.1f} kg/acre
                    3. {pred['top3'][2]['crop']} — {pred['top3'][2]['predicted_yield']:.1f} kg/acre
                    """)

elif page == "📍 Location Prediction":
    st.markdown("# 📍 Location-Based Crop Prediction")
    
    # Location selection
    col1, col2 = st.columns(2)
    
    with col1:
        # Get unique states and districts from actual dataset
        if df_clean is not None:
            states = sorted(df_clean['state'].unique())
            selected_state = st.selectbox("Select State", states, key="location_state")
            
            # Filter districts based on selected state
            state_districts = sorted(df_clean[df_clean['state'] == selected_state]['district'].unique())
            selected_district = st.selectbox("Select District", state_districts, key="location_district")
        else:
            st.error("Dataset not available. Please ensure the data files are present.")
            st.stop()
    
    with col2:
        pass
    
    # Get location-based recommendations (ML models only)
    if st.button("🔍 Get Recommendations", type="primary"):
        if df_clean is not None:
            # Filter data for selected location
            location_data = df_clean[(df_clean['state'] == selected_state) & (df_clean['district'] == selected_district)]
            
            # Build input features by using location means
            feature_cols = [
                'groundwater_ph',
                'ec_groundwater_(µs/cm)',
                'hardness_groundwater_(mg/l)',
                'nitrate_groundwater_(mg/l)',
                'rainfall_mm',
                'soil_ph',
                'soil_organic_carbon',
                'soil_nitrogen',
                'soil_phosphorus',
                'soil_potassium'
            ]
            if len(location_data) == 0:
                # Fall back to global means if location has no rows
                location_data = df_clean
            means = {col: float(location_data[col].mean()) for col in feature_cols if col in location_data.columns}
            
            # Add required fields if missing
            soil_defaults = {
                'soil_ph': 7.0,
                'soil_organic_carbon': 0.5,
                'soil_nitrogen': 200,
                'soil_phosphorus': 20,
                'soil_potassium': 250,
                'rainfall_mm': 800
            }
            input_data = {**soil_defaults, **means, 'state': selected_state, 'district': selected_district}
            
            # Run ML-only recommendations for selected location
            crop_recommendations, crop_predictions = get_crop_recommendations_ml_only(
                input_data, models, df_clean, state_override=selected_state, district_override=selected_district
            )
            
            if crop_predictions:
                st.success(f"✅ Generated recommendations for {selected_state}, {selected_district}")
                
                # Top 3 crops by predicted yield
                top3 = sorted(crop_predictions.items(), key=lambda kv: kv[1]['avg_yield'], reverse=True)[:3]
                
                for idx, (crop, data) in enumerate(top3):
                    with st.expander(f"""<span style='font-size: 24px; font-weight: 800;'>#{idx+1}: {crop}</span> <span style='font-size: 16px;'>(Expected Yield: {data['avg_yield']:.1f} kg/acre)</span>""", expanded=True):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown("### 🌾 Why This Crop?")
                            
                            # Get soil analysis
                            soil_condition = []
                            ph = input_data.get('soil_ph', 7.0)
                            if ph < 6.0:
                                soil_condition.append("Your soil is acidic")
                            elif ph > 7.5:
                                soil_condition.append("Your soil is alkaline")
                            else:
                                soil_condition.append("Your soil pH is ideal")
                            
                            nitrogen = input_data.get('soil_nitrogen', 0)
                            if nitrogen < 140:
                                soil_condition.append("nitrogen levels are low")
                            elif nitrogen > 350:
                                soil_condition.append("nitrogen levels are high")
                            else:
                                soil_condition.append("nitrogen levels are moderate")
                            
                            rainfall = input_data.get('rainfall_mm', 0)
                            if rainfall < 600:
                                soil_condition.append("rainfall is relatively low")
                            elif rainfall > 1200:
                                soil_condition.append("rainfall is abundant")
                            else:
                                soil_condition.append("rainfall is moderate")
                            
                            # Format the explanation
                            location_text = f"Based on historical data from {data['location_used']}, {crop} is an excellent choice for your farm. "
                            yield_text = f"You can expect a yield of approximately {data['avg_yield']:.1f} kg/acre under proper cultivation conditions. "
                            condition_text = "Current conditions show that " + ", ".join(soil_condition) + ". "
                            
                            recommendation_text = ""
                            if 'Rice' in crop or 'Wheat' in crop:
                                recommendation_text = "This crop is a staple food grain with consistent market demand. "
                            elif 'Cotton' in crop:
                                recommendation_text = "This is a valuable cash crop with strong commercial potential. "
                            elif 'Sugarcane' in crop:
                                recommendation_text = "This crop offers high yield potential and is valuable for sugar production. "
                            elif any(x in crop.lower() for x in ['pulse', 'gram', 'lentil']):
                                recommendation_text = "This legume crop will also help improve soil nitrogen content naturally. "
                            
                            st.markdown(f"""
                            #### 📊 Crop Analysis
                            
                            {location_text}{yield_text}
                            
                            #### 🌱 Growing Conditions
                            
                            {condition_text}{recommendation_text}
                            """)
                            
                        with col2:
                            # Fertilizer recommendations
                            fertilizer_recs = get_fertilizer_recommendations(input_data)
                            if fertilizer_recs:
                                st.markdown("### 🌿 Soil Enhancement Plan")
                                
                                # Group recommendations by priority
                                critical_needs = []
                                improvements = []
                                maintenance = []
                                
                                for rec in fertilizer_recs:
                                    if rec['status'] == 'Low':
                                        critical_needs.append(rec)
                                    else:
                                        improvements.append(rec)
                                
                                if critical_needs:
                                    st.markdown("#### � Priority Actions")
                                    for rec in critical_needs:
                                        nutrient = rec['nutrient'].replace('soil_', '').replace('_', ' ').title()
                                        
                                        # Create natural language recommendations
                                        if 'fertilizers' in rec:
                                            main_fert = rec['fertilizers'][0]  # Primary recommendation
                                            alt_ferts = rec['fertilizers'][1:]  # Alternatives
                                            
                                            st.markdown(f"""
                                            **{nutrient} Enhancement:**
                                            
                                            To address the low {nutrient.lower()} levels, we recommend using {main_fert}. 
                                            
                                            Alternative options include {', '.join(alt_ferts)}.
                                            """)
                                        
                                        if 'organic_options' in rec:
                                            org_options = rec['organic_options']
                                            st.markdown(f"""
                                            **Organic Alternatives:**
                                            
                                            For organic farming, you can use {', '.join(org_options[:-1])} or {org_options[-1]}.
                                            """)
                                
                                if improvements:
                                    st.markdown("#### 🔄 Maintenance Recommendations")
                                    for rec in improvements:
                                        if 'adjustments' in rec:
                                            treatment = rec['adjustments'][0]
                                            st.markdown(f"""
                                            To maintain optimal soil conditions, consider applying {treatment.lower()} 
                                            according to soil test recommendations.
                                            """)
                                        if 'notes' in rec:
                                            st.info(rec['notes'])
                
                # Save to history with additional details
                st.session_state.predictions_history.append({
                    'type': 'location_rf_top3',
                    'state': selected_state,
                    'district': selected_district,
                    'timestamp': pd.Timestamp.now(),
                    'top3': [{
                        'crop': crop,
                        'predicted_yield': data['avg_yield'],
                        'location': data['location_used'],
                        'model_count': data['model_count']
                    } for crop, data in top3]
                })
            else:
                st.error("Unable to generate ML predictions for this location.")
        else:
            st.error("Dataset not available. Please ensure the data files are present.")

elif page == "🔬 Parameter Prediction":
    st.markdown("# 🔬 Parameter-Based Crop Recommendation")
    st.markdown("### 🌾 Find the Best Crop for Your Environmental Conditions")
    
    # Create columns for organized input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌱 Soil Conditions")
        soil_ph = st.slider("Soil pH", 4.5, 9.0, 7.0, 0.1)
        organic_carbon = st.slider("Organic Carbon %", 0.4, 1.1, 0.7, 0.01)
        nitrogen = st.slider("Nitrogen (kg/acre)", 161, 416, 200, 1)
        phosphorus = st.slider("Phosphorus (kg/acre)", 11, 33, 20, 1)
        potassium = st.slider("Potassium (kg/acre)", 100, 500, 250, 1)
        
        st.markdown("#### 💧 Water Conditions")
        groundwater_ph = st.slider("Groundwater pH", 6.4, 8.8, 7.5, 0.1)
        water_hardness = st.slider("Water Hardness (mg/l)", 200, 1000, 400, 10)
        nitrate_level = st.slider("Nitrate Level (mg/l)", 5, 281, 40, 1)
        ec_level = st.slider("EC Level (µS/cm)", 650, 2065, 1200, 10)
    
    with col2:
        st.markdown("#### 🌧️ Environmental Conditions")
        rainfall = st.slider("Expected Rainfall (mm)", 400, 1000, 650, 10)
    
    # Prediction button (ML models only)
    if st.button("🔮 Find Best Crop", type="primary", use_container_width=True):
        # Prepare input data (without location and crop)
        input_data = {
            'groundwater_ph': groundwater_ph,
            'ec_groundwater_(µs/cm)': ec_level,
            'hardness_groundwater_(mg/l)': water_hardness,
            'nitrate_groundwater_(mg/l)': nitrate_level,
            'rainfall_mm': rainfall,
            'soil_ph': soil_ph,
            'soil_organic_carbon': organic_carbon,
            'soil_nitrogen': nitrogen,
            'soil_phosphorus': phosphorus,
            'soil_potassium': potassium
        }
        
        # ML-only recommendations
        crop_recommendations, crop_predictions = get_crop_recommendations_ml_only(input_data, models, df_clean)
        
        if crop_predictions:
            st.success("✅ Analysis complete! Here are your personalized crop recommendations:")
            
            # Get top 3 crops
            top3 = sorted(crop_predictions.items(), key=lambda kv: kv[1]['avg_yield'], reverse=True)[:3]
            
            # Display each crop with detailed analysis
            for idx, (crop, data) in enumerate(top3):
                with st.expander(f"#{idx+1}: {crop}", expanded=True):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Analyze soil conditions for this crop
                        soil_analysis = []
                        if soil_ph < 6.0:
                            soil_analysis.append("your soil is acidic")
                        elif soil_ph > 7.5:
                            soil_analysis.append("your soil is alkaline")
                        else:
                            soil_analysis.append("your soil pH is in the optimal range")
                        
                        if nitrogen < 140:
                            soil_analysis.append("nitrogen levels need improvement")
                        elif nitrogen > 350:
                            soil_analysis.append("nitrogen levels are abundant")
                        else:
                            soil_analysis.append("nitrogen levels are good")
                        
                        if phosphorus < 10:
                            soil_analysis.append("phosphorus levels are low")
                        elif phosphorus > 25:
                            soil_analysis.append("phosphorus levels are high")
                        else:
                            soil_analysis.append("phosphorus levels are suitable")
                        
                        # Generate natural language explanation
                        st.markdown("### 🌱 Crop Analysis")
                        
                        # Yield prediction
                        st.markdown(f"""
                        Based on your soil and environmental conditions, {crop.lower()} shows excellent potential 
                        with an expected yield of {data['avg_yield']:.1f} kg/acre.
                        
                        #### 📊 Growing Conditions Assessment
                        Our analysis shows that {', and '.join(soil_analysis)}. {get_crop_specific_advice(crop)}
                        """)
                        
                        # Rainfall analysis
                        rainfall_text = ""
                        if rainfall < 600:
                            rainfall_text = ("The rainfall in your area is relatively low. "
                                          "Consider implementing irrigation systems for optimal growth.")
                        elif rainfall > 900:
                            rainfall_text = ("Your area receives abundant rainfall. "
                                          "Ensure good drainage to prevent waterlogging.")
                        else:
                            rainfall_text = "The rainfall in your area is well-suited for this crop."
                        
                        st.markdown(f"""
                        #### 🌧️ Water Availability
                        {rainfall_text}
                        """)
                    
                    with col2:
                        # Fertilizer recommendations based on soil parameters
                        # st.markdown(f"""
                        # <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #2E8B57;'>
                        #     <h2 style='color: #000000; margin-bottom: 20px;'>🌿 Soil Enhancement Plan for <span style='font-size: 1.8em; font-weight: 800; color: #000000; display: block; margin-top: 10px; text-transform: uppercase; text-decoration: underline;'>{crop.upper()}</span></h2>
                        # </div>
                        # """, unsafe_allow_html=True)
                        
                        fertilizer_recs = get_fertilizer_recommendations(input_data)
                        if fertilizer_recs:
                            for rec in fertilizer_recs:
                                if rec['status'] == 'Low':
                                    with st.expander(f"📋 {rec['nutrient']} Management", expanded=True):
                                        if 'fertilizers' in rec:
                                            st.markdown(f"""
                                            <div style='background-color: white; padding: 15px; border-radius: 5px; margin-bottom: 10px;'>
                                                <p style='font-size: 1.1em; color: #2E8B57;'>To optimize <strong>{crop.upper()}</strong> growth, we recommend:</p>
                                                <div style='background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                                                    <strong>Primary Choice:</strong> {rec['fertilizers'][0]}
                                                </div>
                                                
                                                <p style='margin-top: 15px;'><strong>Alternative Options:</strong></p>
                                                <ul style='list-style-type: none;'>
                                                    {' '.join([f"<li style='margin: 5px 0;'>• {fert}</li>" for fert in rec['fertilizers'][1:]])}
                                                </ul>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        
                                        if 'organic_options' in rec:
                                            st.markdown(f"""
                                            <div style='background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin-top: 10px;'>
                                                <p style='color: #2E8B57; font-weight: bold;'>🌱 Organic Alternatives:</p>
                                                <ul style='list-style-type: none; margin: 10px 0;'>
                                                    {' '.join([f"<li style='margin: 5px 0;'>• {opt}</li>" for opt in rec['organic_options']])}
                                                </ul>
                                            </div>
                                            """, unsafe_allow_html=True)
            
            # Save prediction to history
            st.session_state.predictions_history.append({
                'type': 'parameter_rf_top3',
                'parameters': input_data,
                'timestamp': pd.Timestamp.now(),
                'top3': [{
                    'crop': name,
                    'predicted_yield': data['avg_yield'],
                    'soil_conditions': {
                        'ph': soil_ph,
                        'nitrogen': nitrogen,
                        'phosphorus': phosphorus,
                        'potassium': potassium,
                        'rainfall': rainfall
                    }
                } for name, data in top3]
            })
        else:
            st.error("Unable to generate crop recommendations. Please check your input parameters.")

elif page == "📊 Analytics Dashboard":
    st.markdown("# 📊 Analytics Dashboard")
    
    if df_clean is not None:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df_clean))
        with col2:
            st.metric("Unique Crops", df_clean['crop'].nunique())
        with col3:
            st.metric("States Covered", df_clean['state'].nunique())
        with col4:
            st.metric("Avg Yield", f"{df_clean['crop_yield'].mean():.1f} kg/acre")
        
        # Visualizations
        tab1, tab2, tab3 = st.tabs(["🌾 Crop Analysis", "📍 Regional Analysis", "📈 Trends"])
        
        with tab1:
            # Normalize crop names to avoid duplicates due to casing/whitespace
            df_crop = df_clean.copy()
            if 'crop' in df_crop.columns:
                df_crop['crop_clean'] = df_crop['crop'].astype(str).str.strip().str.title()
            else:
                df_crop['crop_clean'] = 'Unknown'
            
            # Crop performance
            crop_performance = df_crop.groupby('crop_clean')['crop_yield'].agg(['mean', 'count']).reset_index()
            crop_performance = crop_performance.sort_values('mean', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(crop_performance.head(10), x='crop_clean', y='mean',
                           title="Top 10 Crops by Average Yield",
                           labels={'crop_clean': 'Crop', 'mean': 'Average Yield'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Show parameter dependency via Random Forest feature importances
                rf_entry = models.get('Random Forest') if isinstance(models, dict) else None
                rf_model = None
                if isinstance(rf_entry, dict) and 'model' in rf_entry:
                    rf_model = rf_entry['model']
                elif rf_entry is not None:
                    rf_model = rf_entry
                
                feature_names = None
                try:
                    with open('notebooks/feature_info.pkl', 'rb') as f:
                        fi = pickle.load(f)
                        feature_names = fi.get('feature_columns')
                except Exception:
                    pass
                
                if rf_model is not None and hasattr(rf_model, 'feature_importances_'):
                    importances = getattr(rf_model, 'feature_importances_')
                    # Build DataFrame for plotting
                    if feature_names is not None and len(feature_names) == len(importances):
                        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                    else:
                        imp_df = pd.DataFrame({'Feature': [f'F{i}' for i in range(len(importances))], 'Importance': importances})
                    # Show top 12 most important parameters
                    imp_df = imp_df.sort_values('Importance', ascending=False).head(12)
                    fig = px.bar(imp_df, x='Feature', y='Importance',
                                 title="Model Feature Importances (Parameter Dependency)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Random Forest feature importances unavailable.")
        
        with tab2:
            # Regional analysis
            regional_performance = df_clean.groupby(['state', 'district'])['crop_yield'].mean().reset_index()
            regional_performance = regional_performance.sort_values('crop_yield', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Replace with: Which crop is grown max in which state
                df_state_crop = df_clean.copy()
                # Normalize crop names similar to tab1 to avoid duplicates
                df_state_crop['crop_clean'] = df_state_crop['crop'].astype(str).str.strip().str.title()
                state_crop_perf = df_state_crop.groupby(['state', 'crop_clean'])['crop_yield'].mean().reset_index()
                # For each state, pick crop with highest avg yield
                top_crop_by_state = state_crop_perf.sort_values(['state', 'crop_yield'], ascending=[True, False]).groupby('state').head(1)
                fig = px.bar(top_crop_by_state, x='state', y='crop_yield', color='crop_clean',
                             title="Top-Yielding Crop per State",
                             labels={'state': 'State', 'crop_yield': 'Avg Yield', 'crop_clean': 'Top Crop'},
                             text='crop_clean')
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # State-wise analysis
                state_performance = df_clean.groupby('state')['crop_yield'].mean().reset_index()
                fig = px.bar(state_performance, x='state', y='crop_yield',
                           title="Average Yield by State")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Yearly trends
            yearly_trends = df_clean.groupby('year')['crop_yield'].mean().reset_index()
            
            fig = px.line(yearly_trends, x='year', y='crop_yield',
                         title="Average Yield Trends Over Time")
            st.plotly_chart(fig, use_container_width=True)
            
            # Seasonal analysis
            if 'month' in df_clean.columns:
                seasonal_data = df_clean.groupby('month')['crop_yield'].mean().reset_index()
                fig = px.line(seasonal_data, x='month', y='crop_yield',
                             title="Seasonal Yield Patterns")
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("Analytics data not available.")

elif page == "💬 AI Assistant":
    st.markdown("# 💬 AI Assistant")
    
    # Show API key setup if Gemini is not available
    if not GEMINI_AVAILABLE:
        st.warning("⚠️ **AI Assistant requires Google Gemini API setup**")
        
        with st.expander("🔑 Setup Instructions (Click to expand)", expanded=True):
            st.markdown(f"""
            ### To set up the AI Assistant:
            
            1. **Install required package**:
               ```bash
               pip install google-generativeai
               ```
            
            2. **Get your API key** from [Google AI Studio](https://makersuite.google.com/app/apikey)
            
            3. **Create a `.env` file** in the project root with:
               ```env
               GOOGLE_API_KEY=your_key_here
               ```
            
            4. **Restart the Streamlit app**
            
            **Example `.env` file location**: Create a file named `.env` in the same directory as `app.py`
            
            **Current Status**: {GEMINI_REASON or "Unknown error"}
            """)
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Helper function to extract location from prompt
    def extract_location_context(prompt_text, dataset):
        """Extract location information from user prompt"""
        if dataset is None:
            return None
        
        prompt_lower = prompt_text.lower()
        
        # Try to find state/district mentions
        states = dataset['state'].unique()
        districts = dataset['district'].unique()
        
        location_info = None
        
        # Check for state mention
        for state in states:
            if state.lower() in prompt_lower:
                # Try to find district in same sentence/context
                state_data = dataset[dataset['state'] == state]
                state_districts = state_data['district'].unique()
                
                for district in state_districts:
                    if district.lower() in prompt_lower:
                        location_info = {'state': state, 'district': district}
                        break
                
                # If no district found, use first district from that state
                if location_info is None and len(state_districts) > 0:
                    location_info = {'state': state, 'district': state_districts[0]}
                break
        
        return location_info
    
    # Helper function to extract soil parameters from prompt
    def extract_soil_context(prompt_text):
        """Extract soil parameters from user prompt if mentioned"""
        soil_params = {}
        prompt_lower = prompt_text.lower()
        
        # Look for pH mentions
        if 'ph' in prompt_lower:
            ph_match = re.search(r'ph\s*[=:]\s*([\d.]+)', prompt_lower)
            if ph_match:
                soil_params['soil_ph'] = float(ph_match.group(1))
        
        # Look for nitrogen mentions
        if 'nitrogen' in prompt_lower or 'n' in prompt_lower:
            n_match = re.search(r'nitrogen\s*[=:]\s*([\d.]+)', prompt_lower)
            if n_match:
                soil_params['soil_nitrogen'] = float(n_match.group(1))
        
        return soil_params if soil_params else None
    
    # Chat input
    if prompt := st.chat_input("Ask me about crop predictions, soil conditions, or farming advice..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Extract context from prompt
        location_context = extract_location_context(prompt, df_clean)
        soil_context = extract_soil_context(prompt)
        
        # Generate AI response with context
        with st.chat_message("assistant"):
            with st.spinner("🤖 AI is thinking..."):
                response = generate_ai_response(
                    prompt, 
                    location=location_context,
                    soil_params=soil_context,
                    df_clean=df_clean
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

elif page == "📚 Help":
    st.markdown("# 📚 Help & Guide")
    
    st.markdown("""
    ## 🌾 How to Use Crop Prediction AI
    
    ### 📍 Location-Based Prediction
    1. Select your state and district
    2. Choose the year for prediction
    3. Optionally filter by crop preference
    4. Get recommendations based on historical data
    
    ### 🔬 Parameter-Based Prediction
    1. Enter your soil test results
    2. Input environmental conditions
    3. Select your location and crop type
    4. Get yield predictions with confidence intervals
    
    ### 📊 Understanding the Results
    
    **Yield Predictions:**
    - Based on machine learning models trained on historical data
    - Confidence levels: High (>3 models), Medium (2-3 models), Low (<2 models)
    - SHAP explanations show which factors most influence predictions
    
    **Feature Importance:**
    - Soil pH: Optimal range 6.5-7.5 for most crops
    - Nitrogen: Essential for plant growth
    - Rainfall: Critical for crop success
    - Organic Carbon: Improves soil structure
    
    ### 💡 Tips for Better Predictions
    - Use recent soil test results
    - Consider seasonal variations
    - Account for local weather patterns
    - Regular soil testing improves accuracy
    """)

elif page == "🔍 Model Details":
    selected_model = st.session_state.get('selected_model')
    if not selected_model:
        st.info("Select a model from Home to view its details.")
    else:
        if st.button("← Back to Home", type="secondary"):
            st.session_state.navigate_to_page = "🏠 Home"
            st.rerun()
        st.markdown(f"# 🔍 Model Details — {selected_model}")
        model_obj = models.get(selected_model)
        if model_obj is None:
            st.error("Model not loaded or unavailable.")
        else:
            # Dataset stats
            if df_clean is not None:
                st.markdown("### 📊 Dataset Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(df_clean))
                with col2:
                    st.metric("Unique Crops", df_clean['crop'].nunique())
                with col3:
                    st.metric("States", df_clean['state'].nunique())
            
            # Feature names from feature_info if available
            feature_names = None
            try:
                with open('notebooks/feature_info.pkl', 'rb') as f:
                    fi = pickle.load(f)
                    feature_names = fi.get('feature_columns')
            except Exception:
                pass
            
            # Metrics
            st.markdown("### 📐 Metrics")
            # Prefer a precomputed metrics file (useful for stable display / CI)
            metrics_all = load_model_metrics_file()
            if metrics_all is None:
                metrics_all = evaluate_all_models(models, df_clean)
            if metrics_all and selected_model in metrics_all:
                m = metrics_all[selected_model]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R²", f"{m.get('r2', 0):.3f}")
                with col2:
                    st.metric("MAE", f"{m.get('mae', 0):.2f}")
                with col3:
                    st.metric("RMSE", f"{m.get('rmse', 0):.2f}")
            else:
                st.info("Computing metrics...")
                # Simple fallback metrics
                if df_clean is not None:
                    st.info(f"Dataset: {len(df_clean)} records, {df_clean['crop_yield'].nunique()} unique yields")
                    st.info(f"Model: {selected_model}")
            
            # All models metrics table
            if metrics_all:
                st.markdown("### 📋 All Models — Metrics")
                df_metrics = pd.DataFrame([
                    {
                        'Model': name,
                        'R2': vals.get('r2', 0),
                        'MAE': vals.get('mae', 0),
                        'RMSE': vals.get('rmse', 0)
                    } for name, vals in metrics_all.items() if isinstance(vals, dict)
                ])
                if not df_metrics.empty and 'R2' in df_metrics.columns:
                    df_metrics = df_metrics.sort_values('R2', ascending=False)
                st.dataframe(df_metrics, use_container_width=True)
            
            # Model insights removed per user request
            
            # Per-crop average yield overview for context
            if df_clean is not None:
                st.markdown("### 🌾 Crop Yield Overview")
                crop_perf = df_clean.groupby('crop')['crop_yield'].mean().reset_index().sort_values('crop_yield', ascending=False).head(15)
                fig2 = px.bar(crop_perf, x='crop', y='crop_yield', title="Top 15 Crops by Average Yield")
                st.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🌾 Crop Prediction AI | Powered by Machine Learning | Built with Streamlit
</div>
""", unsafe_allow_html=True)
