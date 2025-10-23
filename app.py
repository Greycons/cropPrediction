import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings('ignore')

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
def generate_ai_response(prompt):
    """Generate AI response based on user prompt"""
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ['soil', 'ph', 'nitrogen']):
        return """
        🌱 **Soil Health Tips:**
        
        - **Soil pH**: Most crops prefer 6.5-7.5. Add lime if too acidic, sulfur if too alkaline
        - **Nitrogen**: Essential for leaf growth. Use organic matter or nitrogen fertilizers
        - **Organic Carbon**: Improves water retention and soil structure
        - **Regular Testing**: Test soil every 2-3 years for best results
        """
    
    elif any(word in prompt_lower for word in ['yield', 'production', 'harvest']):
        return """
        📈 **Yield Optimization:**
        
        - **Crop Selection**: Choose crops suited to your soil and climate
        - **Timing**: Plant at optimal times for your region
        - **Fertilization**: Follow soil test recommendations
        - **Water Management**: Ensure adequate irrigation
        - **Pest Control**: Monitor and treat pests early
        """
    
    elif any(word in prompt_lower for word in ['weather', 'rainfall', 'climate']):
        return """
        🌧️ **Weather Considerations:**
        
        - **Rainfall**: Most crops need 500-800mm annually
        - **Temperature**: Check optimal growing temperatures
        - **Seasonal Patterns**: Plan planting around rainy seasons
        - **Drought Management**: Consider drought-resistant varieties
        """
    
    elif any(word in prompt_lower for word in ['crop', 'plant', 'grow']):
        return """
        🌾 **Crop Selection Guide:**
        
        - **Rice**: Needs standing water, high rainfall areas
        - **Wheat**: Cool season crop, moderate water needs
        - **Maize**: Warm season, good drainage required
        - **Sugarcane**: Tropical/subtropical, high water needs
        - **Cotton**: Warm climate, well-drained soil
        """
    
    else:
        return """
        🤖 **I'm here to help with:**
        
        - Crop selection and yield predictions
        - Soil health and fertilization advice
        - Weather and climate considerations
        - Farming best practices
        - Understanding prediction results
        
        Ask me anything about farming, soil conditions, or crop predictions!
        """

# Page configuration
st.set_page_config(
    page_title="🌾 Crop Prediction AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
    }
    .prediction-card {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #2E8B57;
        margin: 1rem 0;
    }
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
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E8B57;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_data
def load_models():
    """Load all trained models with robust error handling"""
    import os
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
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    models[name] = pickle.load(f)
                st.success(f"✅ Loaded {name} model")
            else:
                st.warning(f"⚠️ Model {name} not found at {file_path}")
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            st.error(f"❌ Corrupted model file {name}: {str(e)}")
            st.info(f"💡 Try regenerating the model by running the notebooks")
        except Exception as e:
            st.error(f"❌ Error loading {name}: {str(e)}")
    
    if not models:
        st.error("🚨 No models loaded! Please run the model training notebooks first.")
        st.info("📝 To fix this:")
        st.info("1. Run `jupyter notebook notebooks/model.ipynb`")
        st.info("2. Execute all cells to train the models")
        st.info("3. Or run `python generate_demo_data.py` for demo models")
    
    return models

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

# Initialize session state
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

# Load data and models
with st.spinner("Loading models and data..."):
    models = load_models()
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
         "📊 Analytics Dashboard", "💬 AI Assistant", "📚 Help"]
    )
    
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
        <div class="metric-card">
            <h3>Welcome to Crop Prediction AI!</h3>
            <p>Get intelligent crop recommendations based on your location and soil conditions. 
            Our AI models analyze environmental factors to suggest the best crops for your farm.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick access buttons
    st.markdown("### 🚀 Quick Start")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📍 Predict by Location", use_container_width=True):
            st.session_state.page = "📍 Location Prediction"
            st.rerun()
    
    with col2:
        if st.button("🔬 Predict by Parameters", use_container_width=True):
            st.session_state.page = "🔬 Parameter Prediction"
            st.rerun()
    
    # Recent predictions
    if st.session_state.predictions_history:
        st.markdown("### 📋 Recent Predictions")
        for i, pred in enumerate(st.session_state.predictions_history[-3:]):
            with st.expander(f"Prediction #{len(st.session_state.predictions_history)-i}"):
                st.json(pred)

elif page == "📍 Location Prediction":
    st.markdown("# 📍 Location-Based Crop Prediction")
    
    # Location selection
    col1, col2 = st.columns(2)
    
    with col1:
        # Get unique states and districts
        states = df_clean['state'].unique() if df_clean is not None else ['Andhra Pradesh', 'Karnataka', 'Tamil Nadu']
        districts = df_clean['district'].unique() if df_clean is not None else ['Anantapur', 'Bangalore', 'Chennai']
        
        selected_state = st.selectbox("Select State", states)
        
        # Filter districts based on state
        if df_clean is not None:
            state_districts = df_clean[df_clean['state'] == selected_state]['district'].unique()
        else:
            state_districts = districts
            
        selected_district = st.selectbox("Select District", state_districts)
    
    with col2:
        selected_year = st.selectbox("Select Year", range(2024, 2031))
        crop_preference = st.multiselect("Crop Preference (Optional)", 
                                       ['Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton'])
    
    # Get location-based recommendations
    if st.button("🔍 Get Recommendations", type="primary"):
        if df_clean is not None:
            # Filter data for selected location
            location_data = df_clean[
                (df_clean['state'] == selected_state) & 
                (df_clean['district'] == selected_district)
            ]
            
            if len(location_data) > 0:
                # Get top crops for this location
                crop_performance = location_data.groupby('crop')['crop_yield'].agg(['mean', 'count']).reset_index()
                crop_performance = crop_performance.sort_values('mean', ascending=False)
                
                # Filter by preference if selected
                if crop_preference:
                    crop_performance = crop_performance[crop_performance['crop'].isin(crop_preference)]
                
                # Display results
                st.markdown("### 🌾 Recommended Crops")
                
                for i, (_, row) in enumerate(crop_performance.head(5).iterrows()):
                    with st.container():
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h4>#{i+1} {row['crop']}</h4>
                            <p><strong>Average Yield:</strong> {row['mean']:.1f} kg/acre</p>
                            <p><strong>Data Points:</strong> {row['count']} records</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Visualization
                fig = px.bar(crop_performance.head(5), 
                           x='crop', y='mean',
                           title="Top 5 Crops by Average Yield",
                           labels={'mean': 'Average Yield (kg/acre)', 'crop': 'Crop Type'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Historical trends
                if len(location_data) > 1:
                    st.markdown("### 📈 Historical Trends")
                    yearly_data = location_data.groupby(['year', 'crop'])['crop_yield'].mean().reset_index()
                    
                    fig = px.line(yearly_data, x='year', y='crop_yield', color='crop',
                                title="Yield Trends Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Save prediction to history
                prediction = {
                    'type': 'location',
                    'state': selected_state,
                    'district': selected_district,
                    'year': selected_year,
                    'top_crop': crop_performance.iloc[0]['crop'],
                    'avg_yield': crop_performance.iloc[0]['mean']
                }
                st.session_state.predictions_history.append(prediction)
                
            else:
                st.warning("No historical data found for this location. Try parameter-based prediction instead.")
        else:
            st.error("Dataset not available. Please ensure the data files are present.")

elif page == "🔬 Parameter Prediction":
    st.markdown("# 🔬 Parameter-Based Crop Prediction")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📝 Manual Input", "📄 Upload Soil Report"])
    
    with tab1:
        st.markdown("### 🌡️ Enter Your Farm Parameters")
        
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
            year = st.selectbox("Year", range(2024, 2031))
            
            st.markdown("#### 📍 Location")
            state = st.selectbox("State", df_clean['state'].unique() if df_clean is not None else ['Andhra Pradesh'])
            district = st.selectbox("District", df_clean['district'].unique() if df_clean is not None else ['Anantapur'])
            crop = st.selectbox("Crop Type", df_clean['crop'].unique() if df_clean is not None else ['Rice'])
        
        # Prediction button
        if st.button("🔮 Predict Yield", type="primary", use_container_width=True):
            # Prepare input data
            input_data = {
                'state': state,
                'district': district,
                'year': year,
                'crop': crop,
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
            
            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Normalize data (simplified - in real app, use proper scaler)
            numeric_cols = ['groundwater_ph', 'ec_groundwater_(µs/cm)', 'hardness_groundwater_(mg/l)',
                           'nitrate_groundwater_(mg/l)', 'rainfall_mm', 'soil_ph',
                           'soil_organic_carbon', 'soil_nitrogen', 'soil_phosphorus', 'soil_potassium']
            
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
                
                st.info(f"📊 Using {len(expected_columns)} features as expected by models")
            elif df is not None:
                # Fallback to original method
                training_cols = [col for col in df.columns if col != 'crop_yield']
                
                # Add missing columns with 0
                for col in training_cols:
                    if col not in input_encoded.columns:
                        input_encoded[col] = 0
                
                # Reorder columns to match training data
                input_encoded = input_encoded[training_cols]
            else:
                st.error("No training data or feature info available")
                st.stop()
            
            # Make predictions with all available models
            predictions = {}
            for model_name, model in models.items():
                try:
                    if model_name in ['MLP', 'SVR'] and isinstance(model, dict):
                        # Handle models with scalers
                        scaler = model.get('scaler')
                        model_obj = model.get('model')
                        if scaler and model_obj:
                            input_scaled = scaler.transform(input_encoded)
                            pred = model_obj.predict(input_scaled)[0]
                        else:
                            pred = model.predict(input_encoded)[0]
                    else:
                        pred = model.predict(input_encoded)[0]
                    predictions[model_name] = pred
                except Exception as e:
                    st.warning(f"Error with {model_name}: {str(e)}")
            
            if predictions:
                # Display results
                st.markdown("### 🎯 Prediction Results")
                
                # Average prediction
                avg_prediction = np.mean(list(predictions.values()))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Yield", f"{avg_prediction:.1f} kg/acre")
                with col2:
                    st.metric("Confidence", "High" if len(predictions) > 3 else "Medium")
                with col3:
                    st.metric("Models Used", len(predictions))
                
                # Individual model predictions
                st.markdown("#### 📊 Model-wise Predictions")
                pred_df = pd.DataFrame(list(predictions.items()), columns=['Model', 'Prediction'])
                fig = px.bar(pred_df, x='Model', y='Prediction', 
                           title="Predictions by Different Models")
                st.plotly_chart(fig, use_container_width=True)
                
                # SHAP explanation (if available)
                if SHAP_AVAILABLE and MATPLOTLIB_AVAILABLE and 'Random Forest' in models:
                    try:
                        st.markdown("#### 🔍 Feature Importance (SHAP)")
                        explainer = shap.TreeExplainer(models['Random Forest'])
                        shap_values = explainer.shap_values(input_encoded)
                        
                        # Create SHAP summary plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(shap_values, input_encoded, show=False)
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"SHAP explanation not available: {str(e)}")
                elif not SHAP_AVAILABLE:
                    st.info("SHAP not available. Install with: pip install shap")
                
                # Save prediction
                prediction = {
                    'type': 'parameter',
                    'parameters': input_data,
                    'predicted_yield': avg_prediction,
                    'model_predictions': predictions
                }
                st.session_state.predictions_history.append(prediction)
                
            else:
                st.error("No models available for prediction.")
        else:
            st.error("Training data not available.")
    
    with tab2:
        st.markdown("### 📄 Upload Soil Test Report")
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                st.success("File uploaded successfully!")
                st.dataframe(data.head())
                
                # Process uploaded data
                if st.button("🔮 Predict from Uploaded Data"):
                    st.info("Processing uploaded data...")
                    # Add processing logic here
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

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
            # Crop performance
            crop_performance = df_clean.groupby('crop')['crop_yield'].agg(['mean', 'count']).reset_index()
            crop_performance = crop_performance.sort_values('mean', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(crop_performance.head(10), x='crop', y='mean',
                           title="Top 10 Crops by Average Yield")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(crop_performance.head(8), values='count', names='crop',
                           title="Crop Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Regional analysis
            regional_performance = df_clean.groupby(['state', 'district'])['crop_yield'].mean().reset_index()
            regional_performance = regional_performance.sort_values('crop_yield', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(regional_performance.head(15), x='district', y='crop_yield',
                           title="Top 15 Districts by Average Yield")
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
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about crop predictions, soil conditions, or farming advice..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            # Simple AI responses based on keywords
            response = generate_ai_response(prompt)
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

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🌾 Crop Prediction AI | Powered by Machine Learning | Built with Streamlit
</div>
""", unsafe_allow_html=True)
