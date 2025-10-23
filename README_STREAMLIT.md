# 🌾 Crop Prediction AI - Streamlit Application

A comprehensive web application for crop yield prediction using machine learning models and SHAP explanations.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Required data files (generated from notebooks)

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

2. **Generate required data files:**
   ```bash
   # Run the data cleaning notebook
   jupyter notebook notebooks/data_clean.ipynb
   
   # Run the model training notebook
   jupyter notebook notebooks/model.ipynb
   ```

3. **Run the application:**
   ```bash
   python run_app.py
   ```
   
   Or directly with Streamlit:
   ```bash
   streamlit run app.py
   ```

## 📱 Application Features

### 🏠 Home Page
- Welcome message and project overview
- Quick access to prediction methods
- Recent predictions history
- Application statistics

### 📍 Location-Based Prediction
- **Input**: Select state, district, and year
- **Output**: Top recommended crops based on historical data
- **Visualizations**: 
  - Bar charts of crop performance
  - Historical yield trends
  - Success probability distributions

### 🔬 Parameter-Based Prediction
- **Input Methods**:
  - Manual parameter entry (sliders and dropdowns)
  - Upload soil test reports (CSV/Excel)
- **Parameters**:
  - Soil conditions (pH, organic carbon, nutrients)
  - Water quality (pH, hardness, nitrate levels)
  - Environmental factors (rainfall, year)
  - Location and crop type
- **Output**:
  - Yield predictions with confidence intervals
  - Model-wise predictions comparison
  - SHAP feature importance explanations

### 📊 Analytics Dashboard
- **Performance Metrics**: Total records, unique crops, states covered
- **Crop Analysis**: Top performing crops, distribution charts
- **Regional Analysis**: State and district-wise performance
- **Trends**: Yearly yield patterns, seasonal analysis

### 💬 AI Assistant
- **Chatbot Interface**: Natural language queries
- **SHAP-Powered Explanations**: Understand prediction reasoning
- **Farming Advice**: Soil health, yield optimization, weather considerations
- **Contextual Responses**: Based on user queries about crops, soil, weather

### 📚 Help & Guide
- **Usage Instructions**: Step-by-step guides for each feature
- **Parameter Explanations**: What each input parameter means
- **Tips for Better Predictions**: Best practices and recommendations

## 🔧 Technical Features

### Model Integration
- **8+ ML Models**: XGBoost, Random Forest, Gradient Boosting, CatBoost, LightGBM, Ridge, MLP, Bayesian Ridge
- **Ensemble Predictions**: Combines multiple models for better accuracy
- **Model Comparison**: Side-by-side performance analysis

### SHAP Integration
- **Feature Importance**: Understand which factors most influence predictions
- **Individual Explanations**: Why specific predictions were made
- **Interactive Visualizations**: SHAP summary plots and feature analysis

### Data Processing
- **Input Validation**: Ensures parameters are within valid ranges
- **Data Normalization**: Handles different input formats
- **Error Handling**: Graceful handling of missing data or models

## 📊 Visualizations

### Interactive Charts
- **Plotly Integration**: Interactive bar charts, line graphs, pie charts
- **Real-time Updates**: Charts update based on user inputs
- **Responsive Design**: Works on desktop and mobile devices

### SHAP Visualizations
- **Summary Plots**: Feature importance across all predictions
- **Individual Explanations**: Single prediction breakdowns
- **Feature Interactions**: How parameters work together

## 🎯 User Experience

### Farmer-Friendly Interface
- **Intuitive Design**: Easy-to-use sliders and dropdowns
- **Clear Labels**: Agricultural terminology with explanations
- **Mobile Responsive**: Works on smartphones and tablets

### Personalization
- **Prediction History**: Track previous predictions
- **Custom Preferences**: Save favorite crops and locations
- **Export Options**: Download prediction reports

## 🔍 Model Performance

### Accuracy Metrics
- **R² Score**: Most models achieve >0.99
- **Confidence Intervals**: Statistical uncertainty quantification
- **Cross-Validation**: Robust performance across different regions

### SHAP Insights
- **Feature Ranking**: Most important factors for crop yield
- **Interaction Effects**: How parameters combine to affect yield
- **Regional Patterns**: Different factors matter in different areas

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py --server.port 8501
```

### Production Deployment
- **Streamlit Cloud**: Deploy directly from GitHub
- **Docker**: Containerized deployment
- **AWS/GCP**: Cloud deployment options

## 📁 File Structure

```
├── app.py                          # Main Streamlit application
├── utils.py                        # Utility functions
├── run_app.py                      # Application runner
├── requirements_streamlit.txt      # Python dependencies
├── README_STREAMLIT.md             # This file
└── notebooks/                      # Jupyter notebooks
    ├── data_clean.ipynb            # Data preprocessing
    ├── model.ipynb                 # Model training
    ├── Normalized_Dataset.csv      # Processed data
    ├── Cleaned_Dataset.csv        # Cleaned data
    └── *.pkl                       # Trained models
```

## 🛠️ Troubleshooting

### Common Issues

1. **Missing Data Files**
   - Run the notebooks first to generate CSV files
   - Ensure notebooks are in the correct directory

2. **Model Loading Errors**
   - Check that .pkl files exist in notebooks/ directory
   - Verify model training completed successfully

3. **SHAP Import Errors**
   - Install SHAP: `pip install shap`
   - Some features will be limited without SHAP

4. **Memory Issues**
   - Reduce batch sizes for large datasets
   - Use model caching for better performance

### Performance Tips
- **Model Caching**: Models are cached for faster loading
- **Data Caching**: Datasets are cached to reduce I/O
- **Progressive Loading**: Large visualizations load progressively

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of a capstone project for crop prediction using machine learning.

## 🙏 Acknowledgments

- **Data Source**: Agricultural dataset with soil and environmental parameters
- **ML Models**: Scikit-learn, XGBoost, CatBoost, LightGBM
- **Visualization**: Plotly, SHAP, Streamlit
- **Framework**: Streamlit for web application

---

**🌾 Built with ❤️ for farmers and agricultural researchers**
