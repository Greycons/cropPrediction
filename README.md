# cropPrediction
The Crop Prediction Capstone Project aims to develop a data-driven model that predicts the most suitable crop to cultivate based on various environmental and soil parameters.

## 🌟 Features

- **Location-Based Crop Prediction**: Get crop recommendations based on your state and district
- **Parameter-Based Prediction**: Enter soil and environmental parameters for personalized recommendations
- **Multiple ML Models**: Uses 8 different machine learning models for accurate predictions
- **AI Assistant** (NEW): Powered by Google Gemini for intelligent farming advice
- **Analytics Dashboard**: Comprehensive visualizations and insights

## 🤖 AI Assistant with Gemini

The application includes an AI Assistant powered by Google's Gemini LLM. To use this feature:

1. **Install the packages**: `pip install google-generativeai python-dotenv`
2. **Get your API key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
3. **Create a `.env` file** in the project root with:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```
4. **Restart the app**

For detailed setup instructions, see [GEMINI_SETUP.md](GEMINI_SETUP.md)

**Note**: The AI Assistant works in fallback mode if the Gemini API is not configured.