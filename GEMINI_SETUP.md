# 🌟 Gemini AI Assistant Setup Guide

This guide explains how to set up and use the Google Gemini AI integration for the AI Assistant feature in the Crop Prediction application.

## 🎯 What is Gemini?

Google Gemini is a powerful language model that provides intelligent, context-aware responses about farming, crop prediction, soil health, and agricultural best practices.

## 📋 Prerequisites

- Python 3.8 or higher
- Internet connection (required for API calls)
- Google account

## 🔧 Setup Instructions

### Step 1: Install Required Packages

Install the required packages:

```bash
pip install google-generativeai python-dotenv
```

**Important**: If you already have `google-generativeai` installed, make sure it's the latest version:

```bash
pip install --upgrade google-generativeai python-dotenv
```

Or if using the full requirements:

```bash
pip install -r requirements.txt
```

Or if using Streamlit requirements:

```bash
pip install -r requirements_streamlit.txt
```

### Step 2: Get Your API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key" or "Get API Key"
4. Copy the generated API key (you'll need this in the next step)

### Step 3: Create .env File

1. Copy the example file:
   ```bash
   copy env.example .env
   ```
   (On Linux/Mac: `cp env.example .env`)

2. Open the `.env` file in a text editor

3. Replace `your_api_key_here` with your actual API key:
   ```env
   GOOGLE_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
   ```

4. Save the file

**Important**: The `.env` file should be in the same directory as `app.py` (the project root).

### Step 4: Restart the Application

Restart the Streamlit app for the changes to take effect:

### Step 5: Verify Setup

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Navigate to "💬 AI Assistant" page in the sidebar

3. If setup is successful, you should see the chat interface without any warning messages

## 🔄 Alternative: Using Environment Variables

If you prefer to use environment variables instead of a `.env` file:

### Windows (PowerShell)
```powershell
[System.Environment]::SetEnvironmentVariable('GOOGLE_API_KEY', 'YOUR_API_KEY', 'User')
```

### Windows (CMD)
```cmd
setx GOOGLE_API_KEY "YOUR_API_KEY"
```

### Linux/Mac
```bash
export GOOGLE_API_KEY='YOUR_API_KEY'
```

Then restart the application.

## 🎨 How to Use

1. **Navigate** to the "💬 AI Assistant" page from the sidebar
2. **Type** your question in the chat input box at the bottom
3. **Get** intelligent, contextual responses about:
   - Crop selection and recommendations
   - Soil health and management
   - Yield optimization strategies
   - Weather and climate considerations
   - Farming best practices

## 💡 Example Questions

- "What crops grow best in soil with pH 7.2?"
- "How can I improve my crop yield?"
- "What is the ideal rainfall for rice cultivation?"
- "How do I manage nitrogen levels in my soil?"
- "Explain the importance of organic carbon in soil"

## 🚨 Troubleshooting

### Issue: "GOOGLE_API_KEY not found in .env file or environment variables"

**Solution**: Make sure you:
1. Created a `.env` file in the project root (same directory as `app.py`)
2. Added `GOOGLE_API_KEY=your_actual_api_key` to the `.env` file
3. Installed `python-dotenv` package: `pip install python-dotenv`
4. Restarted the Streamlit app

### Issue: "Invalid API key"

**Solution**:
1. Verify your API key is correct in the `.env` file
2. Make sure there are no extra spaces around the `=` sign
3. Ensure you got the key from [Google AI Studio](https://makersuite.google.com/app/apikey)
4. Check that the key hasn't been revoked or expired

### Issue: "google-generativeai not installed"

**Solution**:
```bash
pip install google-generativeai python-dotenv
```

### Issue: ".env file not loading"

**Solution**:
1. Make sure the `.env` file is in the correct location (same directory as `app.py`)
2. Check that the file is actually named `.env` (not `env.txt` or anything else)
3. Verify you installed `python-dotenv`: `pip install python-dotenv`
4. Restart the Streamlit app completely

### Issue: "404 models/gemini-pro is not found"

**Solution**: This error means the model name is deprecated. The app now automatically tries multiple model names:
1. `gemini-1.5-flash` (fast, latest)
2. `gemini-1.5-pro` (high quality, latest)
3. `gemini-pro` (legacy)

If all models fail, check:
- Your API key has access to Gemini models
- Update your `google-generativeai` package: `pip install --upgrade google-generativeai`
- Try generating a new API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Issue: API calls are failing

**Possible causes**:
- No internet connection
- Invalid API key
- API quota exceeded
- Network firewall blocking requests

**Solution**:
1. Check your internet connection
2. Verify your API key is valid
3. Check API usage in [Google AI Studio](https://makersuite.google.com/app/apikey)
4. Try again later

## 🔄 Fallback Mode

If the API is not available or not set up, the AI Assistant will automatically use a fallback mode with local responses based on keyword matching. This ensures the application always works, even without the Gemini API.

## 💰 Pricing

Google Gemini API offers free tier usage for development. Check [Google AI Studio](https://makersuite.google.com/app/apikey) for current pricing and quotas.

## 🔒 Security Notes

- **Never commit** your `.env` file to version control
- Keep your API key secret
- The `.env` file is stored locally on your machine
- The `.env` file should be listed in `.gitignore` to prevent accidental commits
- Your API key is loaded securely when the app starts

## 📚 Additional Resources

- [Google Gemini Documentation](https://ai.google.dev/docs)
- [Google AI Studio](https://makersuite.google.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## ✅ Verification

To verify your setup is working:

1. Start the app: `streamlit run app.py`
2. Go to "💬 AI Assistant"
3. Ask: "What is the best soil pH for crops?"
4. If you get a detailed, contextual response, your setup is successful!

---

**Need help?** Check the issue on the GitHub repository or refer to the main README.md file.

