#!/usr/bin/env python3
"""
Crop Prediction AI - Streamlit Application Runner
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements_streamlit.txt")
        return False
    
    return True

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        'notebooks/Normalized_Dataset.csv',
        'notebooks/Cleaned_Dataset.csv'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Missing data files: {', '.join(missing_files)}")
        print("Please run the notebooks first to generate the required CSV files.")
        return False
    
    return True

def main():
    """Main function to run the Streamlit app"""
    print("🌾 Crop Prediction AI - Starting Application")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check data files
    if not check_data_files():
        print("\nTo generate the required data files:")
        print("1. Run notebooks/data_clean.ipynb")
        print("2. Run notebooks/model.ipynb")
        print("3. Then run this application again")
        sys.exit(1)
    
    # Run Streamlit app
    try:
        print("\n🚀 Starting Streamlit application...")
        print("The app will open in your default browser.")
        print("Press Ctrl+C to stop the application.")
        print("=" * 50)
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user.")
    except Exception as e:
        print(f"\n❌ Error running application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
