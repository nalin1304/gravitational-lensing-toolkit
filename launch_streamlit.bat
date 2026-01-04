@echo off
echo.
echo ========================================
echo   GRAVITATIONAL LENSING PLATFORM
echo ========================================
echo.
echo Starting Multi-Page Streamlit Application...
echo.
echo Features Available:
echo   - Simple Lensing Visualization
echo   - PINN Neural Network Inference
echo   - Multi-Plane Lensing Systems
echo   - Real FITS Data Analysis
echo   - Scientific Validation
echo   - Bayesian Uncertainty Quantification
echo.
echo Server will start on: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m streamlit run app/Home.py

pause
