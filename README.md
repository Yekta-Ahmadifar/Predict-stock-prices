# Apple Stock Return Predictor

An end-to-end Machine Learning and Data Science project that analyzes historical Apple Inc. (AAPL) stock data and predicts tomorrow's percentage return using a tree-based regression model.

## Project Overview
* **Exploratory Data Analysis (EDA):** Analyzed 30+ years of historical stock data to understand market phases, statistical distributions of risk, and feature engineering viability.
* **Feature Engineering:** Extracted normalized time-series features including 14-day RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence), and historical lag variables to capture trend momentum.
* **Machine Learning Model:** Evaluated Linear Regression and Random Forest Regressors, tuning tree depth to optimize Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). 
* **Target Normalization:** Solved historical scale mismatch issues by training the model to predict **percentage returns** rather than absolute dollar changes.
* **Web Application:** Built an interactive web dashboard using **Streamlit** for live predictions based on custom technical user inputs.

## How to Run the App

1. Clone this repository and ensure your model file (`apple_rf_model.pkl`) is in the directory.
2. Install dependencies:
   ```bash
   pip install streamlit pandas numpy scikit-learn joblib
