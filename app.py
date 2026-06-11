import streamlit as st
import numpy as np
import pandas as pd
import joblib

@st.cache_resource 
def load_model():
    return joblib.load("RF_model.pkl")

model = load_model()

st.set_page_config(page_title="Apple Stock Predictor", layout="centered")
st.title("Apple Price Predictor")
st.markdown("""
This application uses a machine learning **Random Forest Regressor** to predict tomorrow's expected percentage return based on technical indicators.
""")

col1, col2 = st.columns(2)

with col1:
    # Make sure this variable name is EXACTLY 'close_now'
    close_now = st.number_input("Today's Close Price ($ Close)", min_value=0.1, value=150.0, step=0.1)
    close_lag1 = st.number_input("Yesterday's Close Price ($ Close_lag1)", min_value=0.1, value=149.0, step=0.1)

with col2:
    rsi_14 = st.slider("Current RSI (RSI_14)", min_value=0.0, max_value=100.0, value=55.0, step=0.1)
    macd = st.number_input("Current MACD Value (MACD)", min_value=-10.0, max_value=10.0, value=0.5, step=0.01)


st.info("Remember An RSI above 70 means Overbought. A positive MACD means upward momentum.")  


# Look for this block inside your app.py file:
if st.button("🚀 Predict Tomorrow's Return", type="primary"):
    
    # FIX: Make sure input_data has EXACTLY 4 elements inside the double brackets [[ ]]
    # The order must match your training columns: ['Close', 'Close_lag1', 'RSI_14', 'MACD']
    input_data = np.array([[close_now, close_lag1, rsi_14, macd]])
    
    # Now this will run perfectly without the feature mismatch error!
    prediction = model.predict(input_data)[0]
    
    st.subheader("🎯 Model Prediction:")
    if prediction > 0:
        st.success(f"**Bullish Signal:** Expected tomorrow return: **+{prediction:.4%}**")
    elif prediction < 0:
        st.error(f"**Bearish Signal:** Expected tomorrow return: **{prediction:.4%}**")
    else:
        st.warning(f"**Neutral:** Expected tomorrow return is flat (**0.00%**).")

    # Interactive expected absolute change display
    expected_change = close_now * prediction
    st.write(f"Based on Today's Close of *${close_now}*, this implies an estimated change of **${expected_change:+.2f}** tomorrow.") 

  

