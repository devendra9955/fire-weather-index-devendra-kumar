import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("Linear_regression_models.pkl")
scaler = joblib.load("Scalers.pkl")

st.set_page_config(
    page_title="Fire Weather Index Predictor",
    page_icon="",
    layout="centered",
)

st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #16222A, #3A6073);
            color: #fff;
        }
        .title {
            font-family: 'Segoe UI';
            text-align: center;
            color: #FFD369;
            margin-bottom: 0px;
        }
        .subtitle {
            text-align: center;
            font-size: 16px;
            color: #ccc;
            margin-bottom: 30px;
        }
        .result-box {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            margin-top: 20px;
        }
        .fire-low {
            color: #2ECC71;
            font-weight: bold;
        }
        .fire-medium {
            color: #F1C40F;
            font-weight: bold;
        }
        .fire-high {
            color: #E74C3C;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>Fire Weather Index (FWI) Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Estimate fire risk using daily weather conditions.</p>", unsafe_allow_html=True)

features = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI']

with st.form(key='fwi_form'):
    col1, col2 = st.columns(2)
    with col1:
        Temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0, 0.1)
        RH = st.number_input("Relative Humidity (%)", 0.0, 100.0, 45.0, 0.1)
        Ws = st.number_input("Wind Speed (km/h)", 0.0, 100.0, 15.0, 0.1)
        Rain = st.number_input("Rainfall (mm)", 0.0, 50.0, 0.0, 0.1)
        FFMC = st.number_input("Fine Fuel Moisture Code (FFMC)", 0.0, 100.0, 85.0, 0.1)
    with col2:
        DMC = st.number_input("Duff Moisture Code (DMC)", 0.0, 200.0, 25.0, 0.1)
        DC = st.number_input("Drought Code (DC)", 0.0, 800.0, 120.0, 0.1)
        ISI = st.number_input("Initial Spread Index (ISI)", 0.0, 30.0, 10.0, 0.1)
        BUI = st.number_input("Buildup Index (BUI)", 0.0, 200.0, 60.0, 0.1)

    submitted = st.form_submit_button("Predict Fire Weather Index ")

if submitted:
    df_input = pd.DataFrame([[
        Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI
    ]], columns=features)

    scaled_data = scaler.transform(df_input)
    predicted_fwi = model.predict(scaled_data)[0]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>Prediction Result</h3>", unsafe_allow_html=True)

    if predicted_fwi < 5:
        level = "Low"
        style = "fire-low"
    elif 5 <= predicted_fwi < 15:
        level = "Moderate"
        style = "fire-medium"
    else:
        level = "High"
        style = "fire-high"

    st.markdown(f"""
        <div class='result-box'>
            <h2>Predicted FWI: {predicted_fwi:.2f}</h2>
         <h3 class='{style}'> Fire Risk Level: {level}</h3>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br><hr><p style='text-align:center;color:#aaa;'> Developed by Devendra </p>", unsafe_allow_html=True)
