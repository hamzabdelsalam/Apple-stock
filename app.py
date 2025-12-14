import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
from datetime import timedelta
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Apple Stock Price Forecast (RNN/LSTM/GRU)",
    layout="wide",
    page_icon="🔮"
)

# Define file paths
SCALER_FILE = "minmax_scaler.pkl"
META_FILE = "metadata.pkl"
MODEL_FILES = {
    'RNN': "rnn_model.h5",
    'LSTM': "lstm_model.h5",
    'GRU': "gru_model.h5"
}
CUSTOM_OBJECTS = {'mse': 'mse', 'mae': 'mae'} # For Keras loading fix

# ==========================================
# 2. MODEL AND DATA LOADING
# ==========================================

@st.cache_resource
def load_prediction_assets():
    """Loads all models, scaler, and metadata."""
    
    # 1. Check for required files (Removed RESULTS_FILE dependency)
    required_files = list(MODEL_FILES.values()) + [SCALER_FILE, META_FILE]
    for f in required_files:
        if not os.path.exists(f):
            st.error(f"Missing required file: {f}. Please ensure you ran the export script in the notebook.")
            st.stop()
            
    st.info("Loading models: RNN, LSTM, GRU...")
    
    # 2. Load Models
    models = {}
    for name, path in MODEL_FILES.items():
        models[name] = load_model(path, custom_objects=CUSTOM_OBJECTS)
    
    # 3. Load Scaler
    scaler = joblib.load(SCALER_FILE)
    
    # 4. Load Metadata
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
        
    return scaler, models, meta

try:
    SCALER, MODELS, META = load_prediction_assets()
    SEQ_LEN = META['seq_len']
    FEATURES_COLS = META['features_cols']
    SCALED_DATA = META['scaled_data_for_prediction']
    LAST_KNOWN_DATE = pd.to_datetime(META['last_data_date']).date()
    st.success(f"All 3 models and data loaded successfully. Last known data point: {LAST_KNOWN_DATE}")
    PREDICTION_READY = True
except Exception as e:
    st.error(f"Fatal Error during asset loading: {e}")
    PREDICTION_READY = False
    st.stop()

# ==========================================
# 3. PREDICTION UTILITY FUNCTION
# ==========================================
def predict_next_n_days(model, n_days, initial_scaled_data, seq_len, features_cols, scaler):
    
    temp_input = list(initial_scaled_data.copy()) 
    lst_output = []
    target_index = features_cols.index('Close')
        
    for i in range(n_days):
        
        current_sequence = np.array(temp_input[-seq_len:]) 
        # Reshape for Keras (1, sequence_length, number_of_features)
        x_input = current_sequence.reshape(1, seq_len, len(features_cols))
        y_pred_scaled = model.predict(x_input, verbose=0)[0] 
        lst_output.append(y_pred_scaled[0])
        
        # Update the sequence for the next day's prediction (Roll Forward)
        next_feature_vector = current_sequence[-1].copy()
        next_feature_vector[target_index] = y_pred_scaled[0]
        temp_input.append(next_feature_vector)
        
    # Inverse transform the predicted 'Close' prices
    prediction_dummy = np.zeros((n_days, len(features_cols)))
    prediction_dummy[:, target_index] = lst_output
    
    inversed_prediction_data = scaler.inverse_transform(prediction_dummy)
    final_predictions = inversed_prediction_data[:, target_index]
    
    return final_predictions.tolist()

# ==========================================
# 4. STREAMLIT UI (Prediction Focus)
# ==========================================

st.title("🔮 Apple Stock Price Forecast (RNN, LSTM, GRU)")
st.markdown("Select a recurrent model and the forecast horizon to see projected Close prices for Apple stock.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Prediction Settings")
    model_choice = st.selectbox("Select Model:", list(MODELS.keys()))
    n_days = st.slider("Days to Predict Forward:", min_value=1, max_value=30, value=7)
    
    st.markdown("---")
    st.subheader("Model Parameters")
    st.metric("Sequence Length (Days)", SEQ_LEN)
    st.metric("Number of Input Features", len(FEATURES_COLS))
    st.info(f"The model uses the past **{SEQ_LEN} days** of data to predict the next day's Close price.")

# --- MAIN DASHBOARD ---
if PREDICTION_READY:
    
    if st.button(f"Generate Forecast for Next {n_days} Days using {model_choice}", type="primary"):
        
        model_to_use = MODELS[model_choice]
        
        with st.spinner(f"Forecasting next {n_days} days using {model_choice}..."):
            predicted_prices = predict_next_n_days(
                model_to_use, n_days, SCALED_DATA, SEQ_LEN, FEATURES_COLS, SCALER
            )
            
            # Generate prediction dates (starting one day after the last known date)
            prediction_dates = [LAST_KNOWN_DATE + timedelta(days=i + 1) for i in range(n_days)]
            
            predictions_df = pd.DataFrame({
                'Date': prediction_dates,
                'Predicted Close Price (USD)': predicted_prices
            })
            
        # Display results
        col_table, col_chart = st.columns([1, 2])
        
        with col_table:
            st.markdown("#### **Forecast Table**")
            st.dataframe(predictions_df.style.format({'Predicted Close Price (USD)': '${:,.2f}'}), height=250)
            
        with col_chart:
            st.markdown("#### **Forecast Chart**")
            st.line_chart(predictions_df.set_index('Date'))
            st.info(f"The prediction starts one day after the last known data point: {LAST_KNOWN_DATE}")

        st.balloons()
        
        # Display structural info about the chosen model type
        st.markdown("---")
        st.subheader(f"Understanding the {model_choice} Architecture")
        if model_choice == 'GRU':
            st.image('https://i.imgur.com/8Qj8n6T.png', caption='GRU Architecture', use_column_width=True) # 
            st.markdown("The **GRU (Gated Recurrent Unit)** uses only two gates (**Update** and **Reset**) to manage information flow, offering a computationally more efficient alternative to LSTM while still addressing the vanishing gradient problem.")
        elif model_choice == 'LSTM':
            st.image('https://i.imgur.com/Gj3Hj7Z.png', caption='LSTM Architecture', use_column_width=True) # 
            st.markdown("The **LSTM (Long Short-Term Memory)** utilizes three distinct gates (**Input, Forget, and Output**) to explicitly control which information is stored, forgotten, or passed to the next step, making it highly effective for time series with long-range dependencies.")
        elif model_choice == 'RNN':
            st.image('https://i.imgur.com/8Nf9rYd.png', caption='Simple RNN Architecture', use_column_width=True) # 
            st.markdown("The **Simple RNN** uses a basic recurrent connection, which is computationally fast but often suffers from the **vanishing gradient problem**, making it less effective at learning patterns over long time sequences.")
        
else:
    st.error("Prediction is disabled. Please ensure the necessary assets (`rnn_model.h5`, `lstm_model.h5`, `gru_model.h5`, `minmax_scaler.pkl`, `metadata.pkl`) are created by running the export script in your notebook.")
