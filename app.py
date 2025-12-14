import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import math
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Apple Stock Prediction Dashboard",
    layout="wide",
    page_icon="📈"
)

# Define file paths
MODEL_FILE = os.path.join(os.getcwd(), 'lstm_best_model.h5') # Assuming LSTM was the best, change if necessary
SCALER_FILE = "minmax_scaler.pkl"
META_FILE = "metadata.pkl"
RESULTS_FILE = "comparison_results.csv"

# ==========================================
# 2. ASSET LOADING (Unified Function)
# ==========================================

# FIX: Use string identifiers for metrics to bypass Keras deserialization errors
CUSTOM_OBJECTS = {'mse': 'mse', 'mae': 'mae'}

@st.cache_resource
def load_prediction_assets(model_path, scaler_path, meta_path):
    """Loads all models, scaler, and metadata for prediction."""
    st.write("Loading prediction assets...")
    
    required = [model_path, scaler_path, meta_path]
    for f in required:
        if not os.path.exists(f):
            st.error(f"Missing prediction file: {f}. Please run the export script in the notebook.")
            st.stop()
            
    # Load Model
    model = load_model(model_path, custom_objects=CUSTOM_OBJECTS)
    
    # Load Scaler
    scaler = joblib.load(scaler_path)
    
    # Load Metadata
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
        
    return model, scaler, meta

@st.cache_data
def load_comparison_data(file_path):
    """Loads the pre-calculated performance results."""
    if not os.path.exists(file_path):
        st.error(f"Missing results file: {file_path}. Please run the export script in the notebook.")
        return pd.DataFrame()
    return pd.read_csv(file_path)

try:
    # We load the comparison data first for the first tab
    comparison_df = load_comparison_data(RESULTS_FILE)
    
    # We load the prediction assets only if the files exist
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(META_FILE):
        PREDICTION_MODEL, SCALER, META = load_prediction_assets(MODEL_FILE, SCALER_FILE, META_FILE)
        SEQ_LEN = META['seq_len']
        FEATURES_COLS = META['features_cols']
        SCALED_DATA = META['scaled_data_for_prediction']
        LAST_KNOWN_DATE = pd.to_datetime(META['last_data_date']).date()
        PREDICTION_READY = True
    else:
        PREDICTION_READY = False
        st.warning("Prediction assets (model, scaler, metadata) are missing. Only the Analysis tab will be active.")
        
except Exception as e:
    st.error(f"Fatal Error during asset loading: {e}")
    st.stop()

# ==========================================
# 3. PREDICTION UTILITY FUNCTIONS
# ==========================================
def predict_next_n_days(model, n_days, initial_scaled_data, seq_len, features_cols, scaler):
    
    temp_input = list(initial_scaled_data.copy()) 
    lst_output = []
    
    # Find the index of the 'Close' price target
    target_index = features_cols.index('Close')
        
    for i in range(n_days):
        
        # FIX: Always take the last SEQ_LEN elements for the input sequence
        current_sequence = np.array(temp_input[-seq_len:]) 
        
        # Reshape for the RNN model (1, seq_len, n_features)
        x_input = current_sequence.reshape(1, seq_len, len(features_cols))
        
        # Predict the next CLOSE price (which is scaled)
        y_pred_scaled = model.predict(x_input, verbose=0)[0] 
        lst_output.append(y_pred_scaled[0])
        
        # --- Update the sequence for the next prediction ---
        next_feature_vector = current_sequence[-1].copy() # Start with the last vector's features
        next_feature_vector[target_index] = y_pred_scaled[0] # Inject the new predicted 'Close'
        temp_input.append(next_feature_vector)
        
    # Inverse transform the predicted 'Close' prices
    prediction_dummy = np.zeros((n_days, len(features_cols)))
    prediction_dummy[:, target_index] = lst_output
    
    inversed_prediction_data = scaler.inverse_transform(prediction_dummy)
    final_predictions = inversed_prediction_data[:, target_index]
    
    return final_predictions.tolist()


# ==========================================
# 4. STREAMLIT UI
# ==========================================

# --- Create Tabs ---
tab1, tab2 = st.tabs(["📊 Performance Analysis", "🔮 Live Prediction"])

# --- Tab 1: Performance Analysis ---
with tab1:
    st.title("📊 Model Workflow Comparison")
    st.markdown("Review of the Mean Absolute Error (MAE) achieved by different RNN models using Manual vs. Scikit-learn Pipeline preprocessing.")

    if not comparison_df.empty:
        col_chart, col_table = st.columns([2, 1])

        with col_chart:
            st.subheader("MAE Comparison Chart")
            
            # Recreate the comparison bar plot using Matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(comparison_df["Model"]))
            width = 0.35

            plt.bar(x - width/2, comparison_df["Manual MAE"], width, label="Manual Preprocessing", color='#1f77b4')
            plt.bar(x + width/2, comparison_df["Pipeline MAE"], width, label="End-to-End Pipeline", color='#ff7f0e')

            plt.xticks(x, comparison_df["Model"])
            plt.ylabel("Mean Absolute Error (MAE)")
            plt.title("Manual vs Pipeline Model Performance Comparison")
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig) 

        with col_table:
            st.subheader("Error Summary")
            st.dataframe(
                comparison_df.style.format({"Manual MAE": "{:.4f}", "Pipeline MAE": "{:.4f}"}),
                use_container_width=True
            )
            
            # Find the best performing model/method
            best_mae = min(comparison_df["Manual MAE"].min(), comparison_df["Pipeline MAE"].min())
            st.info(f"**Overall Best MAE:** {best_mae:.4f}")
            st.markdown("""
            The **End-to-End Pipeline** using Scikit-learn's `ColumnTransformer` provides consistency and reproducibility, which is critical for real-world deployment.
            """)
            
# --- Tab 2: Live Prediction ---
with tab2:
    st.title("🔮 Live Stock Price Forecasting")
    st.markdown(f"Predicting the next few trading days using the best trained model (Last known data point: **{LAST_KNOWN_DATE}**).")
    
    if PREDICTION_READY:
        
        with st.sidebar:
            st.header("Prediction Settings")
            n_days = st.slider("Days to Predict Forward:", min_value=1, max_value=30, value=7)
            
            st.markdown("---")
            st.subheader("Model Info")
            st.metric("Model Used", PREDICTION_MODEL.name.upper() if PREDICTION_MODEL.name else "Keras Model")
            st.metric("Sequence Length", SEQ_LEN)
            st.metric("Input Features", len(FEATURES_COLS))

        if st.button(f"Generate Forecast for Next {n_days} Days", type="primary"):
            
            with st.spinner(f"Forecasting next {n_days} days..."):
                predicted_prices = predict_next_n_days(
                    PREDICTION_MODEL, n_days, SCALED_DATA, SEQ_LEN, FEATURES_COLS, SCALER
                )
                
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
            
    else:
        st.error("Prediction is disabled. Required model files (e.g., `lstm_best_model.h5`, `minmax_scaler.pkl`, `metadata.pkl`) are missing.")
