import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Apple Stock Prediction Dashboard (All Models)",
    layout="wide",
    page_icon="📈"
)

# Define file paths
SCALER_FILE = "minmax_scaler.pkl"
META_FILE = "metadata.pkl"
RESULTS_FILE = "comparison_results.csv"
MODEL_FILES = {
    'RNN': "rnn_model.h5",
    'LSTM': "lstm_model.h5",
    'GRU': "gru_model.h5"
}
CUSTOM_OBJECTS = {'mse': 'mse', 'mae': 'mae'} # For Keras loading fix


@st.cache_resource
def load_prediction_assets():
    """Loads all models, scaler, and metadata."""

    
    # 1. Check for required files
    required_files = list(MODEL_FILES.values()) + [SCALER_FILE, META_FILE, RESULTS_FILE]
    for f in required_files:
        if not os.path.exists(f):
            st.error(f"Missing required file: {f}. Please run the export script in the notebook.")
            st.stop()
            
    # 2. Load Models
    models = {}
    for name, path in MODEL_FILES.items():
        models[name] = load_model(path, custom_objects=CUSTOM_OBJECTS)
    
    # 3. Load Scaler
    scaler = joblib.load(SCALER_FILE)
    
    # 4. Load Metadata
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
        
    # 5. Load Comparison Data
    comparison_df = pd.read_csv(RESULTS_FILE)
    
    return scaler, models, meta, comparison_df

try:
    SCALER, MODELS, META, comparison_df = load_prediction_assets()
    SEQ_LEN = META['seq_len']
    FEATURES_COLS = META['features_cols']
    SCALED_DATA = META['scaled_data_for_prediction']
    LAST_KNOWN_DATE = pd.to_datetime(META['last_data_date']).date()
    st.success(f"All 3 models and data loaded successfully. Last known date: {LAST_KNOWN_DATE}")
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


# --- Create Tabs ---
tab1, tab2 = st.tabs(["📊 Performance Analysis", "🔮 Live Prediction"])

# --- Tab 1: Performance Analysis ---
with tab1:
    st.title("📊 Model Workflow Comparison")
    st.markdown("Review of the Mean Absolute Error (MAE) achieved by different RNN models using Manual vs. Scikit-learn Pipeline preprocessing.")

    col_chart, col_table = st.columns([2, 1])

    with col_chart:
        st.subheader("MAE Comparison Chart")
        
        # Recreate the comparison bar plot
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
        
        best_mae = min(comparison_df["Manual MAE"].min(), comparison_df["Pipeline MAE"].min())
        st.info(f"**Overall Best MAE:** {best_mae:.4f}")
        st.markdown(
            """
            ### Methodology
            The **End-to-End Pipeline** uses Scikit-learn's `ColumnTransformer` and `Pipeline`  to ensure that scaling is consistently and correctly applied to all feature columns before training.
            """
        )
            
# --- Tab 2: Live Prediction ---
with tab2:
    st.title("🔮 Live Stock Price Forecasting")
    st.markdown(f"Forecasting the next few days using your choice of model (Last data point: **{LAST_KNOWN_DATE}**).")
    
    with st.sidebar:
        st.header("Prediction Settings")
        model_choice = st.selectbox("Select Model:", list(MODELS.keys()))
        n_days = st.slider("Days to Predict Forward:", min_value=1, max_value=30, value=7)
        
        st.markdown("---")
        st.subheader("Model Parameters")
        st.metric("Sequence Length", SEQ_LEN)
        st.metric("Input Features", len(FEATURES_COLS))

    if st.button(f"Generate Forecast for Next {n_days} Days ({model_choice})", type="primary"):
        
        model_to_use = MODELS[model_choice]
        
        with st.spinner(f"Forecasting next {n_days} days using {model_choice}..."):
            predicted_prices = predict_next_n_days(
                model_to_use, n_days, SCALED_DATA, SEQ_LEN, FEATURES_COLS, SCALER
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

