import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import math
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer # Required for Attention layer, if it was used (good to keep)
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Apple Stock Price Prediction (RNN)",
    layout="wide",
    page_icon="📈"
)

# A placeholder for the Attention layer in case your final model used it,
# although it wasn't in the provided code, it's safer to include if you might
# have trained a slightly different model.
# If your model fails to load, try removing the `custom_objects` argument.
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], input_shape[-1]), initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[-1],), initializer='zeros', trainable=True)
        self.u = self.add_weight(name='att_u', shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
        super(Attention, self).build(input_shape)
    def call(self, inputs):
        u_t = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        att = tf.nn.softmax(tf.tensordot(u_t, self.u, axes=1), axis=1)
        output = tf.reduce_sum(inputs * att, axis=1)
        return output
        
custom_objects = {'Attention': Attention}

# ==========================================
# 2. MODEL AND DATA LOADING
# ==========================================
@st.cache_resource
def load_all_assets():
    st.write("Loading models and data...")
    
    # Check for required files
    required_files = [
        "minmax_scaler.pkl", "rnn_model.h5", "lstm_model.h5", 
        "gru_model.h5", "metadata.pkl"
    ]
    for f in required_files:
        if not os.path.exists(f):
            st.error(f"Missing file: {f}. Please run the export script in the notebook.")
            st.stop()
            
    # Load Scaler
    scaler = joblib.load("minmax_scaler.pkl")
    
    # Load Models
    models = {
        'RNN': load_model("rnn_model.h5", custom_objects=custom_objects),
        'LSTM': load_model("lstm_model.h5", custom_objects=custom_objects),
        'GRU': load_model("gru_model.h5", custom_objects=custom_objects)
    }
    
    # Load Metadata
    with open("metadata.pkl", "rb") as f:
        meta = pickle.load(f)
        seq_len = meta['seq_len']
        features_cols = meta['features_cols']
        scaled_data = meta['scaled_data_for_prediction']
        last_date = meta['last_data_date']
        
    return scaler, models, seq_len, features_cols, scaled_data, last_date

try:
    scaler, models, SEQ_LEN, features_cols, scaled_data, last_known_date = load_all_assets()
    st.success(f"Models and data loaded. Last known data point: {last_known_date.strftime('%Y-%m-%d')}")
except Exception as e:
    st.error(f"Error during asset loading: {e}")
    st.info("Check if your model files (e.g., rnn_model.h5) were created correctly and if the 'Attention' class is needed.")
    st.stop()

# ==========================================
# 3. PREDICTION UTILITY FUNCTIONS
# ==========================================
def predict_next_n_days(model, n_days, initial_scaled_data, seq_len, features_cols):
    
    # 1. Get the last sequence from the loaded data
    # Create a deep copy to avoid modifying the cached data
    temp_input = list(initial_scaled_data[-seq_len:].copy()) 
    lst_output = []
    i = 0
    
    # 2. Find the index of the 'Close' price target
    target_index = features_cols.index('Close')
        
    # 3. Predict day by day (Roll Forward)
    while i < n_days:
        if len(temp_input) > seq_len:
            x_input = np.array(temp_input[1:])
        else:
            x_input = np.array(temp_input)
            
        # Reshape for the RNN model (1, seq_len, n_features)
        x_input = x_input.reshape(1, seq_len, len(features_cols))
        
        # Predict the next CLOSE price (which is scaled)
        y_pred_scaled = model.predict(x_input, verbose=0)[0] 
        lst_output.append(y_pred_scaled[0])
        
        # --- Update the sequence for the next prediction ---
        # Take the last known full feature vector
        next_feature_vector = temp_input[-1].copy()
        
        # Update the 'Close' value in the feature vector with the new prediction
        next_feature_vector[target_index] = y_pred_scaled[0]
        
        # Append the new vector to the sequence
        temp_input.append(next_feature_vector)
        i += 1
        
    # 4. Inverse transform the predicted 'Close' prices
    # Create a dummy array filled with 0s, with the predicted Close prices in the target column
    prediction_dummy = np.zeros((n_days, len(features_cols)))
    prediction_dummy[:, target_index] = lst_output
    
    # Inverse transform the dummy array
    inversed_prediction_data = scaler.inverse_transform(prediction_dummy)
    
    # Extract only the inverse transformed Close price
    final_predictions = inversed_prediction_data[:, target_index]
    
    return final_predictions.tolist()

# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.title("📈 Apple Stock Price Prediction with RNNs")
st.markdown(f"Forecasting future **Close** prices based on the past {SEQ_LEN} days of market data.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Prediction Settings")
    model_choice = st.selectbox("Select Model:", list(models.keys()))
    n_days = st.slider("Days to Predict Forward:", min_value=1, max_value=30, value=7)
    
    # Display key model parameters
    st.markdown("---")
    st.subheader("Model Parameters")
    st.metric("Sequence Length (Time Steps)", SEQ_LEN)
    st.metric("Input Features", f"{len(features_cols)} ({', '.join(features_cols)})")
    st.info(f"The model uses the past **{SEQ_LEN} days** of data to predict the next day's Close price.")

# --- MAIN DASHBOARD ---
model_to_use = models[model_choice]

if st.button(f"Generate Prediction for Next {n_days} Days ({model_choice})", type="primary"):
    
    st.subheader(f"Results for **{model_choice}** Model")
    
    with st.spinner(f"Predicting next {n_days} days..."):
        # Generate predictions
        predicted_prices = predict_next_n_days(
            model_to_use, n_days, scaled_data, SEQ_LEN, features_cols
        )
        
        # Generate prediction dates (starting one day after the last known date)
        prediction_dates = [last_known_date + timedelta(days=i + 1) for i in range(n_days)]
        
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
            
            # Show conceptual diagram of the chosen RNN architecture
            if model_choice == 'GRU':
                st.info("The GRU uses a Reset and an Update Gate to efficiently manage information flow.") 
            elif model_choice == 'LSTM':
                 st.info("The LSTM uses Input, Forget, and Output Gates to combat the vanishing gradient problem.") 
            elif model_choice == 'RNN':
                 st.info("The Simple RNN processes sequences but often suffers from short-term memory.") 

        st.balloons()