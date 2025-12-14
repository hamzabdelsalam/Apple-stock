import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split # Used for splitting sequences
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- Define the URL for the raw CSV data (Verified public source) ---
DATA_URL = "https://raw.githubusercontent.com/mwitiderrick/stockmarket/master/AAPL.csv" 

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Apple Stock Data Analyzer & LSTM Model", 
    page_icon="🍎",
    layout="wide"
)

# --- Core Sequence Data Preprocessing Function ---
def prepare_sequence_data(data_array: np.ndarray, seq_len: int):
    """
    Transforms a 2D numpy array into sequence data (X) and a target (y).
    X shape: (N_samples, seq_len, N_features - 1)
    y shape: (N_samples,)
    """
    X, y = [], []
    n_samples = len(data_array)
    
    if data_array.shape[1] < 2:
        # This will only happen if the data is only one column, which is unlikely for stock data
        return np.array([]), np.array([])

    for i in range(seq_len, n_samples):
        # X: historical data (i-seq_len to i), excluding the last (target) column
        X.append(data_array[i-seq_len:i, :-1]) 
        # y: the target value at index i (the last column)
        y.append(data_array[i, -1]) 
    
    if not X:
        return np.array([]), np.array([])
        
    return np.array(X), np.array(y)


# ==========================================
# 2. DATA LOADING & SCALING
# ==========================================
@st.cache_data
def load_and_scale_data(data_url):
    """
    Loads data directly from a URL, cleans it, and applies MinMaxScaler.
    """
    st.info(f"Attempting to load data from public URL: **{data_url}**")
    try:
        df = pd.read_csv(data_url)
        
        # Basic cleaning and date parsing
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
        # Ensure only numeric data remains
        df_numeric = df.select_dtypes(include=np.number)
        
        if df_numeric.empty:
            st.error("The loaded data frame is empty after dropping non-numeric columns.")
            return None, None
            
        # --- Data Scaling ---
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit the scaler to all numeric data and transform
        scaled_data = scaler.fit_transform(df_numeric)
        
        # Convert back to DataFrame
        df_scaled = pd.DataFrame(scaled_data, columns=df_numeric.columns, index=df_numeric.index)
        
        st.success("🍎 Data loaded, cleaned, and scaled successfully!")
        return df_scaled, scaler
    
    except Exception as e:
        st.error(f"Error loading data from URL or processing file: {e}")
        return None, None

# ==========================================
# 3. LSTM MODEL BUILD & TRAIN
# ==========================================
def build_and_train_model(X_train, y_train, units=50, dropout=0.2, epochs=50, batch_size=32):
    """
    Builds and trains a simple LSTM model.
    """
    # X_train shape is (N_samples, seq_len, N_features). N_features is the last dimension.
    n_features = X_train.shape[2] 
    seq_len = X_train.shape[1]
    
    # 1. Build the model
    model = Sequential([
        LSTM(units=units, return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(dropout),
        LSTM(units=units),
        Dropout(dropout),
        Dense(1) # Output layer for a single prediction (the next 'Close' price)
    ])

    # 2. Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    st.subheader("LSTM Model Architecture")
    st.text(model.summary())
    
    # 3. Train the model
    with st.spinner("Training LSTM model... this may take a moment."):
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1, # Use 10% of the training data for validation
            verbose=0 # Suppress verbose output in Streamlit
        )
    st.success("🎉 LSTM Model Training Complete!")
    return model, history


# ==========================================
# 4. STREAMLIT APP LAYOUT & LOGIC
# ==========================================

# --- Load Data and Scaler ---
df_scaled, scaler = load_and_scale_data(DATA_URL)

if df_scaled is None or df_scaled.empty:
    st.warning("Could not load and process data. Check the data source.")
else:
    # --- Sidebar Configuration ---
    st.sidebar.header("Configuration")
    
    # Sequence Length Input
    max_seq_len = len(df_scaled) - 1
    seq_len = st.sidebar.slider(
        "Sequence Length (Timesteps)",
        min_value=1, max_value=min(max_seq_len, 30), value=10, step=1
    )
    
    # Target Column Selection
    target_column = st.sidebar.selectbox(
        "Target Column (y)",
        options=df_scaled.columns,
        index=df_scaled.columns.get_loc('Close') if 'Close' in df_scaled.columns else 0
    )
    
    # Model Hyperparameters
    st.sidebar.subheader("Model Parameters")
    train_ratio = st.sidebar.slider("Train Split (%)", 50, 90, 80, 5) / 100
    epochs = st.sidebar.slider("Training Epochs", 5, 100, 25, 5)
    
    # --- Prepare Sequence Data (X, y) ---
    feature_cols = [col for col in df_scaled.columns if col != target_column]
    df_reordered = df_scaled[feature_cols + [target_column]]
    data_array = df_reordered.values
    
    X, y = prepare_sequence_data(data_array, seq_len)
    
    if X.size > 0:
        # --- Split Data into Train/Test ---
        # NOTE: For time series, shuffle MUST be False to preserve chronological order
        split_index = int(train_ratio * len(X))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        st.subheader("Data Split Summary")
        col_x, col_y = st.columns(2)
        col_x.metric("Training Samples (X_train)", X_train.shape[0])
        col_y.metric("Testing Samples (X_test)", X_test.shape[0])
        st.caption(f"Data split chronologically ({int(train_ratio*100)}% Train / {int((1-train_ratio)*100)}% Test)")

        # --- Train Model ---
        model, history = build_and_train_model(X_train, y_train, epochs=epochs)
        
        # --- Plot Training History ---
        st.subheader("Training Loss History")
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(history.history['loss'], label='Train Loss')
        ax_loss.plot(history.history['val_loss'], label='Validation Loss')
        ax_loss.set_title("Model Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Mean Squared Error (MSE)")
        ax_loss.legend()
        st.pyplot(fig_loss)
        
        # --- Evaluation and Prediction Placeholder ---
        st.header("5. Model Evaluation")
        
        # NOTE: The actual prediction and inverse transform logic is more complex 
        # for a multi-variate model and would require an additional function.
        st.info("The model is trained! The next steps would involve predicting on the test set, inverse transforming the predictions, and calculating metrics like RMSE.")
        
    else:
        st.error("Not enough data to create sequences with the current sequence length.")
