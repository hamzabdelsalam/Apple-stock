import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import joblib # For saving the scaler (optional but good practice)

# --- Define the URL for the raw CSV data (Using a public source as a workaround for KaggleHub) ---
# NOTE: Replace this URL with your own CSV file hosted publicly if needed.
DATA_URL = "https://raw.githubusercontent.com/datasets/finance-data/main/apple_stock.csv"

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Apple Stock Data Analyzer & Preprocessor", 
    page_icon="🍎",
    layout="wide"
)

# --- The Core Sequence Data Preprocessing Function ---
def prepare_sequence_data(data_array: np.ndarray, seq_len: int):
    """
    Transforms a 2D numpy array into sequence data (X) and a target (y).
    X shape: (N_samples, seq_len, N_features - 1)
    y shape: (N_samples,)
    """
    X, y = [], []
    n_samples = len(data_array)
    
    if data_array.shape[1] < 2:
        st.error("Data must have at least two columns: one feature and one target.")
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
# 2. DATA LOADING, SCALING, & PREPROCESSING
# ==========================================
@st.cache_data
def load_and_scale_data(data_url):
    """
    Loads data directly from a URL, cleans it, and applies MinMaxScaler.
    """
    st.info(f"Attempting to load data from public URL: **{data_url}**")
    try:
        df = pd.read_csv(data_url)
        
        # Basic cleaning and date parsing for stock data
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
        # Fit the scaler to all numeric data
        scaled_data = scaler.fit_transform(df_numeric)
        
        # Convert back to DataFrame for easy column management
        df_scaled = pd.DataFrame(scaled_data, columns=df_numeric.columns, index=df_numeric.index)
        
        st.success("🍎 Data loaded, cleaned, and scaled successfully!")
        return df_scaled, scaler
    
    except Exception as e:
        st.error(f"Error loading data from URL or processing file: {e}")
        return None, None

# ==========================================
# 3. STREAMLIT APP LAYOUT & LOGIC
# ==========================================

# --- Load Data and Scaler ---
df_scaled, scaler = load_and_scale_data(DATA_URL)

if df_scaled is None or df_scaled.empty:
    st.warning("Could not load and process data. Check the data source or your connection.")
else:
    # --- Sidebar Configuration ---
    st.sidebar.header("Configuration")
    
    # 1. Sequence Length Input
    max_seq_len = len(df_scaled) - 1
    seq_len = st.sidebar.slider(
        "Select Sequence Length (Look-back Steps)",
        min_value=1,
        max_value=min(max_seq_len, 30),
        value=10,
        step=1,
        help="The number of historical time steps to include in each feature sample (X)."
    )
    
    # 2. Target Column Selection
    target_column = st.sidebar.selectbox(
        "Select Target Column (y)",
        options=df_scaled.columns,
        index=df_scaled.columns.get_loc('Close') if 'Close' in df_scaled.columns else 0,
        help="This column will be used as the prediction target (y)."
    )
    
    # --- Main Content ---
    
    st.header("📊 Scaled Stock Data Overview")
    st.write(f"Scaled Dataset Shape: **{df_scaled.shape}**")
    st.dataframe(df_scaled.tail())
    
    # Plotting the target column
    st.subheader(f"Time Series Plot: Scaled {target_column} Price (0 to 1)")
    fig, ax = plt.subplots(figsize=(10, 4))
    df_scaled[target_column].plot(ax=ax, title=f'Scaled {target_column} Price Over Time')
    st.pyplot(fig)
    
    
    st.header("⚙️ Sequence Data Preparation")
    
    # 1. Prepare the input data array: Move target column to the end
    feature_cols = [col for col in df_scaled.columns if col != target_column]
    df_reordered = df_scaled[feature_cols + [target_column]]
    data_array = df_reordered.values
    
    # 2. Process Data
    if len(data_array) < seq_len + 1:
        st.error(f"Data is too short. Requires at least {seq_len + 1} rows to create one sample with seq_len={seq_len}. Current rows: {len(data_array)}")
    else:
        X, y = prepare_sequence_data(data_array, seq_len)
        
        st.success(f"✅ Data successfully scaled and transformed with seq_len **{seq_len}**!")
        
        st.subheader("Results Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Feature Data (X)**")
            st.write(f"**Shape:** `{X.shape}`")
            st.markdown(
                f"""
                * $X$ is the input sequence.
                * **Shape:** $(\\text{{N}}_{{\\text{{samples}}}}, {seq_len}, \\text{{N}}_{{\\text{{features}}}})$
                """
            )
        
        with col2:
            st.info("**Target Data (y)**")
            st.write(f"**Shape:** `{y.shape}`")
            st.markdown(
                """
                * $y$ is the predicted next value (scaled).
                * **Shape:** $(\\text{{N}}_{{\\text{{samples}}}},)$
                """
            )

        st.subheader("Example of First Sample")
        
        # Display the first sample sequence (X[0])
        st.code(f"X[0] (Historical sequence of scaled features):", language='text')
        st.dataframe(
            pd.DataFrame(X[0], 
                         columns=feature_cols,
                         index=[f't-{seq_len-i}' for i in range(seq_len)])
        )
        
        # Display the target (y[0])
        st.code(f"y[0] (The scaled target value):", language='text')
        st.write(f"**{y[0]:.6f}** (Scaled {target_column})")
        
        st.caption("The sequence data is now ready for training a Recurrent Neural Network (RNN) or LSTM.")
