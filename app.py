import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub # Requires 'pip install kagglehub'

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Apple Stock Data Analyzer", 
    page_icon="🍎",
    layout="wide"
)

# --- The Core Sequence Data Preprocessing Function (from previous response) ---
def prepare_sequence_data(data_array: np.ndarray, seq_len: int):
    """
    Transforms a 2D numpy array into sequence data (X) and a target (y).
    """
    X, y = [], []
    n_samples = len(data_array)
    
    # Ensure there's at least one feature column (i.e., data_array.shape[1] > 1)
    if data_array.shape[1] < 2:
        st.error("Data must have at least two columns: one feature and one target.")
        return np.array([]), np.array([])

    for i in range(seq_len, n_samples):
        # X: historical data (i-seq_len to i), excluding the last (target) column
        X.append(data_array[i-seq_len:i, :-1]) 
        # y: the target value at index i (the last column)
        y.append(data_array[i, -1]) 
    
    # Handle the case where no sequences can be formed
    if not X:
        return np.array([]), np.array([])
        
    return np.array(X), np.array(y)


# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
@st.cache_data
def load_data(dataset_name="rafsunahmad/apple-stock-price", filename="apple_stock.csv"):
    """
    Downloads the dataset using kagglehub, loads the CSV file, and performs initial cleaning.
    """
    st.info(f"Attempting to load data from Kaggle dataset: **{dataset_name}**")
    try:
        # Download the dataset files (it uses a local cache for subsequent runs)
        download_path = kagglehub.model_download(dataset_name)
        file_path = os.path.join(download_path, filename)

        if not os.path.exists(file_path):
            st.error(f"File not found at: {file_path}. Check the filename.")
            return None
            
        df = pd.read_csv(file_path)
        
        # Basic cleaning and date parsing for stock data
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
        # Ensure only numeric data remains (useful for sequence preparation)
        df_numeric = df.select_dtypes(include=np.number)
        
        st.success("🍎 Data loaded and cached successfully!")
        return df_numeric
    
    except Exception as e:
        st.error(f"Error loading data from KaggleHub or processing file: {e}")
        st.warning("Ensure you have a valid Kaggle API configuration if this persists.")
        return None

# ==========================================
# 3. STREAMLIT APP LAYOUT & LOGIC
# ==========================================

# --- Load Data ---
df = load_data()

if df is None or df.empty:
    st.warning("Could not load data. Please check the dataset name and file path in the `load_data` function.")
else:
    # --- Sidebar Configuration ---
    st.sidebar.header("Configuration")
    
    # Sequence Length Input
    # Max value is capped at (Total Rows - 1) for sensible processing
    max_seq_len = len(df) - 1
    seq_len = st.sidebar.slider(
        "Select Sequence Length (Look-back Steps)",
        min_value=1,
        max_value=min(max_seq_len, 30), # Cap at 30 for visualization
        value=10,
        step=1,
        help="The number of historical time steps to include in each feature sample (X)."
    )
    
    # Target Column Selection
    target_column = st.sidebar.selectbox(
        "Select Target Column (y)",
        options=df.columns,
        index=df.columns.get_loc('Close') if 'Close' in df.columns else 0, # Default to 'Close'
        help="This column will be moved to the last position and used as the prediction target (y)."
    )
    
    # --- Main Content ---
    
    st.header("📊 Stock Data Overview")
    st.write(f"Dataset Shape: **{df.shape}**")
    st.dataframe(df.tail())
    
    # Plotting the target column
    st.subheader(f"Time Series Plot: {target_column} Price")
    fig, ax = plt.subplots(figsize=(10, 4))
    df[target_column].plot(ax=ax, title=f'{target_column} Price Over Time')
    st.pyplot(fig)
    

    st.header("⚙️ Sequence Data Preparation")
    
    # 1. Prepare the input data array: Move target column to the end
    feature_cols = [col for col in df.columns if col != target_column]
    df_reordered = df[feature_cols + [target_column]]
    data_array = df_reordered.values
    
    # 2. Process Data
    if len(data_array) < seq_len + 1:
        st.error(f"Data is too short. Requires at least {seq_len + 1} rows to create one sample with seq_len={seq_len}. Current rows: {len(data_array)}")
    else:
        X, y = prepare_sequence_data(data_array, seq_len)
        
        st.success(f"✅ Data successfully transformed with a sequence length of **{seq_len}**!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Feature Data (X)**")
            st.write(f"**Shape:** `{X.shape}`")
            st.markdown(
                """
                * $X$ is the historical sequence of features (excluding target).
                * **$X$ Shape:** $(\text{N}_{\text{samples}}, \text{seq\_len}, \text{N}_{\text{features}})$
                """
            )
        
        with col2:
            st.info("**Target Data (y)**")
            st.write(f"**Shape:** `{y.shape}`")
            st.markdown(
                """
                * $y$ is the next value to be predicted (the target).
                * **$y$ Shape:** $(\text{N}_{\text{samples}},)$
                """
            )

        st.subheader("Example of First Sample")
        
        # Display the first sample sequence (X[0])
        st.code(f"X[0] (Historical sequence of features):", language='text')
        st.dataframe(
            pd.DataFrame(X[0], 
                         columns=feature_cols,
                         index=[f't-{seq_len-i}' for i in range(seq_len)])
        )
        
        # Display the target (y[0])
        st.code(f"y[0] (The value predicted by X[0]):", language='text')
        st.write(f"**{y[0]:.4f}** (Target Column: {target_column})")
