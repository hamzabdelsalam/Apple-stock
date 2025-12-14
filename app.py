import streamlit as st
import pandas as pd
import numpy as np

# --- 1. The Core Data Preprocessing Function ---

# This function is the equivalent of your sequence_pipeline_stage
def prepare_sequence_data(data_array: np.ndarray, seq_len: int):
    """
    Transforms a 2D numpy array into sequence data (X) and a target (y).
    X will contain data from the past 'seq_len' steps (excluding the last column).
    y will contain the current step's target value (the last column).

    Args:
        data_array: A 2D numpy array (e.g., [n_samples, n_features]).
        seq_len: The look-back sequence length.

    Returns:
        A tuple (X, y) of numpy arrays.
        X shape: (n_samples - seq_len, seq_len, n_features - 1)
        y shape: (n_samples - seq_len,)
    """
    X, y = [], []
    n_samples = len(data_array)
    
    # Start loop from the first index that can form a complete sequence
    for i in range(seq_len, n_samples):
        # X: data from i - seq_len up to i (exclusive), for all columns except the last one
        X.append(data_array[i-seq_len:i, :-1]) 
        # y: the target value at index i (the last column)
        y.append(data_array[i, -1]) 
    
    return np.array(X), np.array(y)


# --- 2. Streamlit GUI Layout ---

st.set_page_config(
    page_title="Sequence Data Preparation Tool",
    layout="wide"
)

st.title("🔗 Sequence Data Preparation Tool")
st.markdown("Use this tool to transform time-series data into the $X$ (features) and $y$ (target) format required for sequence models (e.g., LSTMs, RNNs).")


# --- Sidebar for User Input ---
st.sidebar.header("Configuration")

# 1. Sequence Length Input
seq_len = st.sidebar.slider(
    "Select Sequence Length (Look-back Steps)",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
    help="The number of historical time steps to include in each feature sample (X)."
)

# 2. File Uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV File",
    type=["csv"],
    help="The last column of your CSV will be treated as the target (y)."
)


if uploaded_file is not None:
    try:
        # Read the file using pandas
        df = pd.read_csv(uploaded_file)
        st.subheader("1. Raw Data Preview")
        st.dataframe(df.head())
        
        # Convert DataFrame to numpy array for processing
        data_array = df.values
        
        # Check if the data is large enough for the chosen sequence length
        if len(data_array) < seq_len + 1:
            st.error(f"Data is too short. Requires at least {seq_len + 1} rows to create one sample with seq_len={seq_len}. Current rows: {len(data_array)}")
        else:
            # --- 3. Process Data ---
            X, y = prepare_sequence_data(data_array, seq_len)
            
            st.success(f"✅ Data successfully transformed with a sequence length of **{seq_len}**!")
            
            st.subheader("2. Transformed Data Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("**Feature Data (X)**")
                st.write(f"**Shape:** `{X.shape}`")
                st.markdown(
                    """
                    This 3D array is structured as:
                    * **Dimension 1:** Number of Samples ($\text{N}$)
                    * **Dimension 2:** Sequence Length ($\text{seq\_len}$)
                    * **Dimension 3:** Number of Input Features ($\text{N}_{\text{features}} - 1$)
                    """
                )
            
            with col2:
                st.info("**Target Data (y)**")
                st.write(f"**Shape:** `{y.shape}`")
                st.markdown(
                    """
                    This 1D array contains the target value corresponding to the prediction *after* each sequence.
                    """
                )

            st.subheader("3. Preview of the First Sample ($X[0]$ and $y[0]$)")
            
            st.code(f"X[0] (The historical sequence):", language='text')
            st.dataframe(pd.DataFrame(X[0], columns=df.columns[:-1]))
            
            st.code(f"y[0] (The target value):", language='text')
            st.write(y[0])
            
            st.markdown(
                f"**Interpretation:** The model will learn to predict the value **{y[0]}** using the historical sequence data shown above."
            )
            
            # Button to display the full datasets (optional, for larger data might be slow)
            if st.checkbox("Show Full X and y Arrays (Use with Caution for Large Datasets)"):
                st.subheader("Full X Array")
                # Showing X is tricky in 3D, let's show the flattened 2D for simplicity
                st.dataframe(pd.DataFrame(X.reshape(X.shape[0], -1)))
                
                st.subheader("Full y Array")
                st.dataframe(pd.DataFrame(y, columns=[df.columns[-1] + ' (Target)']))
                
    except Exception as e:
        st.error(f"An error occurred during file processing: {e}")

else:
    st.info("""
        **How to use:**
        1.  Adjust the **Sequence Length** in the sidebar.
        2.  **Upload a CSV file** (make sure the column you want to predict is the last one).
        3.  The tool will process your data and show the shapes and a sample of the resulting $X$ and $y$ arrays.
    """)
