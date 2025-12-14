import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="RNN Model Performance Comparison",
    layout="wide",
    page_icon="🏆"
)

RESULTS_FILE = "comparison_results.csv"

# ==========================================
# 2. DATA LOADING
# ==========================================
@st.cache_data
def load_comparison_data(file_path):
    """Loads the pre-calculated performance results."""
    if not os.path.exists(file_path):
        st.error(f"Missing results file: {file_path}. Please run the export script in the notebook.")
        st.stop()
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    return df

comparison_df = load_comparison_data(RESULTS_FILE)

# ==========================================
# 3. UI LAYOUT
# ==========================================

st.title("🏆 Apple Stock Prediction: Manual vs. Pipeline Performance")
st.markdown("A comparison of the Mean Absolute Error (MAE) for different Recurrent Neural Network (RNN) architectures using two distinct preprocessing approaches (Manual vs. Scikit-learn Pipeline).")

if comparison_df.empty:
    st.warning("Cannot display results as the comparison data failed to load.")
else:
    
    # --- Performance Table ---
    st.header("1. Performance Summary Table")
    st.dataframe(
        comparison_df.style.format({
            "Manual MAE": "{:.4f}", 
            "Pipeline MAE": "{:.4f}"
        }),
        use_container_width=True
    )
    
    # --- Key Findings ---
    st.markdown("---")
    st.header("2. Methodology and Visualization")
    
    col_chart, col_text = st.columns([2, 1])

    with col_chart:
        st.subheader("MAE Comparison Chart")
        
        # Recreate the comparison bar plot using Matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(comparison_df["Model"]))
        width = 0.35

        plt.bar(
            x - width/2,
            comparison_df["Manual MAE"],
            width,
            label="Manual Preprocessing",
            color='#1f77b4'
        )

        plt.bar(
            x + width/2,
            comparison_df["Pipeline MAE"],
            width,
            label="End-to-End Pipeline",
            color='#ff7f0e'
        )

        plt.xticks(x, comparison_df["Model"])
        plt.ylabel("Mean Absolute Error (MAE)")
        plt.title("Manual vs Pipeline Model Performance Comparison")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        st.pyplot(fig) # Display the Matplotlib figure 

    with col_text:
        st.subheader("Analysis & Takeaways")
        
        # Find the best performing model/method
        best_manual_mae = comparison_df["Manual MAE"].min()
        best_pipeline_mae = comparison_df["Pipeline MAE"].min()
        
        if best_pipeline_mae < best_manual_mae:
            best_model_row = comparison_df[comparison_df["Pipeline MAE"] == best_pipeline_mae].iloc[0]
            st.success(f"**Best Performing Approach:** The **End-to-End Pipeline** yielded the lowest error, with the **{best_model_row['Model']}** achieving MAE of **{best_pipeline_mae:.4f}**.")
        else:
            best_model_row = comparison_df[comparison_df["Manual MAE"] == best_manual_mae].iloc[0]
            st.success(f"**Best Performing Approach:** The **Manual Preprocessing** yielded the lowest error, with the **{best_model_row['Model']}** achieving MAE of **{best_manual_mae:.4f}**.")
            
        st.markdown(
            """
            ### Pipeline Advantage
            The End-to-End Pipeline utilizes `ColumnTransformer` and `Pipeline` to encapsulate the scaling and transformation steps . This setup ensures that the exact same preprocessing logic is applied consistently during training and testing, which is often crucial for reproducible results in production environments.
            
            ### RNN Architecture Insight
            * **SimpleRNN:** Prone to the vanishing gradient problem.
            * **LSTM/GRU:** Both use gating mechanisms to manage information flow over long sequences, leading to better performance in time-series forecasting.
            """
        )

