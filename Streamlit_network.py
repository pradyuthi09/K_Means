import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
 
# Set page config
st.set_page_config(page_title="Traffic Anomaly Detection", layout="wide", initial_sidebar_state="expanded")
 
# Title and description
st.title("🚗 Traffic Flow Anomaly Detection")
st.markdown("Detect anomalies in network traffic using Isolation Forest algorithm")
 
# Sidebar for parameters
st.sidebar.header("⚙️ Configuration")
contamination = st.sidebar.slider("Contamination Rate", 0.01, 0.2, 0.04, 0.01)
n_estimators = st.sidebar.slider("Number of Trees", 50, 200, 100, 10)
 
# Load data
st.sidebar.header("📁 Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
 
# Create sample dataset option
use_sample = st.sidebar.checkbox("Use sample dataset", value=True)
 
if use_sample and uploaded_file is None:
    # Load the default dataset
    try:
        df = pd.read_csv('embedded_system_network_security_dataset.csv')
        st.sidebar.success("✓ Sample dataset loaded")
    except FileNotFoundError:
        st.error("Dataset file not found. Please upload a CSV file.")
        st.stop()
elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✓ File uploaded successfully")
else:
    st.error("Please upload a CSV file or use sample dataset")
    st.stop()
 
# Data preprocessing
st.sidebar.header("⚙️ Data Processing")
with st.sidebar.expander("Processing Steps"):
    st.write("1. Drop label column (if exists)")
    st.write("2. Convert bool to int")
    st.write("3. Scale features")
    st.write("4. Train Isolation Forest")
    st.write("5. Predict anomalies")
 
# Process data
features = df.drop(columns=['label'], errors='ignore')
 
# Convert bool to int
for col in features.columns:
    if features[col].dtype == 'bool':
        features[col] = features[col].astype(int)
 
# Handle missing values
features = features.fillna(features.mean())
 
# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
 
# Train model
model = IsolationForest(
    n_estimators=n_estimators,
    contamination=contamination,
    max_samples=256,
    random_state=42
)
model.fit(scaled_df)
anomaly_labels = model.predict(scaled_df)
scaled_df['anomaly'] = anomaly_labels
 
# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Visualizations", "📋 Details", "📥 Export"])
 
# Tab 1: Overview
with tab1:
    col1, col2, col3, col4 = st.columns(4)
   
    normal_count = len(scaled_df[scaled_df['anomaly'] == 1])
    anomaly_count = len(scaled_df[scaled_df['anomaly'] == -1])
    total_count = len(scaled_df)
   
    with col1:
        st.metric("Total Records", total_count, delta=None)
    with col2:
        st.metric("Normal", normal_count, f"{normal_count/total_count*100:.1f}%")
    with col3:
        st.metric("Anomalies", anomaly_count, f"{anomaly_count/total_count*100:.1f}%")
    with col4:
        st.metric("Anomaly Rate", f"{contamination*100:.1f}%", delta=None)
   
    st.divider()
   
    # Dataset preview
    st.subheader("Dataset Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Original Data**")
        st.dataframe(df.head(5), use_container_width=True)
    with col2:
        st.write("**With Anomaly Labels**")
        st.dataframe(scaled_df.head(5), use_container_width=True)
 
# Tab 2: Visualizations
with tab2:
    st.subheader("Anomaly Detection Visualizations")
   
    normal = scaled_df[scaled_df['anomaly'] == 1]
    anomaly = scaled_df[scaled_df['anomaly'] == -1]
   
    # Get numeric columns
    numeric_cols = scaled_df.columns.drop('anomaly').tolist()
   
    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
       
        with col1:
            st.write("**Select Features for 2D Plot**")
            feat1 = st.selectbox("X-axis:", numeric_cols, key="feat1")
            feat2 = st.selectbox("Y-axis:", numeric_cols, key="feat2", index=1 if len(numeric_cols) > 1 else 0)
       
        with col2:
            st.write("")
            st.write("")
            plot_type = st.radio("Plot Type:", ["Scatter", "Density"], horizontal=True)
       
        # Create matplotlib plot
        fig, ax = plt.subplots(figsize=(10, 6))
       
        if plot_type == "Scatter":
            ax.scatter(normal[feat1], normal[feat2], c='blue', label='Normal', alpha=0.6, s=20)
            ax.scatter(anomaly[feat1], anomaly[feat2], c='red', label='Anomaly', alpha=0.8, s=100, marker='x', linewidths=2)
        else:
            ax.hist2d(normal[feat1], normal[feat2], bins=30, alpha=0.7, label='Normal', cmap='Blues')
            ax.scatter(anomaly[feat1], anomaly[feat2], c='red', label='Anomaly', s=100, marker='x', linewidths=2)
       
        ax.set_xlabel(feat1, fontsize=11, fontweight='bold')
        ax.set_ylabel(feat2, fontsize=11, fontweight='bold')
        ax.set_title(f'{feat1} vs {feat2}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
   
    # 3D Interactive Plot
    st.write("")
    st.subheader("3D Interactive Visualization")
   
    if len(numeric_cols) >= 3:
        col1, col2, col3 = st.columns(3)
        with col1:
            feat_x = st.selectbox("X-axis:", numeric_cols, key="3d_feat1")
        with col2:
            feat_y = st.selectbox("Y-axis:", numeric_cols, key="3d_feat2", index=1 if len(numeric_cols) > 1 else 0)
        with col3:
            feat_z = st.selectbox("Z-axis:", numeric_cols, key="3d_feat3", index=2 if len(numeric_cols) > 2 else 0)
       
        fig = go.Figure()
       
        fig.add_trace(go.Scatter3d(
            x=normal[feat_x], y=normal[feat_y], z=normal[feat_z],
            mode='markers', name='Normal',
            marker=dict(size=4, color='blue', opacity=0.6)
        ))
       
        fig.add_trace(go.Scatter3d(
            x=anomaly[feat_x], y=anomaly[feat_y], z=anomaly[feat_z],
            mode='markers', name='Anomaly',
            marker=dict(size=8, color='red', opacity=0.9, symbol='diamond')
        ))
       
        fig.update_layout(
            title="3D Plot: Normal vs Anomaly",
            scene=dict(
                xaxis_title=feat_x,
                yaxis_title=feat_y,
                zaxis_title=feat_z
            ),
            height=600,
            hovermode='closest'
        )
       
        st.plotly_chart(fig, use_container_width=True)
 
# Tab 3: Details
with tab3:
    st.subheader("Anomaly Statistics")
   
    col1, col2 = st.columns(2)
   
    with col1:
        st.write("**Normal Data Statistics**")
        st.dataframe(normal.describe().drop('anomaly', axis=1), use_container_width=True)
   
    with col2:
        st.write("**Anomaly Data Statistics**")
        st.dataframe(anomaly.describe().drop('anomaly', axis=1), use_container_width=True)
   
    st.divider()
   
    # Feature importance based on variance difference
    st.subheader("Feature Analysis")
   
    variance_diff = []
    for col in numeric_cols:
        normal_var = normal[col].var()
        anomaly_var = anomaly[col].var()
        diff = abs(anomaly_var - normal_var)
        variance_diff.append({'Feature': col, 'Variance_Diff': diff})
   
    var_df = pd.DataFrame(variance_diff).sort_values('Variance_Diff', ascending=False)
   
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(var_df['Feature'], var_df['Variance_Diff'], color='steelblue')
    ax.set_xlabel('Variance Difference', fontweight='bold')
    ax.set_title('Feature Variance Difference (Anomaly vs Normal)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    st.pyplot(fig, use_container_width=True)
 
# Tab 4: Export
with tab4:
    st.subheader("Export Results")
   
    # Add predictions to original dataframe
    results_df = df.copy()
    results_df['anomaly_prediction'] = anomaly_labels
    results_df['anomaly_type'] = results_df['anomaly_prediction'].map({1: 'Normal', -1: 'Anomaly'})
   
    col1, col2 = st.columns(2)
   
    with col1:
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Results (CSV)",
            data=csv,
            file_name="anomaly_detection_results.csv",
            mime="text/csv"
        )
   
    with col2:
        anomaly_only = results_df[results_df['anomaly_type'] == 'Anomaly']
        csv_anomaly = anomaly_only.to_csv(index=False)
        st.download_button(
            label="📥 Download Anomalies Only (CSV)",
            data=csv_anomaly,
            file_name="detected_anomalies.csv",
            mime="text/csv"
        )
   
    st.divider()
    st.subheader("Summary Report")
   
    summary = f"""
    ## Traffic Anomaly Detection Report
   
    ### Dataset Summary
    - **Total Records**: {total_count}
    - **Normal Records**: {normal_count} ({normal_count/total_count*100:.2f}%)
    - **Anomalous Records**: {anomaly_count} ({anomaly_count/total_count*100:.2f}%)
   
    ### Model Configuration
    - **Algorithm**: Isolation Forest
    - **Number of Trees**: {n_estimators}
    - **Contamination Rate**: {contamination}
    - **Sample Size**: 256
    - **Features Used**: {len(numeric_cols)}
   
    ### Detected Anomalies
    - **Total Anomalies**: {anomaly_count}
    - **Detection Rate**: {anomaly_count/total_count*100:.2f}%
   
    ---
    *Report Generated on 2026-02-16*
    """
   
    st.markdown(summary)
   
    # Download report
    report_text = summary.replace("## ", "").replace("### ", "").replace("**", "").replace("- ", "")
    st.download_button(
        label="📥 Download Report (TXT)",
        data=report_text,
        file_name="anomaly_detection_report.txt",
        mime="text/plain"
    )
 
# Footer
st.divider()
st.caption("🔬 Traffic Flow Anomaly Detection System | Powered by Streamlit")
 
 