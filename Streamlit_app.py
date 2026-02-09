import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Mall Customer Clustering",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .stMetric {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #667eea;
        }
        h1 {
            color: #ffffff;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            margin-bottom: 10px;
        }
        h2 {
            color: #ffffff;
            margin-top: 30px;
        }
        .prediction-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }
        .cluster-info {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 5px solid #667eea;
        }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    """Load the trained Decision Tree model"""
    try:
        model = joblib.load("knn_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please train the model first.")
        return None

# Load the dataset to get statistics
@st.cache_data
def load_data():
    """Load the original customer data"""
    try:
        df = pd.read_csv("Mall_Customers.csv")
        return df
    except FileNotFoundError:
        st.error("Data file not found.")
        return None

# Cluster information with descriptions
CLUSTER_INFO = {
    0: {
        "name": "High Value Customers",
        "description": "Young customers with high spending score and good income",
        "color": "#FF6B6B",
        "characteristics": ["Age: Young (25-40)", "Income: High (40-80k)", "Spending Score: High (70-100)"]
    },
    1: {
        "name": "Potential Target",
        "description": "Middle-aged customers with moderate to high spending",
        "color": "#4ECDC4",
        "characteristics": ["Age: Middle-aged (35-50)", "Income: Moderate (30-70k)", "Spending Score: Moderate to High (50-100)"]
    },
    2: {
        "name": "Average Customers",
        "description": "Young customers with low to moderate spending",
        "color": "#45B7D1",
        "characteristics": ["Age: Young (20-50)", "Income: Low (20-50k)", "Spending Score: Low to Moderate (20-60)"]
    },
    3: {
        "name": "Loyal Customers",
        "description": "Older customers with variable spending patterns",
        "color": "#FFA07A",
        "characteristics": ["Age: Older (40-70)", "Income: Variable", "Spending Score: Variable"]
    },
    4: {
        "name": "Budget Conscious",
        "description": "Customers with high income but low spending",
        "color": "#98D8C8",
        "characteristics": ["Age: Varied", "Income: High (50-150k)", "Spending Score: Low (10-50)"]
    }
}

# Title and Description
st.markdown("""
    <h1>🛍️ Mall Customer Clustering Prediction</h1>
    <p style='color: white; font-size: 16px; text-align: center;'>
    Predict customer segments and discover which group your customer belongs to!
    </p>
""", unsafe_allow_html=True)

# Load model and data
model = load_model()
df = load_data()

if model is not None and df is not None:
    # Create two columns for layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📊 Customer Information")
        st.markdown("---")
        
        # Input fields with sliders
        age = st.slider(
            "👤 Age",
            min_value=int(df['Age'].min()),
            max_value=int(df['Age'].max()),
            value=30,
            step=1
        )
        
        annual_income = st.slider(
            "💰 Annual Income (k$)",
            min_value=int(df['Annual Income (k$)'].min()),
            max_value=int(df['Annual Income (k$)'].max()),
            value=50,
            step=1
        )
        
        spending_score = st.slider(
            "🎯 Spending Score (1-100)",
            min_value=1,
            max_value=100,
            value=50,
            step=1
        )
        
        st.markdown("---")
        
        # Display input values
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Age", f"{age} years")
        with col_b:
            st.metric("Income", f"${annual_income}k")
        with col_c:
            st.metric("Spending", f"{spending_score}/100")
    
    with col2:
        st.markdown("### 📈 Dataset Statistics")
        st.markdown("---")
        
        # Show statistics
        col_x, col_y, col_z = st.columns(3)
        with col_x:
            st.metric("Total Customers", f"{len(df)}")
        with col_y:
            st.metric("Avg Age", f"{df['Age'].mean():.1f} years")
        with col_z:
            st.metric("Avg Income", f"${df['Annual Income (k$)'].mean():.1f}k")
        
        st.markdown("---")
        
        col_p, col_q, col_r = st.columns(3)
        with col_p:
            st.metric("Min Income", f"${df['Annual Income (k$)'].min():.0f}k")
        with col_q:
            st.metric("Max Income", f"${df['Annual Income (k$)'].max():.0f}k")
        with col_r:
            st.metric("Avg Spending", f"{df['Spending Score (1-100)'].mean():.1f}")
    
    st.markdown("---")
    
    # Prediction section
    if st.button("🚀 Predict Cluster", use_container_width=True, key="predict_btn"):
        # Create a DataFrame with the input values (need to scale them)
        input_data = pd.DataFrame({
            'Age': [age],
            'Annual Income (k$)': [annual_income],
            'Spending Score (1-100)': [spending_score]
        })
        
        # Fit scaler on original data to ensure consistency
        scaler = StandardScaler()
        scaler.fit(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
        
        # Scale the input data
        scaled_input = scaler.transform(input_data)
        
        # Make prediction
        cluster = model.predict(scaled_input)[0]
        
        # Display prediction result
        st.markdown("<br>", unsafe_allow_html=True)
        
        cluster_details = CLUSTER_INFO[cluster]
        
        st.markdown(f"""
            <div class="prediction-box">
                <h2 style='margin: 0; font-size: 28px;'>Cluster Prediction</h2>
                <h1 style='margin: 10px 0; font-size: 48px; text-shadow: none;'>{cluster}</h1>
                <h3 style='margin: 10px 0; font-size: 24px; text-shadow: none;'>{cluster_details['name']}</h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display cluster information
        col1_info, col2_info = st.columns(2, gap="large")
        
        with col1_info:
            st.markdown("### 📋 Cluster Description")
            st.markdown(f"""
                <div class="cluster-info">
                    <p><strong>{cluster_details['description']}</strong></p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🎯 Key Characteristics")
            for char in cluster_details['characteristics']:
                st.markdown(f"• {char}")
        
        with col2_info:
            st.markdown("### 💡 Cluster Insights")
            
            # Calculate cluster statistics from original data
            clustered_data = pd.read_csv("Mall_Customers_Clustered.csv")
            cluster_stats = clustered_data[clustered_data['Cluster'] == cluster]
            
            st.markdown(f"""
                <div class="cluster-info">
                    <p><strong>Size:</strong> {len(cluster_stats)} customers ({len(cluster_stats)/len(clustered_data)*100:.1f}%)</p>
                    <p><strong>Avg Age:</strong> {df[df.index.isin(cluster_stats.index)]['Age'].mean():.1f} years</p>
                    <p><strong>Avg Income:</strong> ${df[df.index.isin(cluster_stats.index)]['Annual Income (k$)'].mean():.1f}k</p>
                    <p><strong>Avg Spending:</strong> {df[df.index.isin(cluster_stats.index)]['Spending Score (1-100)'].mean():.1f}/100</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("### 📊 Visualizations")
        
        viz_col1, viz_col2 = st.columns(2, gap="large")
        
        with viz_col1:
            # 3D scatter plot showing the prediction
            clustered_df = pd.read_csv("Mall_Customers_Clustered.csv")
            
            # Inverse transform to original scale for visualization
            original_scaled = clustered_df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].copy()
            original_unscaled = scaler.inverse_transform(original_scaled)
            
            fig = px.scatter_3d(
                x=original_unscaled[:, 0],
                y=original_unscaled[:, 1],
                z=original_unscaled[:, 2],
                color=clustered_df['Cluster'].astype(str),
                labels={'x': 'Age', 'y': 'Annual Income (k$)', 'z': 'Spending Score'},
                title='3D Cluster Distribution',
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            
            # Add the prediction point
            fig.add_scatter3d(
                x=[age], y=[annual_income], z=[spending_score],
                mode='markers',
                marker=dict(size=15, color='red', symbol='diamond'),
                name='Your Input',
                showlegend=True
            )
            
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_col2:
            # Cluster distribution pie chart
            cluster_counts = clustered_df['Cluster'].value_counts().sort_index()
            colors_list = [CLUSTER_INFO[i]['color'] for i in cluster_counts.index]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=[f"Cluster {i}: {CLUSTER_INFO[i]['name']}" for i in cluster_counts.index],
                values=cluster_counts.values,
                marker=dict(colors=colors_list),
                text=cluster_counts.values,
                textposition='inside'
            )])
            
            fig_pie.update_layout(
                title='Customer Distribution by Cluster',
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Additional insights
        st.markdown("### 💼 Business Recommendations")
        
        recommendations = {
            0: "🎯 Premium targeting strategy - Focus on retention and upselling premium products",
            1: "📈 Growth opportunity - Target with seasonal promotions and loyalty programs",
            2: "🎁 Budget offerings - Create value packs and discounts to increase engagement",
            3: "🤝 Relationship building - Personalized communication and special offers",
            4: "💎 Exclusive products - Premium/investment products despite lower spending"
        }
        
        st.info(f"**Recommendation:** {recommendations[cluster]}")

else:
    st.error("Failed to load model or data. Please ensure the files are present.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white; padding: 20px;'>
    <p>🛍️ Mall Customer Clustering Analysis | Powered by K-Means & Decision Tree ML Models</p>
    </div>
""", unsafe_allow_html=True)