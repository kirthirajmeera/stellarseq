#!/usr/bin/env python3
"""
StellarSeq - Astronaut Genetic Stress Response Predictor
Streamlit Web Application

A public-facing dashboard for predicting stress severity scores
from gene expression data using NASA OSDR spaceflight genomics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="StellarSeq - Astronaut Stress Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stress-low { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .stress-moderate { background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%); }
    .stress-high { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
    .stress-severe { background: linear-gradient(135deg, #8E2DE2 0%, #4A00E0 100%); }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model and scaler."""
    model_path = Path(".")
    
    try:
        model = joblib.load(model_path / "best_model.pkl")
        scaler = joblib.load(model_path / "scaler.pkl")
        
        # Load metadata if available
        metadata_path = model_path / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        return model, scaler, metadata
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, {}


def get_stress_category(score):
    """Categorize stress score."""
    if score < 25:
        return "LOW", "stress-low", "🟢"
    elif score < 50:
        return "MODERATE", "stress-moderate", "🟡"
    elif score < 75:
        return "HIGH", "stress-high", "🟠"
    else:
        return "SEVERE", "stress-severe", "🔴"


def create_gauge_chart(score):
    """Create a gauge chart for stress score visualization."""
    category, _, _ = get_stress_category(score)
    
    # Color based on score
    if score < 25:
        bar_color = "#38ef7d"
    elif score < 50:
        bar_color = "#F2C94C"
    elif score < 75:
        bar_color = "#f45c43"
    else:
        bar_color = "#4A00E0"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Stress Severity Score", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': bar_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': 'rgba(56, 239, 125, 0.3)'},
                {'range': [25, 50], 'color': 'rgba(242, 201, 76, 0.3)'},
                {'range': [50, 75], 'color': 'rgba(244, 92, 67, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(74, 0, 224, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "darkgray", 'family': "Arial"}
    )
    
    return fig


def create_feature_importance_chart(model, feature_names):
    """Create feature importance visualization."""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            return None
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            x=sorted_importances,
            y=sorted_features,
            orientation='h',
            marker=dict(
                color=sorted_importances,
                colorscale='Viridis',
                showscale=True
            )
        ))
        
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    except Exception:
        return None


def predict_stress(model, scaler, features_df, feature_cols):
    """Make predictions on input features."""
    # Ensure columns are in correct order
    X = features_df[feature_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    return predictions


# Main app
def main():
    # Header
    st.markdown('<p class="main-header">🚀 StellarSeq</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Astronaut Genetic Stress Response Predictor</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://www.nasa.gov/wp-content/themes/flavor/flavor/assets/img/nasa-logo.svg", width=100)
        st.markdown("### About")
        st.markdown("""
        **StellarSeq** predicts astronaut stress severity 
        from gene expression data using machine learning 
        trained on NASA OSDR spaceflight genomics.
        
        ---
        
        **Model Performance:**
        - R² Score: 0.974
        - RMSE: 2.92
        - Trained on 112 samples
        
        ---
        
        **Data Sources:**
        - NASA Open Science Data Repository
        - 9 spaceflight studies
        - Mouse & Human samples
        """)
        
        st.markdown("---")
        st.markdown("### Stress Categories")
        st.markdown("🟢 **LOW** (0-25)")
        st.markdown("🟡 **MODERATE** (25-50)")
        st.markdown("🟠 **HIGH** (50-75)")
        st.markdown("🔴 **SEVERE** (75-100)")
    
    # Load model
    model, scaler, metadata = load_model()
    
    if model is None:
        st.error("⚠️ Model not loaded. Please ensure model files are in the 'model/' directory.")
        return
    
    # Feature columns (must match training)
    feature_cols = [
        'mean_expression', 'std_expression', 'median_expression', 'gene_count',
        'mean_fold_change', 'std_fold_change', 'mean_abs_fold_change',
        'n_upregulated', 'n_downregulated', 'n_significant',
        'organism_code_encoded', 'condition_encoded', 'tissue_encoded', 'study_id_encoded'
    ]
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📊 Single Prediction", "📁 Batch Upload", "ℹ️ About"])
    
    # Tab 1: Single Prediction (Manual Input)
    with tab1:
        st.markdown("### Manual Feature Input")
        st.markdown("Enter sample characteristics to predict stress severity.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Expression Statistics")
            mean_expr = st.number_input("Mean Expression (log2)", value=8.5, min_value=0.0, max_value=20.0, step=0.1)
            std_expr = st.number_input("Std Expression", value=2.0, min_value=0.0, max_value=10.0, step=0.1)
            median_expr = st.number_input("Median Expression (log2)", value=8.0, min_value=0.0, max_value=20.0, step=0.1)
            gene_count = st.number_input("Gene Count", value=15000, min_value=1000, max_value=60000, step=1000)
            
            st.markdown("#### Fold Change Metrics")
            mean_fc = st.number_input("Mean Fold Change", value=0.5, min_value=-5.0, max_value=5.0, step=0.1)
            std_fc = st.number_input("Std Fold Change", value=1.0, min_value=0.0, max_value=5.0, step=0.1)
            mean_abs_fc = st.number_input("Mean Absolute Fold Change", value=0.8, min_value=0.0, max_value=5.0, step=0.1)
        
        with col2:
            st.markdown("#### Differential Expression")
            n_up = st.number_input("# Upregulated Genes", value=500, min_value=0, max_value=10000, step=100)
            n_down = st.number_input("# Downregulated Genes", value=500, min_value=0, max_value=10000, step=100)
            n_sig = st.number_input("# Significant Genes", value=1000, min_value=0, max_value=20000, step=100)
            
            st.markdown("#### Sample Metadata")
            organism = st.selectbox("Organism", ["Mouse", "Human"], index=0)
            organism_code = 0 if organism == "Mouse" else 1
            
            condition = st.selectbox("Condition", ["Ground Control", "Spaceflight", "Radiation", "Hindlimb Unload"])
            condition_map = {"Ground Control": 0, "Spaceflight": 1, "Radiation": 2, "Hindlimb Unload": 3}
            condition_code = condition_map[condition]
            
            tissue = st.selectbox("Tissue", ["Thymus", "Liver", "Muscle", "Blood", "Retina", "Skin", "Other"])
            tissue_map = {"Thymus": 0, "Liver": 1, "Muscle": 2, "Blood": 3, "Retina": 4, "Skin": 5, "Other": 6}
            tissue_code = tissue_map[tissue]
            
            study_id = st.slider("Study ID (encoded)", 0, 10, 5)
        
        if st.button("🔮 Predict Stress Score", type="primary", use_container_width=True):
            # Create feature vector
            features = pd.DataFrame([[
                mean_expr, std_expr, median_expr, gene_count,
                mean_fc, std_fc, mean_abs_fc,
                n_up, n_down, n_sig,
                organism_code, condition_code, tissue_code, study_id
            ]], columns=feature_cols)
            
            # Predict
            prediction = predict_stress(model, scaler, features, feature_cols)[0]
            prediction = np.clip(prediction, 0, 100)  # Clip to valid range
            
            category, css_class, emoji = get_stress_category(prediction)
            
            # Display results
            st.markdown("---")
            st.markdown("### Prediction Results")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.plotly_chart(create_gauge_chart(prediction), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Stress Score", f"{prediction:.1f}")
            with col2:
                st.metric("Category", f"{emoji} {category}")
            with col3:
                st.metric("Condition", condition)
            
            # Interpretation
            st.markdown("---")
            st.markdown("### Interpretation")
            
            if category == "LOW":
                st.success("""
                **Low Stress Response** - Gene expression patterns indicate minimal stress-related changes.
                This is typical of well-adapted samples or short-duration exposures.
                """)
            elif category == "MODERATE":
                st.warning("""
                **Moderate Stress Response** - Gene expression shows measurable stress signatures.
                The sample exhibits adaptive responses that are within manageable ranges.
                """)
            elif category == "HIGH":
                st.warning("""
                **High Stress Response** - Significant gene dysregulation detected.
                Consider monitoring for potential physiological impacts and intervention strategies.
                """)
            else:
                st.error("""
                **Severe Stress Response** - Extensive gene expression changes detected.
                Immediate attention may be required. Consult with medical and mission specialists.
                """)
    
    # Tab 2: Batch Upload
    with tab2:
        st.markdown("### Batch Prediction from CSV")
        st.markdown("Upload a CSV file with sample features to get predictions for multiple samples.")
        
        # Template download
        st.markdown("#### Download Template")
        template_df = pd.DataFrame({
            'sample_id': ['Sample_001', 'Sample_002'],
            'mean_expression': [8.5, 9.2],
            'std_expression': [2.0, 2.5],
            'median_expression': [8.0, 8.8],
            'gene_count': [15000, 18000],
            'mean_fold_change': [0.5, 1.2],
            'std_fold_change': [1.0, 1.5],
            'mean_abs_fold_change': [0.8, 1.4],
            'n_upregulated': [500, 1200],
            'n_downregulated': [500, 800],
            'n_significant': [1000, 2000],
            'organism_code_encoded': [0, 1],
            'condition_encoded': [1, 1],
            'tissue_encoded': [0, 3],
            'study_id_encoded': [5, 7]
        })
        
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Template",
            data=csv_template,
            file_name="stellarseq_template.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # File upload
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.markdown(f"**Loaded {len(df)} samples**")
                st.dataframe(df.head(), use_container_width=True)
                
                # Check for required columns
                missing_cols = [col for col in feature_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                else:
                    if st.button("🔮 Run Batch Prediction", type="primary", use_container_width=True):
                        # Make predictions
                        predictions = predict_stress(model, scaler, df, feature_cols)
                        predictions = np.clip(predictions, 0, 100)
                        
                        # Add predictions to dataframe
                        results_df = df.copy()
                        results_df['predicted_stress_score'] = predictions
                        results_df['stress_category'] = [get_stress_category(p)[0] for p in predictions]
                        
                        st.markdown("### Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean Score", f"{predictions.mean():.1f}")
                        with col2:
                            st.metric("Min Score", f"{predictions.min():.1f}")
                        with col3:
                            st.metric("Max Score", f"{predictions.max():.1f}")
                        with col4:
                            st.metric("Std Dev", f"{predictions.std():.1f}")
                        
                        # Distribution plot
                        fig = px.histogram(
                            results_df, x='predicted_stress_score',
                            nbins=20, title="Distribution of Predicted Stress Scores",
                            color_discrete_sequence=['#667eea']
                        )
                        fig.add_vline(x=25, line_dash="dash", line_color="green", annotation_text="Low/Moderate")
                        fig.add_vline(x=50, line_dash="dash", line_color="orange", annotation_text="Moderate/High")
                        fig.add_vline(x=75, line_dash="dash", line_color="red", annotation_text="High/Severe")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv_results = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv_results,
                            file_name="stellarseq_predictions.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # Tab 3: About
    with tab3:
        st.markdown("### About StellarSeq")
        
        st.markdown("""
        **StellarSeq** is an AI-powered tool for predicting astronaut genetic stress responses
        using machine learning trained on real NASA spaceflight genomics data.
        
        #### 🎯 Purpose
        
        Spaceflight induces complex biological stress responses including:
        - Microgravity-induced gene expression changes
        - Radiation exposure effects
        - Circadian rhythm disruption
        - Immune system modulation
        
        This tool helps researchers and mission planners:
        - Assess stress severity from gene expression profiles
        - Identify high-risk samples requiring intervention
        - Compare stress responses across different conditions
        
        #### 📊 Model Details
        
        | Metric | Value |
        |--------|-------|
        | Algorithm | XGBoost Regressor |
        | Training Samples | 112 |
        | Features | 14 |
        | R² Score | 0.974 |
        | RMSE | 2.92 |
        | MAE | 2.09 |
        
        #### 🔬 Data Sources
        
        Trained on data from NASA's Open Science Data Repository (OSDR):
        - 9 spaceflight and ground analog studies
        - Mouse and human samples
        - Various tissues (thymus, liver, muscle, blood, etc.)
        - Conditions: spaceflight, radiation, hindlimb unloading
        
        #### 📖 References
        
        1. [NASA OSDR](https://osdr.nasa.gov/) - Open Science Data Repository
        2. [GeneLab Analysis Working Group](https://genelab.nasa.gov/awg)
        
        #### 👨‍💻 Development
        
        Built with:
        - Python 3.11+
        - Streamlit
        - XGBoost / Scikit-learn
        - Plotly
        
        ---
        
        **Disclaimer:** This tool is for research and educational purposes only.
        Predictions should be validated by qualified researchers before making
        any medical or mission-critical decisions.
        """)
        
        # Feature importance
        st.markdown("### Feature Importance")
        importance_fig = create_feature_importance_chart(model, feature_cols)
        if importance_fig:
            st.plotly_chart(importance_fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">StellarSeq v1.0 | '
        'Powered by NASA OSDR Data | Built with Streamlit</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
