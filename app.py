#!/usr/bin/env python3
"""
StellarSeq - Astronaut Genetic Stress Response Predictor
Space-themed Professional Dashboard

Built by Meera Kirthiraj
Powered by NASA OSDR Data
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
    page_title="StellarSeq | Astronaut Stress Predictor",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SPACE THEME CSS - Professional Dark Mode
# ============================================================================
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Exo+2:wght@300;400;500;600;700&display=swap');
    
    /* Root variables */
    :root {
        --space-black: #0a0a0f;
        --deep-space: #0d1117;
        --nebula-purple: #1a1a2e;
        --cosmic-blue: #16213e;
        --star-white: #e6edf3;
        --nebula-pink: #ff6b9d;
        --cosmic-cyan: #00d4ff;
        --aurora-green: #00ffa3;
        --solar-orange: #ff9500;
        --warning-red: #ff4757;
        --subtle-gray: #8b949e;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0d1117 100%);
        background-attachment: fixed;
    }
    
    /* Animated stars */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, rgba(255,255,255,0.8), transparent),
            radial-gradient(2px 2px at 40px 70px, rgba(255,255,255,0.5), transparent),
            radial-gradient(1px 1px at 90px 40px, rgba(255,255,255,0.6), transparent),
            radial-gradient(2px 2px at 160px 120px, rgba(0,212,255,0.7), transparent),
            radial-gradient(1px 1px at 230px 80px, rgba(255,255,255,0.4), transparent),
            radial-gradient(2px 2px at 300px 150px, rgba(255,107,157,0.5), transparent),
            radial-gradient(1px 1px at 370px 50px, rgba(255,255,255,0.7), transparent),
            radial-gradient(2px 2px at 450px 180px, rgba(255,255,255,0.5), transparent),
            radial-gradient(1px 1px at 520px 90px, rgba(0,255,163,0.4), transparent),
            radial-gradient(2px 2px at 600px 140px, rgba(255,255,255,0.6), transparent),
            radial-gradient(1px 1px at 680px 60px, rgba(0,212,255,0.5), transparent),
            radial-gradient(2px 2px at 750px 200px, rgba(255,255,255,0.4), transparent),
            radial-gradient(1px 1px at 820px 100px, rgba(255,107,157,0.6), transparent),
            radial-gradient(2px 2px at 900px 170px, rgba(255,255,255,0.7), transparent);
        background-repeat: repeat;
        background-size: 1000px 250px;
        animation: twinkle 8s ease-in-out infinite alternate;
        z-index: 0;
    }
    
    @keyframes twinkle {
        0% { opacity: 0.4; }
        50% { opacity: 0.7; }
        100% { opacity: 0.5; }
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(13,17,23,0.95) 0%, rgba(26,26,46,0.95) 100%);
        border-right: 1px solid rgba(0,212,255,0.2);
    }
    
    [data-testid="stSidebar"] .stMarkdown { color: var(--star-white); }
    
    /* Main header */
    .main-header {
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0;
        padding-top: 1rem;
        background: linear-gradient(135deg, #00d4ff 0%, #ff6b9d 50%, #00ffa3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 4px;
        animation: glow 3s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 20px rgba(0,212,255,0.4)); }
        to { filter: drop-shadow(0 0 30px rgba(255,107,157,0.4)); }
    }
    
    .sub-header {
        font-family: 'Exo 2', sans-serif;
        font-size: 1.3rem;
        font-weight: 300;
        color: var(--subtle-gray);
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(13,17,23,0.6);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(0,212,255,0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Exo 2', sans-serif;
        font-weight: 500;
        color: var(--subtle-gray);
        background: transparent;
        border-radius: 8px;
        padding: 12px 24px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--cosmic-cyan);
        background: rgba(0,212,255,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0,212,255,0.2) 0%, rgba(255,107,157,0.2) 100%) !important;
        color: var(--star-white) !important;
        border: 1px solid rgba(0,212,255,0.4);
    }
    
    .stMarkdown, .stText { color: var(--star-white); }
    
    /* Inputs */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        background: rgba(13,17,23,0.8) !important;
        border: 1px solid rgba(0,212,255,0.3) !important;
        color: var(--star-white) !important;
        border-radius: 8px !important;
        font-family: 'Exo 2', sans-serif !important;
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'Orbitron', monospace !important;
        font-weight: 600 !important;
        letter-spacing: 2px !important;
        background: linear-gradient(135deg, #00d4ff 0%, #00ffa3 100%) !important;
        color: #0a0a0f !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 32px !important;
        text-transform: uppercase !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0,212,255,0.4) !important;
    }
    
    .stDownloadButton > button {
        font-family: 'Exo 2', sans-serif !important;
        background: rgba(0,212,255,0.2) !important;
        border: 1px solid rgba(0,212,255,0.4) !important;
        color: var(--cosmic-cyan) !important;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(13,17,23,0.9) 0%, rgba(26,26,46,0.9) 100%);
        border: 1px solid rgba(0,212,255,0.2);
        border-radius: 12px;
        padding: 16px;
    }
    
    [data-testid="stMetric"]:hover {
        border-color: rgba(0,212,255,0.5);
        box-shadow: 0 4px 20px rgba(0,212,255,0.2);
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Exo 2', sans-serif !important;
        color: var(--subtle-gray) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', monospace !important;
        color: var(--cosmic-cyan) !important;
        font-weight: 700 !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(13,17,23,0.6);
        border: 2px dashed rgba(0,212,255,0.3);
        border-radius: 12px;
        padding: 20px;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(0,212,255,0.6);
    }
    
    /* Messages */
    .stSuccess { background: rgba(0,255,163,0.1) !important; border: 1px solid rgba(0,255,163,0.3) !important; }
    .stWarning { background: rgba(255,149,0,0.1) !important; border: 1px solid rgba(255,149,0,0.3) !important; }
    .stError { background: rgba(255,71,87,0.1) !important; border: 1px solid rgba(255,71,87,0.3) !important; }
    
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,212,255,0.3), transparent);
        margin: 2rem 0;
    }
    
    .section-header {
        font-family: 'Orbitron', monospace;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--cosmic-cyan);
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(0,212,255,0.3);
        letter-spacing: 2px;
    }
    
    .stress-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-family: 'Orbitron', monospace;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 1px;
    }
    
    .stress-low { background: rgba(0,255,163,0.2); border: 1px solid rgba(0,255,163,0.5); color: #00ffa3; }
    .stress-moderate { background: rgba(255,149,0,0.2); border: 1px solid rgba(255,149,0,0.5); color: #ff9500; }
    .stress-high { background: rgba(255,107,157,0.2); border: 1px solid rgba(255,107,157,0.5); color: #ff6b9d; }
    .stress-severe { background: rgba(255,71,87,0.2); border: 1px solid rgba(255,71,87,0.5); color: #ff4757; }
    
    .footer {
        font-family: 'Exo 2', sans-serif;
        text-align: center;
        color: var(--subtle-gray);
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid rgba(0,212,255,0.2);
        font-size: 0.9rem;
        letter-spacing: 1px;
    }
    
    .sidebar-card {
        background: linear-gradient(135deg, rgba(0,212,255,0.1) 0%, rgba(255,107,157,0.05) 100%);
        border: 1px solid rgba(0,212,255,0.2);
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--deep-space); }
    ::-webkit-scrollbar-thumb { background: linear-gradient(135deg, var(--cosmic-cyan), var(--nebula-pink)); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_path = Path(".")
    try:
        model = joblib.load(model_path / "best_model.pkl")
        scaler = joblib.load(model_path / "scaler.pkl")
        metadata_path = model_path / "training_metadata.json"
        metadata = json.load(open(metadata_path)) if metadata_path.exists() else {}
        return model, scaler, metadata
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, {}


def get_stress_category(score):
    if score < 25: return "LOW", "stress-low", "🟢", "#00ffa3"
    elif score < 50: return "MODERATE", "stress-moderate", "🟡", "#ff9500"
    elif score < 75: return "HIGH", "stress-high", "🔴", "#ff6b9d"
    else: return "SEVERE", "stress-severe", "⛔", "#ff4757"


def create_gauge_chart(score):
    _, _, _, color = get_stress_category(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'font': {'size': 60, 'color': color, 'family': 'Orbitron'}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "STRESS SEVERITY INDEX", 'font': {'size': 16, 'color': '#8b949e', 'family': 'Exo 2'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "#8b949e", 'tickfont': {'color': '#8b949e', 'family': 'Orbitron'}},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(13,17,23,0.8)",
            'borderwidth': 2,
            'bordercolor': "rgba(0,212,255,0.3)",
            'steps': [
                {'range': [0, 25], 'color': 'rgba(0,255,163,0.15)'},
                {'range': [25, 50], 'color': 'rgba(255,149,0,0.15)'},
                {'range': [50, 75], 'color': 'rgba(255,107,157,0.15)'},
                {'range': [75, 100], 'color': 'rgba(255,71,87,0.15)'}
            ],
        }
    ))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=60, b=30), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def create_feature_importance_chart(model, feature_names):
    try:
        if not hasattr(model, 'feature_importances_'): return None
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig = go.Figure(go.Bar(
            x=importances[indices], y=[feature_names[i] for i in indices],
            orientation='h',
            marker=dict(color=importances[indices], colorscale=[[0, '#00ffa3'], [0.5, '#00d4ff'], [1, '#ff6b9d']])
        ))
        fig.update_layout(
            title=dict(text="FEATURE IMPORTANCE", font=dict(family='Orbitron', size=16, color='#00d4ff')),
            height=400, margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(0,212,255,0.1)', color='#8b949e'),
            yaxis=dict(categoryorder='total ascending', color='#8b949e'),
            font=dict(family='Exo 2', color='#8b949e')
        )
        return fig
    except: return None


def predict_stress(model, scaler, features_df, feature_cols):
    X = features_df[feature_cols].values.astype(np.float32)
    return model.predict(scaler.transform(X))


def parse_sample_name(sample_name):
    s = sample_name.upper()
    org = ('MOUSE', 0) if any(x in s for x in ['MMUS', 'MOUSE']) else ('HUMAN', 1) if any(x in s for x in ['HSAP', 'HUMAN']) else ('UNKNOWN', 0)
    cond = ('SPACEFLIGHT', 1) if any(x in s for x in ['_FLT', 'FLIGHT', '_SF_']) else ('GROUND_CONTROL', 0) if any(x in s for x in ['_GC_', 'GROUND', '_CTRL']) else ('RADIATION', 2) if any(x in s for x in ['HZE', 'RAD', '_IR_']) else ('HINDLIMB_UNLOAD', 3) if any(x in s for x in ['HLU', 'HINDLIMB']) else ('UNKNOWN', 0)
    tissue_map = {'LVR': ('LIVER', 1), 'TMS': ('THYMUS', 0), 'MUS': ('MUSCLE', 2), 'SKN': ('SKIN', 5), 'RTN': ('RETINA', 4), 'BLD': ('BLOOD', 3), 'PBMC': ('BLOOD', 3), 'HUVEC': ('ENDOTHELIAL', 6)}
    tis = next(((t, c) for k, (t, c) in tissue_map.items() if k in s), ('OTHER', 6))
    return {'organism': org[0], 'organism_code': org[1], 'condition': cond[0], 'condition_code': cond[1], 'tissue': tis[0], 'tissue_code': tis[1]}


def process_raw_expression_data(df, control_pattern=None):
    non_sample = ['ENSEMBL', 'SYMBOL', 'GENENAME', 'PROBEID', 'PROBESETID', 'GENE', 'GENE_SYMBOL', 'GENE_ID', 'ID', 'NAME']
    sample_cols = [c for c in df.columns if c.upper() not in [x.upper() for x in non_sample] and pd.to_numeric(df[c], errors='coerce').notna().sum() > len(df) * 0.5]
    if not sample_cols: raise ValueError("No sample columns found.")
    expr = df[sample_cols].apply(pd.to_numeric, errors='coerce')
    if expr.max().max() > 100: expr = np.log2(expr + 1)
    ctrl_cols = [c for c in sample_cols if any(x in c.upper() for x in (['_GC_', '_CTRL', '_CON', 'CONTROL', 'GROUND'] if not control_pattern else [control_pattern.upper()]))]
    ctrl_mean = expr[ctrl_cols].mean(axis=1) if ctrl_cols else expr.mean(axis=1)
    if not ctrl_cols: st.warning("⚠️ No control samples detected.")
    results = []
    for s in sample_cols:
        m = parse_sample_name(s)
        fc = expr[s] - ctrl_mean
        results.append({'sample_id': s, 'mean_expression': expr[s].mean(), 'std_expression': expr[s].std(), 'median_expression': expr[s].median(), 'gene_count': (expr[s] > 0).sum(), 'mean_fold_change': fc.mean(), 'std_fold_change': fc.std(), 'mean_abs_fold_change': fc.abs().mean(), 'n_upregulated': (fc > 1).sum(), 'n_downregulated': (fc < -1).sum(), 'n_significant': ((fc > 1) | (fc < -1)).sum(), 'organism_code_encoded': m['organism_code'], 'condition_encoded': m['condition_code'], 'tissue_encoded': m['tissue_code'], 'study_id_encoded': 0, 'organism': m['organism'], 'condition': m['condition'], 'tissue': m['tissue']})
    return pd.DataFrame(results)


def main():
    st.markdown('<h1 class="main-header">STELLARSEQ</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Astronaut Genetic Stress Response Predictor</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown('<div style="text-align:center;padding:20px 0;"><span style="font-size:4rem;">🛸</span></div>', unsafe_allow_html=True)
        st.markdown('<p class="section-header" style="font-size:1.1rem;">MISSION BRIEF</p>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-card"><p style="color:#e6edf3;font-family:Exo 2;margin:0;font-size:0.9rem;"><strong style="color:#00d4ff;">StellarSeq</strong> predicts astronaut stress from gene expression using ML trained on NASA spaceflight genomics.</p></div>', unsafe_allow_html=True)
        st.markdown('<p class="section-header" style="font-size:1rem;">MODEL STATS</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: st.metric("R² Score", "0.974"); st.metric("Samples", "112")
        with c2: st.metric("RMSE", "2.92"); st.metric("Features", "14")
        st.markdown("---")
        st.markdown('<p class="section-header" style="font-size:1rem;">STRESS INDEX</p>', unsafe_allow_html=True)
        st.markdown('<div style="font-family:Exo 2;font-size:0.85rem;line-height:2;"><span class="stress-badge stress-low">🟢 LOW 0-25</span><br><span class="stress-badge stress-moderate">🟡 MODERATE 25-50</span><br><span class="stress-badge stress-high">🔴 HIGH 50-75</span><br><span class="stress-badge stress-severe">⛔ SEVERE 75-100</span></div>', unsafe_allow_html=True)
    
    model, scaler, metadata = load_model()
    if model is None: st.error("⚠️ Model not loaded."); return
    
    feature_cols = ['mean_expression', 'std_expression', 'median_expression', 'gene_count', 'mean_fold_change', 'std_fold_change', 'mean_abs_fold_change', 'n_upregulated', 'n_downregulated', 'n_significant', 'organism_code_encoded', 'condition_encoded', 'tissue_encoded', 'study_id_encoded']
    
    tab1, tab2, tab3 = st.tabs(["ℹ️ ABOUT", "🎯 SINGLE PREDICTION", "📡 BATCH UPLOAD"])
    
    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown('<p class="section-header">MISSION OVERVIEW</p>', unsafe_allow_html=True)
            st.markdown('<div style="font-family:Exo 2;color:#e6edf3;line-height:1.8;"><strong style="color:#00d4ff;">StellarSeq</strong> is an AI system for predicting astronaut genetic stress using ML trained on NASA OSDR data.<br><br><strong style="color:#ff6b9d;">Spaceflight</strong> induces biological stress:<ul style="color:#8b949e;"><li>Microgravity gene expression changes</li><li>Cosmic radiation effects</li><li>Circadian disruption</li><li>Immune modulation</li></ul></div>', unsafe_allow_html=True)
            st.markdown('<p class="section-header">DATA SOURCES</p>', unsafe_allow_html=True)
            st.markdown('<div style="font-family:Exo 2;color:#8b949e;line-height:1.8;">Trained on <strong style="color:#00ffa3;">9 spaceflight studies</strong> from NASA OSDR:<ul><li>Mouse & Human organisms</li><li>Multiple tissues</li><li>Spaceflight, radiation, ground analog</li></ul></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<p class="section-header">PERFORMANCE</p>', unsafe_allow_html=True)
            st.markdown('<div style="background:linear-gradient(135deg,rgba(0,212,255,0.1),rgba(255,107,157,0.05));border:1px solid rgba(0,212,255,0.3);border-radius:12px;padding:20px;"><table style="width:100%;font-family:Exo 2;color:#e6edf3;"><tr><td style="padding:8px 0;color:#8b949e;">Algorithm</td><td style="text-align:right;color:#00d4ff;font-family:Orbitron;">XGBoost</td></tr><tr><td style="padding:8px 0;color:#8b949e;">R² Score</td><td style="text-align:right;color:#00ffa3;font-family:Orbitron;">0.974</td></tr><tr><td style="padding:8px 0;color:#8b949e;">RMSE</td><td style="text-align:right;color:#00ffa3;font-family:Orbitron;">2.92</td></tr><tr><td style="padding:8px 0;color:#8b949e;">MAE</td><td style="text-align:right;color:#00ffa3;font-family:Orbitron;">2.09</td></tr><tr><td style="padding:8px 0;color:#8b949e;">Samples</td><td style="text-align:right;color:#ff6b9d;font-family:Orbitron;">112</td></tr></table></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<p class="section-header">FEATURE ANALYSIS</p>', unsafe_allow_html=True)
        fig = create_feature_importance_chart(model, feature_cols)
        if fig: st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div style="background:rgba(255,149,0,0.1);border:1px solid rgba(255,149,0,0.3);border-radius:8px;padding:16px;margin-top:20px;"><p style="color:#ff9500;font-family:Exo 2;margin:0;font-size:0.9rem;"><strong>⚠️ DISCLAIMER:</strong> For research and educational purposes only.</p></div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<p class="section-header">MANUAL INPUT</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Expression Statistics")
            mean_expr = st.number_input("Mean Expression (log2)", value=8.5, step=0.1)
            std_expr = st.number_input("Std Expression", value=2.0, step=0.1)
            median_expr = st.number_input("Median Expression", value=8.0, step=0.1)
            gene_count = st.number_input("Gene Count", value=15000, step=1000)
            st.markdown("##### Fold Change Metrics")
            mean_fc = st.number_input("Mean Fold Change", value=0.5, step=0.1)
            std_fc = st.number_input("Std Fold Change", value=1.0, step=0.1)
            mean_abs_fc = st.number_input("Mean Abs Fold Change", value=0.8, step=0.1)
        with c2:
            st.markdown("##### Differential Expression")
            n_up = st.number_input("# Upregulated", value=500, step=100)
            n_down = st.number_input("# Downregulated", value=500, step=100)
            n_sig = st.number_input("# Significant", value=1000, step=100)
            st.markdown("##### Sample Metadata")
            organism = st.selectbox("Organism", ["Mouse", "Human"])
            condition = st.selectbox("Condition", ["Ground Control", "Spaceflight", "Radiation", "Hindlimb Unload"])
            tissue = st.selectbox("Tissue", ["Thymus", "Liver", "Muscle", "Blood", "Retina", "Skin", "Other"])
            study_id = st.slider("Study ID", 0, 10, 5)
        
        if st.button("🚀 INITIATE PREDICTION", type="primary", use_container_width=True):
            org_map = {"Mouse": 0, "Human": 1}
            cond_map = {"Ground Control": 0, "Spaceflight": 1, "Radiation": 2, "Hindlimb Unload": 3}
            tis_map = {"Thymus": 0, "Liver": 1, "Muscle": 2, "Blood": 3, "Retina": 4, "Skin": 5, "Other": 6}
            features = pd.DataFrame([[mean_expr, std_expr, median_expr, gene_count, mean_fc, std_fc, mean_abs_fc, n_up, n_down, n_sig, org_map[organism], cond_map[condition], tis_map[tissue], study_id]], columns=feature_cols)
            pred = np.clip(predict_stress(model, scaler, features, feature_cols)[0], 0, 100)
            cat, css, emoji, color = get_stress_category(pred)
            st.markdown("---")
            st.markdown('<p class="section-header">ANALYSIS RESULTS</p>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2: st.plotly_chart(create_gauge_chart(pred), use_container_width=True)
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Stress Score", f"{pred:.1f}")
            with c2: st.markdown(f'<div style="text-align:center;padding:10px;"><span class="stress-badge {css}">{emoji} {cat}</span></div>', unsafe_allow_html=True)
            with c3: st.metric("Condition", condition)
            st.markdown("---")
            if cat == "LOW": st.success("**✓ LOW STRESS** — Minimal changes detected.")
            elif cat == "MODERATE": st.warning("**⚡ MODERATE STRESS** — Measurable signatures.")
            elif cat == "HIGH": st.warning("**⚠️ HIGH STRESS** — Significant dysregulation.")
            else: st.error("**🚨 SEVERE STRESS** — Immediate assessment recommended.")
    
    with tab3:
        st.markdown('<p class="section-header">NASA OSDR DATA UPLOAD</p>', unsafe_allow_html=True)
        with st.expander("📋 DATA FORMAT"):
            st.markdown("**Expected:** Gene expression matrix (ENSEMBL, SYMBOL, Sample columns)")
        uploaded = st.file_uploader("Upload Expression Matrix", type=['csv', 'tsv', 'txt'])
        c1, c2 = st.columns(2)
        with c1: ctrl = st.text_input("Control pattern (optional)", placeholder="_GC_")
        with c2: sid = st.number_input("Study ID", value=0)
        
        if uploaded:
            try:
                df = pd.read_csv(uploaded, sep='\t' if uploaded.name.endswith(('.tsv', '.txt')) else ',')
                st.success(f"✅ Loaded: **{len(df):,} genes** × **{len(df.columns)} columns**")
                with st.expander("Preview"): st.dataframe(df.head(10))
                if st.button("🔬 PROCESS & ANALYZE", type="primary", use_container_width=True):
                    with st.spinner("Processing..."):
                        feat_df = process_raw_expression_data(df, ctrl if ctrl else None)
                        feat_df['study_id_encoded'] = sid
                        st.success(f"✅ Processed **{len(feat_df)} samples**")
                        st.dataframe(feat_df[['sample_id', 'organism', 'condition', 'tissue']])
                        preds = np.clip(predict_stress(model, scaler, feat_df, feature_cols), 0, 100)
                        feat_df['predicted_stress_score'] = preds
                        feat_df['stress_category'] = [get_stress_category(p)[0] for p in preds]
                        st.markdown("---")
                        st.markdown('<p class="section-header">🎯 RESULTS</p>', unsafe_allow_html=True)
                        c1, c2, c3, c4 = st.columns(4)
                        with c1: st.metric("Mean", f"{preds.mean():.1f}")
                        with c2: st.metric("Min", f"{preds.min():.1f}")
                        with c3: st.metric("Max", f"{preds.max():.1f}")
                        with c4: st.metric("High Stress", f"{(preds >= 50).sum()}/{len(preds)}")
                        st.dataframe(feat_df[['sample_id', 'organism', 'condition', 'tissue', 'predicted_stress_score', 'stress_category']])
                        fig = px.histogram(feat_df, x='predicted_stress_score', color='condition', nbins=20, barmode='overlay', color_discrete_sequence=['#00d4ff', '#ff6b9d', '#00ffa3', '#ff9500'])
                        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family='Exo 2', color='#8b949e'))
                        st.plotly_chart(fig, use_container_width=True)
                        st.download_button("📥 DOWNLOAD", feat_df.to_csv(index=False), "stellarseq_results.csv", "text/csv")
            except Exception as e: st.error(f"Error: {e}")
    
    st.markdown('<div class="footer"><span style="color:#00d4ff;font-family:Orbitron;">STELLARSEQ</span> v1.0 &nbsp;|&nbsp; Powered by NASA OSDR Data &nbsp;|&nbsp; Built by <span style="color:#ff6b9d;">Meera Kirthiraj</span></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
