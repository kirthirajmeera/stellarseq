#!/usr/bin/env python3
"""
StellarSeq - Astronaut Genetic Stress Response Predictor
Professional Space-themed Dashboard

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
# REFINED SPACE THEME - Sora + Barlow
# ============================================================================
st.markdown("""
<style>
    /* Import Sora + Barlow */
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=Barlow:wght@300;400;500;600;700&display=swap');
    
    :root {
        --space-black: #05080d;
        --deep-space: #0a0e17;
        --nebula-dark: #111827;
        --star-white: #f0f4f8;
        --nebula-pink: #ec4899;
        --cosmic-cyan: #06b6d4;
        --aurora-green: #10b981;
        --solar-orange: #f59e0b;
        --warning-red: #ef4444;
        --muted: #9ca3af;
        --card-bg: rgba(17, 24, 39, 0.7);
    }
    
    /* Main background with dense starfield */
    .stApp {
        background: radial-gradient(ellipse at top, #0f172a 0%, #05080d 50%, #000000 100%);
        background-attachment: fixed;
    }
    
    /* Dense animated starfield */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        background-image: 
            /* Bright stars */
            radial-gradient(2px 2px at 50px 50px, #ffffff, transparent),
            radial-gradient(2px 2px at 150px 80px, #ffffff, transparent),
            radial-gradient(2px 2px at 250px 120px, #ffffff, transparent),
            radial-gradient(2px 2px at 350px 40px, #ffffff, transparent),
            radial-gradient(2px 2px at 450px 160px, #ffffff, transparent),
            radial-gradient(2px 2px at 550px 90px, #ffffff, transparent),
            radial-gradient(2px 2px at 650px 140px, #ffffff, transparent),
            radial-gradient(2px 2px at 750px 60px, #ffffff, transparent),
            radial-gradient(2px 2px at 850px 180px, #ffffff, transparent),
            radial-gradient(2px 2px at 950px 100px, #ffffff, transparent),
            /* Medium stars */
            radial-gradient(1.5px 1.5px at 100px 30px, rgba(255,255,255,0.9), transparent),
            radial-gradient(1.5px 1.5px at 200px 150px, rgba(255,255,255,0.9), transparent),
            radial-gradient(1.5px 1.5px at 300px 70px, rgba(255,255,255,0.9), transparent),
            radial-gradient(1.5px 1.5px at 400px 190px, rgba(255,255,255,0.9), transparent),
            radial-gradient(1.5px 1.5px at 500px 110px, rgba(255,255,255,0.9), transparent),
            radial-gradient(1.5px 1.5px at 600px 20px, rgba(255,255,255,0.9), transparent),
            radial-gradient(1.5px 1.5px at 700px 170px, rgba(255,255,255,0.9), transparent),
            radial-gradient(1.5px 1.5px at 800px 130px, rgba(255,255,255,0.9), transparent),
            radial-gradient(1.5px 1.5px at 900px 50px, rgba(255,255,255,0.9), transparent),
            /* Small stars */
            radial-gradient(1px 1px at 25px 80px, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 75px 140px, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 125px 20px, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 175px 100px, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 225px 180px, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 275px 60px, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 325px 130px, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 375px 10px, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 425px 90px, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 475px 170px, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 525px 40px, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 575px 120px, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 625px 200px, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 675px 70px, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 725px 150px, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 775px 30px, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 825px 110px, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 875px 190px, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 925px 55px, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 975px 135px, rgba(255,255,255,0.7), transparent),
            /* Colored accent stars */
            radial-gradient(2px 2px at 180px 45px, rgba(6,182,212,0.8), transparent),
            radial-gradient(2px 2px at 420px 125px, rgba(236,72,153,0.8), transparent),
            radial-gradient(2px 2px at 680px 85px, rgba(16,185,129,0.8), transparent),
            radial-gradient(2px 2px at 880px 165px, rgba(6,182,212,0.8), transparent);
        background-repeat: repeat;
        background-size: 1000px 220px;
        animation: twinkle 4s ease-in-out infinite alternate;
        z-index: 0;
    }
    
    @keyframes twinkle {
        0% { opacity: 0.6; }
        100% { opacity: 1; }
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10,14,23,0.97) 0%, rgba(17,24,39,0.97) 100%);
        border-right: 1px solid rgba(6,182,212,0.15);
    }
    
    [data-testid="stSidebar"] .stMarkdown { color: var(--star-white); }
    
    /* Typography */
    .main-header {
        font-family: 'Sora', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0;
        padding-top: 1rem;
        background: linear-gradient(135deg, #06b6d4 0%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 1px;
    }
    
    .sub-header {
        font-family: 'Barlow', sans-serif;
        font-size: 1.1rem;
        font-weight: 400;
        color: var(--muted);
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    .section-header {
        font-family: 'Sora', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--cosmic-cyan);
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(6,182,212,0.3);
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(17,24,39,0.6);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid rgba(6,182,212,0.15);
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Sora', sans-serif;
        font-weight: 500;
        font-size: 0.9rem;
        color: var(--muted);
        background: transparent;
        border-radius: 6px;
        padding: 10px 20px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--cosmic-cyan);
        background: rgba(6,182,212,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(6,182,212,0.15) !important;
        color: var(--star-white) !important;
        border: 1px solid rgba(6,182,212,0.3);
    }
    
    .stMarkdown, .stText { color: var(--star-white); font-family: 'Barlow', sans-serif; }
    
    /* Inputs */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        background: rgba(17,24,39,0.8) !important;
        border: 1px solid rgba(6,182,212,0.2) !important;
        color: var(--star-white) !important;
        border-radius: 6px !important;
        font-family: 'Barlow', sans-serif !important;
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'Sora', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 1px !important;
        background: linear-gradient(135deg, #06b6d4 0%, #10b981 100%) !important;
        color: #05080d !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 10px 24px !important;
        text-transform: uppercase !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(6,182,212,0.35) !important;
    }
    
    .stDownloadButton > button {
        font-family: 'Barlow', sans-serif !important;
        background: rgba(6,182,212,0.15) !important;
        border: 1px solid rgba(6,182,212,0.3) !important;
        color: var(--cosmic-cyan) !important;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: var(--card-bg);
        border: 1px solid rgba(6,182,212,0.15);
        border-radius: 8px;
        padding: 14px;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Barlow', sans-serif !important;
        color: var(--muted) !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetricValue"] {
        font-family: 'Sora', sans-serif !important;
        color: var(--cosmic-cyan) !important;
        font-weight: 600 !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(17,24,39,0.5);
        border: 1px dashed rgba(6,182,212,0.3);
        border-radius: 8px;
        padding: 16px;
    }
    
    /* Messages */
    .stSuccess { background: rgba(16,185,129,0.1) !important; border: 1px solid rgba(16,185,129,0.3) !important; border-radius: 6px !important; }
    .stWarning { background: rgba(245,158,11,0.1) !important; border: 1px solid rgba(245,158,11,0.3) !important; border-radius: 6px !important; }
    .stError { background: rgba(239,68,68,0.1) !important; border: 1px solid rgba(239,68,68,0.3) !important; border-radius: 6px !important; }
    
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(6,182,212,0.2), transparent);
        margin: 1.5rem 0;
    }
    
    /* Clean Stress Index Tiles */
    .stress-index-container {
        display: flex;
        flex-direction: column;
        gap: 8px;
        margin-top: 8px;
    }
    
    .stress-tile {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 14px;
        border-radius: 6px;
        font-family: 'Barlow', sans-serif;
        font-weight: 500;
        font-size: 0.85rem;
        transition: all 0.2s ease;
    }
    
    .stress-tile:hover {
        transform: translateX(4px);
    }
    
    .stress-tile-low {
        background: rgba(16,185,129,0.1);
        border: 1px solid rgba(16,185,129,0.3);
        color: #10b981;
    }
    
    .stress-tile-moderate {
        background: rgba(245,158,11,0.1);
        border: 1px solid rgba(245,158,11,0.3);
        color: #f59e0b;
    }
    
    .stress-tile-high {
        background: rgba(236,72,153,0.1);
        border: 1px solid rgba(236,72,153,0.3);
        color: #ec4899;
    }
    
    .stress-tile-severe {
        background: rgba(239,68,68,0.1);
        border: 1px solid rgba(239,68,68,0.3);
        color: #ef4444;
    }
    
    .stress-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    
    .stress-range {
        margin-left: auto;
        font-size: 0.75rem;
        opacity: 0.7;
    }
    
    .footer {
        font-family: 'Barlow', sans-serif;
        text-align: center;
        color: var(--muted);
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid rgba(6,182,212,0.15);
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    
    .sidebar-card {
        background: rgba(6,182,212,0.05);
        border: 1px solid rgba(6,182,212,0.15);
        border-radius: 8px;
        padding: 14px;
        margin: 10px 0;
    }
    
    /* Dialog/Modal styling */
    [data-testid="stModal"] {
        background: rgba(10,14,23,0.98) !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--deep-space); }
    ::-webkit-scrollbar-thumb { background: var(--cosmic-cyan); border-radius: 3px; }
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
    if score < 25: return "LOW", "stress-tile-low", "🟢", "#10b981"
    elif score < 50: return "MODERATE", "stress-tile-moderate", "🟡", "#f59e0b"
    elif score < 75: return "HIGH", "stress-tile-high", "🔴", "#ec4899"
    else: return "SEVERE", "stress-tile-severe", "⛔", "#ef4444"


def create_gauge_chart(score):
    _, _, _, color = get_stress_category(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'font': {'size': 48, 'color': color, 'family': 'Sora'}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "#6b7280", 'tickfont': {'color': '#6b7280', 'family': 'Barlow', 'size': 10}},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "rgba(17,24,39,0.8)",
            'borderwidth': 1,
            'bordercolor': "rgba(6,182,212,0.2)",
            'steps': [
                {'range': [0, 25], 'color': 'rgba(16,185,129,0.12)'},
                {'range': [25, 50], 'color': 'rgba(245,158,11,0.12)'},
                {'range': [50, 75], 'color': 'rgba(236,72,153,0.12)'},
                {'range': [75, 100], 'color': 'rgba(239,68,68,0.12)'}
            ],
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def create_feature_importance_chart(model, feature_names):
    try:
        if not hasattr(model, 'feature_importances_'): return None
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig = go.Figure(go.Bar(
            x=importances[indices], y=[feature_names[i] for i in indices],
            orientation='h',
            marker=dict(color=importances[indices], colorscale=[[0, '#10b981'], [0.5, '#06b6d4'], [1, '#ec4899']])
        ))
        fig.update_layout(
            title=dict(text="Feature Importance", font=dict(family='Sora', size=14, color='#06b6d4')),
            height=350, margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(6,182,212,0.1)', color='#9ca3af'),
            yaxis=dict(categoryorder='total ascending', color='#9ca3af'),
            font=dict(family='Barlow', color='#9ca3af', size=11)
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
    results = []
    for s in sample_cols:
        m = parse_sample_name(s)
        fc = expr[s] - ctrl_mean
        results.append({'sample_id': s, 'mean_expression': expr[s].mean(), 'std_expression': expr[s].std(), 'median_expression': expr[s].median(), 'gene_count': (expr[s] > 0).sum(), 'mean_fold_change': fc.mean(), 'std_fold_change': fc.std(), 'mean_abs_fold_change': fc.abs().mean(), 'n_upregulated': (fc > 1).sum(), 'n_downregulated': (fc < -1).sum(), 'n_significant': ((fc > 1) | (fc < -1)).sum(), 'organism_code_encoded': m['organism_code'], 'condition_encoded': m['condition_code'], 'tissue_encoded': m['tissue_code'], 'study_id_encoded': 0, 'organism': m['organism'], 'condition': m['condition'], 'tissue': m['tissue']})
    return pd.DataFrame(results)


# Dialog for Single Prediction Results
@st.dialog("Analysis Results", width="large")
def show_single_result(pred, condition):
    cat, css, emoji, color = get_stress_category(pred)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.plotly_chart(create_gauge_chart(pred), use_container_width=True)
    
    with col2:
        st.markdown(f"""
        <div style="padding: 20px; text-align: center;">
            <div style="font-family: Sora; font-size: 0.8rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Classification</div>
            <div style="font-family: Sora; font-size: 1.8rem; font-weight: 600; color: {color}; margin-bottom: 16px;">{emoji} {cat}</div>
            <div style="font-family: Barlow; font-size: 0.85rem; color: #9ca3af; margin-bottom: 4px;">Condition</div>
            <div style="font-family: Sora; font-size: 1rem; color: #f0f4f8;">{condition}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if cat == "LOW":
        st.success("**Low Stress** — Minimal gene expression changes. Sample within normal parameters.")
    elif cat == "MODERATE":
        st.warning("**Moderate Stress** — Measurable signatures detected. Adaptive responses in manageable range.")
    elif cat == "HIGH":
        st.warning("**High Stress** — Significant dysregulation. Recommend monitoring for physiological impacts.")
    else:
        st.error("**Severe Stress** — Extensive changes detected. Immediate assessment recommended.")
    
    if st.button("Close", use_container_width=True):
        st.rerun()


# Dialog for Batch Prediction Results  
@st.dialog("Batch Analysis Results", width="large")
def show_batch_results(feat_df, preds):
    feat_df['predicted_stress_score'] = preds
    feat_df['stress_category'] = [get_stress_category(p)[0] for p in preds]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Mean", f"{preds.mean():.1f}")
    with col2: st.metric("Min", f"{preds.min():.1f}")
    with col3: st.metric("Max", f"{preds.max():.1f}")
    with col4: st.metric("High Stress", f"{(preds >= 50).sum()}/{len(preds)}")
    
    st.markdown("---")
    
    st.dataframe(
        feat_df[['sample_id', 'organism', 'condition', 'tissue', 'predicted_stress_score', 'stress_category']].round({'predicted_stress_score': 1}),
        use_container_width=True,
        height=250
    )
    
    fig = px.histogram(feat_df, x='predicted_stress_score', color='condition', nbins=15, barmode='overlay', 
                       color_discrete_sequence=['#06b6d4', '#ec4899', '#10b981', '#f59e0b'])
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Barlow', color='#9ca3af', size=11),
        height=250, margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(gridcolor='rgba(6,182,212,0.1)'),
        yaxis=dict(gridcolor='rgba(6,182,212,0.1)')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download CSV", feat_df.to_csv(index=False), "stellarseq_results.csv", "text/csv", use_container_width=True)
    with col2:
        if st.button("Close", use_container_width=True):
            st.rerun()


def main():
    st.markdown('<h1 class="main-header">StellarSeq</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Astronaut Genetic Stress Response Predictor</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown('<div style="text-align:center;padding:16px 0;"><span style="font-size:3rem;">🛸</span></div>', unsafe_allow_html=True)
        st.markdown('<p class="section-header">Mission Brief</p>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-card"><p style="color:#f0f4f8;font-family:Barlow;margin:0;font-size:0.9rem;line-height:1.6;"><strong style="color:#06b6d4;">StellarSeq</strong> predicts astronaut stress from gene expression using ML trained on NASA spaceflight genomics.</p></div>', unsafe_allow_html=True)
        
        st.markdown('<p class="section-header">Model Stats</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: st.metric("R² Score", "0.974"); st.metric("Samples", "112")
        with c2: st.metric("RMSE", "2.92"); st.metric("Features", "14")
        
        st.markdown("---")
        st.markdown('<p class="section-header">Stress Index</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="stress-index-container">
            <div class="stress-tile stress-tile-low">
                <div class="stress-dot" style="background:#10b981;"></div>
                <span>Low</span>
                <span class="stress-range">0-25</span>
            </div>
            <div class="stress-tile stress-tile-moderate">
                <div class="stress-dot" style="background:#f59e0b;"></div>
                <span>Moderate</span>
                <span class="stress-range">25-50</span>
            </div>
            <div class="stress-tile stress-tile-high">
                <div class="stress-dot" style="background:#ec4899;"></div>
                <span>High</span>
                <span class="stress-range">50-75</span>
            </div>
            <div class="stress-tile stress-tile-severe">
                <div class="stress-dot" style="background:#ef4444;"></div>
                <span>Severe</span>
                <span class="stress-range">75-100</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    model, scaler, metadata = load_model()
    if model is None: st.error("⚠️ Model not loaded."); return
    
    feature_cols = ['mean_expression', 'std_expression', 'median_expression', 'gene_count', 'mean_fold_change', 'std_fold_change', 'mean_abs_fold_change', 'n_upregulated', 'n_downregulated', 'n_significant', 'organism_code_encoded', 'condition_encoded', 'tissue_encoded', 'study_id_encoded']
    
    tab1, tab2, tab3 = st.tabs(["ℹ️ About", "🎯 Single Prediction", "📡 Batch Upload"])
    
    # ABOUT TAB
    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown('<p class="section-header">Mission Overview</p>', unsafe_allow_html=True)
            st.markdown("""
            <div style="font-family:Barlow;color:#f0f4f8;line-height:1.8;font-size:0.95rem;">
            <strong style="color:#06b6d4;">StellarSeq</strong> is an AI system for predicting astronaut genetic stress using machine learning trained on NASA's Open Science Data Repository (OSDR).
            <br><br>
            <strong style="color:#ec4899;">Spaceflight</strong> induces biological stress responses:
            </div>
            <ul style="font-family:Barlow;color:#9ca3af;line-height:1.8;font-size:0.9rem;margin-top:8px;">
                <li>Microgravity-induced gene expression changes</li>
                <li>Cosmic radiation exposure effects</li>
                <li>Circadian rhythm disruption</li>
                <li>Immune system modulation</li>
            </ul>
            """, unsafe_allow_html=True)
            
            st.markdown('<p class="section-header">Data Sources</p>', unsafe_allow_html=True)
            st.markdown("""
            <div style="font-family:Barlow;color:#9ca3af;line-height:1.8;font-size:0.9rem;">
            Trained on <strong style="color:#10b981;">9 spaceflight studies</strong> from NASA OSDR including mouse and human organisms across multiple tissue types.
            </div>
            """, unsafe_allow_html=True)
        
        with c2:
            st.markdown('<p class="section-header">Performance</p>', unsafe_allow_html=True)
            st.markdown("""
            <div style="background:rgba(6,182,212,0.05);border:1px solid rgba(6,182,212,0.2);border-radius:8px;padding:16px;">
                <table style="width:100%;font-family:Barlow;color:#f0f4f8;font-size:0.9rem;">
                    <tr><td style="padding:6px 0;color:#9ca3af;">Algorithm</td><td style="text-align:right;color:#06b6d4;font-family:Sora;font-weight:500;">XGBoost</td></tr>
                    <tr><td style="padding:6px 0;color:#9ca3af;">R² Score</td><td style="text-align:right;color:#10b981;font-family:Sora;font-weight:500;">0.974</td></tr>
                    <tr><td style="padding:6px 0;color:#9ca3af;">RMSE</td><td style="text-align:right;color:#10b981;font-family:Sora;font-weight:500;">2.92</td></tr>
                    <tr><td style="padding:6px 0;color:#9ca3af;">MAE</td><td style="text-align:right;color:#10b981;font-family:Sora;font-weight:500;">2.09</td></tr>
                    <tr><td style="padding:6px 0;color:#9ca3af;">Samples</td><td style="text-align:right;color:#ec4899;font-family:Sora;font-weight:500;">112</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="section-header">Feature Analysis</p>', unsafe_allow_html=True)
        fig = create_feature_importance_chart(model, feature_cols)
        if fig: st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);border-radius:6px;padding:14px;margin-top:16px;">
            <p style="color:#f59e0b;font-family:Barlow;margin:0;font-size:0.85rem;">
            <strong>Disclaimer:</strong> For research and educational purposes only. Predictions should be validated by qualified researchers.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # SINGLE PREDICTION TAB
    with tab2:
        st.markdown('<p class="section-header">Manual Input</p>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Expression Statistics")
            mean_expr = st.number_input("Mean Expression (log2)", value=8.5, step=0.1, key="s_mean")
            std_expr = st.number_input("Std Expression", value=2.0, step=0.1, key="s_std")
            median_expr = st.number_input("Median Expression", value=8.0, step=0.1, key="s_med")
            gene_count = st.number_input("Gene Count", value=15000, step=1000, key="s_gc")
            st.markdown("##### Fold Change Metrics")
            mean_fc = st.number_input("Mean Fold Change", value=0.5, step=0.1, key="s_mfc")
            std_fc = st.number_input("Std Fold Change", value=1.0, step=0.1, key="s_sfc")
            mean_abs_fc = st.number_input("Mean Abs Fold Change", value=0.8, step=0.1, key="s_mafc")
        
        with c2:
            st.markdown("##### Differential Expression")
            n_up = st.number_input("# Upregulated", value=500, step=100, key="s_up")
            n_down = st.number_input("# Downregulated", value=500, step=100, key="s_down")
            n_sig = st.number_input("# Significant", value=1000, step=100, key="s_sig")
            st.markdown("##### Sample Metadata")
            organism = st.selectbox("Organism", ["Mouse", "Human"], key="s_org")
            condition = st.selectbox("Condition", ["Ground Control", "Spaceflight", "Radiation", "Hindlimb Unload"], key="s_cond")
            tissue = st.selectbox("Tissue", ["Thymus", "Liver", "Muscle", "Blood", "Retina", "Skin", "Other"], key="s_tis")
            study_id = st.slider("Study ID", 0, 10, 5, key="s_sid")
        
        st.markdown("")
        
        if st.button("🚀 Analyze Sample", type="primary", use_container_width=True, key="single_btn"):
            org_map = {"Mouse": 0, "Human": 1}
            cond_map = {"Ground Control": 0, "Spaceflight": 1, "Radiation": 2, "Hindlimb Unload": 3}
            tis_map = {"Thymus": 0, "Liver": 1, "Muscle": 2, "Blood": 3, "Retina": 4, "Skin": 5, "Other": 6}
            features = pd.DataFrame([[mean_expr, std_expr, median_expr, gene_count, mean_fc, std_fc, mean_abs_fc, n_up, n_down, n_sig, org_map[organism], cond_map[condition], tis_map[tissue], study_id]], columns=feature_cols)
            pred = np.clip(predict_stress(model, scaler, features, feature_cols)[0], 0, 100)
            show_single_result(pred, condition)
    
    # BATCH UPLOAD TAB
    with tab3:
        st.markdown('<p class="section-header">NASA OSDR Data Upload</p>', unsafe_allow_html=True)
        
        with st.expander("📋 Data Format Specification"):
            st.markdown("""
            **Expected format:** Gene expression matrix with genes as rows, samples as columns.
            
            | ENSEMBL | SYMBOL | Sample_FLT_Rep1 | Sample_GC_Rep1 | ... |
            |---------|--------|-----------------|----------------|-----|
            | ENSMUSG... | Gnai3 | 1245.6 | 1198.4 | ... |
            
            **Sample naming:** `Mmus_LVR_FLT_Rep1` → Mouse, Liver, Flight
            """)
        
        uploaded = st.file_uploader("Upload Expression Matrix", type=['csv', 'tsv', 'txt'], key="batch_file")
        
        c1, c2 = st.columns(2)
        with c1: ctrl = st.text_input("Control pattern (optional)", placeholder="_GC_", key="b_ctrl")
        with c2: sid = st.number_input("Study ID", value=0, key="b_sid")
        
        if uploaded:
            try:
                df = pd.read_csv(uploaded, sep='\t' if uploaded.name.endswith(('.tsv', '.txt')) else ',')
                st.success(f"✅ Loaded: **{len(df):,} genes** × **{len(df.columns)} columns**")
                
                with st.expander("Preview Data"):
                    st.dataframe(df.head(8), use_container_width=True)
                
                if st.button("🔬 Process & Analyze", type="primary", use_container_width=True, key="batch_btn"):
                    with st.spinner("Processing expression data..."):
                        feat_df = process_raw_expression_data(df, ctrl if ctrl else None)
                        feat_df['study_id_encoded'] = sid
                        preds = np.clip(predict_stress(model, scaler, feat_df, feature_cols), 0, 100)
                        show_batch_results(feat_df, preds)
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown('<div class="footer"><span style="color:#06b6d4;font-family:Sora;font-weight:500;">StellarSeq</span> v1.0 &nbsp;•&nbsp; Powered by NASA OSDR Data &nbsp;•&nbsp; Built by <span style="color:#ec4899;">Meera Kirthiraj</span></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
