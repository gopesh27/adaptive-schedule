import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --------------------------------
# Page config
# --------------------------------
st.set_page_config(page_title="AI-Driven Adaptive Scheduling", layout="wide")

# --------------------------------
# CSS Styling
# --------------------------------
st.markdown("""
    <style>
    body {
        background-color: #000000;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background-color: #f8fafc;
    }
    h1, h2, h3, h4 {
        color: #1e293b;
        font-weight: 600;
    }
    .stButton > button {
        background: linear-gradient(90deg, #2563eb, #1d4ed8) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6em 1.2em !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        transition: all 0.3s ease-in-out !important;
    }
    .stButton > button:active {
        background: linear-gradient(90deg, #1d4ed8, #2563eb) !important;
        transform: scale(0.97) !important;
    }
    /* Number Input Fields */
    .stNumberInput > div > div > input {
        background-color: #e0f2fe !important; /* light blue */
        color: #1e293b !important;
        border-radius: 8px !important;
        border: 1px solid #94a3b8 !important;
        padding: 6px 10px !important;
    }
    /* Selectbox Fields */
    .stSelectbox > div > div > select {
        background-color: #dcfce7 !important; /* light green */
        color: #1e293b !important;
        border-radius: 8px !important;
        border: 1px solid #94a3b8 !important;
        padding: 6px 10px !important;
    }
    /* Multiselect */
    .stMultiSelect > div > div {
        background-color: #fef9c3 !important; /* light yellow */
        color: #1e293b !important;
        border-radius: 8px !important;
        border: 1px solid #94a3b8 !important;
        padding: 6px 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 AI-Driven Adaptive Scheduling")

# ----------------------------
# Feature Engineering Function
# ----------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "production_load" in_
