import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.set_page_config(page_title="AI-Driven Adaptive Scheduling", layout="wide")

# ----------------------------
# CSS Styling
# ----------------------------
st.markdown("""
    <style>
    body {
        background-color: #000000;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background-color: #f8fafc;
    }
    h1, h2, h3 {
        color: #1e293b;
        font-weight: 600;
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
        border: 1px solid #94a3b8 !impo
