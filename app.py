import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import locale

# Force decimal separator to be "."
locale.setlocale(locale.LC_ALL, "C")

st.set_page_config(page_title="🤖 AI-Driven Adaptive Scheduling", layout="wide")
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #DA70D6;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Titles */
    h1, h2, h3, h4 {
        color: #191970;
        font-weight: 600;
    }

    /* Buttons */
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
    .stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #2563eb) !important;
        transform: scale(1.03) !important;
    }
    .stButton > button:active {
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

    /* Selectbox */
    .stSelectbox > div > div > select {
        background-color: #dcfce7 !important; /* light green */
        color: #1e293b !important;
        border-radius: 8px !important;
        border: 1px solid #94a3b8 !important;
        padding: 6px 10px !important;
    }

    /* MultiSelect */
    .stMultiSelect > div > div {
        background-color: #fef9c3 !important; /* light yellow */
        color: #1e293b !important;
        border-radius: 8px !important;
        border: 1px solid #94a3b8 !important;
        padding: 6px 10px !important;
    }

    /* DataFrame table */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* Success / Info boxes */
    .stSuccess {
        background-color: #dcfce7 !important;
        border-left: 6px solid #22c55e !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    .stInfo {
        background-color: #e0f2fe !important;
        border-left: 6px solid #2563eb !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }

    /* Custom Prediction Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
    }
    </style>
""", unsafe_allow_html=True)
st.title("🤖 AI-Driven Adaptive Scheduling")

# --------------------------
# Feature Engineering
# --------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Production_Load" in df and "Deadline_Hours" in df:
        df["urgency"] = df["Production_Load"] / (df["Deadline_Hours"] + 1e-3)
    if "Available_Operators" in df and "Available_Machines" in df:
        df["operator_machine_ratio"] = df["Available_Operators"] / (df["Available_Machines"] + 1)
    if "Expected_Runtime_Min" in df and "Machine_Efficiency" in df:
        df["adjusted_runtime"] = df["Expected_Runtime_Min"] / (df["Machine_Efficiency"] + 1e-3)
    if "Production_Load" in df and "Available_Operators" in df:
        df["load_per_operator"] = df["Production_Load"] / (df["Available_Operators"] + 1)
    if "Shift" in df:
        df["shift_binary"] = df["Shift"].apply(lambda x: 1 if str(x).lower() == "night" else 0)
    return df

# --------------------------
# File Upload
# --------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    df = add_engineered_features(df)

    engineered_features = ["urgency", "operator_machine_ratio", "adjusted_runtime",
                           "load_per_operator", "shift_binary"]

    st.write("✅ Dataset loaded successfully with engineered features!")
    st.dataframe(df.head())

    all_columns = df.columns.tolist()
    st.subheader("⚙ Select Features and Target Columns")

    input_cols = st.multiselect(
        "Select Input Columns (X)", 
        [c for c in all_columns if c not in engineered_features],
        default=[c for c in all_columns if c not in engineered_features]
    )
    output_cols = st.multiselect(
        "Select Output Columns (y)", 
        [c for c in all_columns if c not in engineered_features],
        default=[c for c in all_columns if c not in engineered_features and c not in input_cols]
    )

    if input_cols and output_cols and st.button("🚀 Train Model"):
        X = df[input_cols]
        y = df[output_cols]

        X_encoded = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred, multioutput="uniform_average")

        st.subheader("📊 Model Accuracy")
        st.write(f"✅ R² Score: {r2*100:.2f}%")

        st.session_state["model"] = model
        st.session_state["features"] = X_encoded.columns
        st.session_state["output_cols"] = output_cols
        st.session_state["input_cols"] = input_cols
        st.session_state["df"] = df

# --------------------------
# Prediction Section
# --------------------------
if "model" in st.session_state:
    st.subheader("🔧 Predict for New Input")

    df = st.session_state["df"]
    input_cols = st.session_state["input_cols"]
    output_cols = st.session_state["output_cols"]

    input_data = {}
    for col in input_cols:
        if df[col].dtype in ["int64", "float64"]:
            # Use text_input to enforce dot decimals
            raw_val = st.text_input(
                f"{col} (use . for decimals)", 
                value=f"{df[col].mean():.2f}"
            )
            try:
                val = float(raw_val.replace(",", "."))  # convert commas to dots
            except ValueError:
                val = float(df[col].mean())
            input_data[col] = val
        else:
            options = df[col].unique().tolist()
            val = st.selectbox(f"{col}", options)
            input_data[col] = val

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])

        input_df = add_engineered_features(input_df)

        input_encoded = pd.get_dummies(input_df, drop_first=True)
        input_encoded = input_encoded.reindex(columns=st.session_state["features"], fill_value=0)

        prediction = st.session_state["model"].predict(input_encoded).flatten()

        st.success("🎯 Predictions:")
        for i, col in enumerate(output_cols):
            if col.lower() in ["machine", "manpower"]:
                val = int(round(prediction[i]))
            else:
                val = float(f"{prediction[i]:.2f}")  # always with dot
            st.markdown(f'<div class="metric-card">{col}: {val}</div>', unsafe_allow_html=True)
else:
    st.info("📥 Please upload a CSV, select columns, and click 🚀 Train Model")
