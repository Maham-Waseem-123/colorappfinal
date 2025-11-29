# app.py
# ============================================
# STREAMLIT RESERVOIR ENGINEERING APP
# DARK THEME, GLASSY SIDEBAR, YELLOW ACCENTS
# All three pages included
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px

# ============================================
# PAGE CONFIG & GLOBAL CSS (GLASS THEME)
# ============================================
st.set_page_config(page_title="Reservoir Engineering App", layout="wide")

GLOBAL_CSS = """
<style>
/* Background image (darkened) */
[data-testid="stAppViewContainer"] {
  background: linear-gradient(rgba(10,10,10,0.65), rgba(10,10,10,0.65)), 
              url("https://raw.githubusercontent.com/Maham-Waseem-123/colorappfinal/main/patrick-hendry-6xeDIZgoPaw-unsplash.jpg");
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
}

/* Base text */
body, .block-container, .stText, .stMarkdown, .stDataFrame, .stButton, .stSlider {
    color: white !important;
    font-family: "Segoe UI", sans-serif;
}

/* Headings & titles (white + bold) */
h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2 {
    color: white !important;
    font-weight: 800 !important;
    text-shadow: 0 1px 6px rgba(0,0,0,0.7);
}

/* Glass sidebar (approximation, Streamlit markup class may vary) */
.css-1d391kg, .css-1lcbmhc { 
    background-color: rgba(255, 255, 255, 0.06) !important; 
    backdrop-filter: blur(14px); 
    color: white !important;
    border-radius: 12px;
    padding: 12px;
}

/* Buttons */
div.stButton > button {
    background-color: rgba(255, 255, 255, 0.12);
    color: white;
    font-weight: 700;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.18);
    padding: 8px 14px;
    transition: transform 0.18s ease, background-color 0.18s ease;
}
div.stButton > button:hover { 
    background-color: rgba(255,255,255,0.28); 
    transform: translateY(-2px);
}

/* Plot card */
.stPlotlyChart { 
    background-color: rgba(0,0,0,0.36); 
    border-radius: 14px; 
    padding: 10px; 
}

/* Dataframe rows */
.stDataFrame div.row_widget { 
    background-color: rgba(255, 255, 255, 0.06); 
    color: white; 
}

/* Sliders styling (prediction page kept yellow) */
div.stSlider > label {
    color: white !important;
    font-weight: 700 !important;
}
div.stSlider > div > div > div > div > div[role="slider"] {
    background-color: #ffd700 !important;
    border: 2px solid #ffd700 !important;
}
div.stSlider > div > div > div > div > div[role="slider"]:hover { box-shadow: 0 0 10px #ffd700; }

/* GLASS CARD */
.glass-card {
    background-color: rgba(255,255,255,0.04);
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 18px;
}

/* Segmented control (glass buttons) */
div[data-baseweb="segmented-control"] {
    background: rgba(255,255,255,0.06) !important;
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 6px;
}
button[data-baseweb="segment"] {
    background: rgba(255,255,255,0.08) !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.08);
    font-weight: 600;
    padding: 8px 10px;
}
button[data-baseweb="segment"][aria-selected="true"] {
    background: rgba(255,255,255,0.92) !important; /* bright white */
    color: black !important;                         /* numbers black */
    font-weight: 800 !important;
}

/* Number inputs styled as white glass with black numbers */
input[type="number"], .stNumberInput input {
    background-color: rgba(255,255,255,0.92) !important;
    color: black !important;
    font-weight: 700 !important;
    border-radius: 10px;
    padding: 8px 10px;
    border: 1px solid rgba(0,0,0,0.08) !important;
}

/* Metric containers (glass) */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 16px;
    border: 1px solid rgba(255,255,255,0.07);
}
[data-testid="metric-container"] .label, [data-testid="metric-container"] .value {
    color: white !important;
    font-weight: 700 !important;
}

/* Success box */
.stAlert, .stSuccess {
    background: rgba(255,255,255,0.06) !important;
    color: white !important;
    border-radius: 10px;
    padding: 12px;
    border: 1px solid rgba(255,255,255,0.07);
}

/* Small helpers */
.small-muted { color: rgba(255,255,255,0.6); font-size:12px; }
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ============================================
# LOAD DATA
# ============================================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Maham-Waseem-123/Final_project/main/Shale_Test.csv"
    df = pd.read_csv(url)
    # Clean column names
    df.columns = df.columns.str.strip().str.replace("\n", "").str.replace("\xa0", "")
    df = df.apply(pd.to_numeric, errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Fill with column means
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df

df = load_data()

# ============================================
# TRAIN MODEL
# ============================================
@st.cache_resource
def train_model(df):
    target = "Production (MMcfge)"
    feature_cols = df.drop(columns=["ID", target]).columns.tolist()
    numeric_cols = feature_cols.copy()
    X = df[feature_cols].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    gbr = GradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.1,
        n_estimators=600,
        max_depth=1,
        random_state=42,
        max_features=5
    )
    gbr.fit(X_train, y_train)
    return gbr, scaler, feature_cols, numeric_cols

model, scaler, feature_cols, numeric_cols = train_model(df)

# ============================================
# SIDEBAR NAVIGATION
# ============================================
st.sidebar.title("Pages")
page = st.sidebar.radio(
    "Select a Page:",
    ["Reservoir Engineering Dashboard", "Reservoir Prediction", "Economic Analysis"],
    index=0,
    format_func=lambda x: "📊 " + x if x == "Reservoir Engineering Dashboard" else ("🛢️ " + x if x == "Reservoir Prediction" else "💰 " + x)
)

# ============================================
# DASHBOARD PAGE
# ============================================
if page == "Reservoir Engineering Dashboard":
    st.markdown("<div class='glass-card'><h1 style='text-align:center;'>Reservoir Engineering Dashboard</h1></div>", unsafe_allow_html=True)

    features_to_plot = [
        "Porosity",
        "Additive per foot (bbls)",
        "Water per foot (bbls)",
        "Proppant per foot (lbs)"
    ]

    col1, col2 = st.columns(2)
    for i, feature in enumerate(features_to_plot):
        target_col = col1 if i % 2 == 0 else col2
        # Safe guard for missing features
        if feature not in df.columns:
            target_col.markdown(f"<div class='glass-card'><h4>{feature} not available in data</h4></div>", unsafe_allow_html=True)
            continue

        df['bin'] = pd.cut(df[feature], bins=10, duplicates='drop')
        binned_df = df.groupby('bin', as_index=False)['Production (MMcfge)'].mean()
        binned_df['bin_center'] = binned_df['bin'].apply(lambda x: x.mid if x is not None else np.nan)
        binned_df = binned_df.dropna(subset=['bin_center', 'Production (MMcfge)']).sort_values('bin_center')

        fig = px.line(
            binned_df,
            x='bin_center',
            y='Production (MMcfge)',
            labels={'bin_center': feature, 'Production (MMcfge)': 'Production (MMcfge)'},
            markers=True,
            title=f"{feature} vs Production"
        )
        fig.update_traces(line=dict(color='yellow', width=3), marker=dict(color='yellow', size=8))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Segoe UI', size=12),
            xaxis=dict(showgrid=False, title=dict(text=feature, font=dict(color='white', size=13))),
            yaxis=dict(showgrid=False, title=dict(text='Production (MMcfge)', font=dict(color='white', size=13)))
        )

        target_col.markdown(f"<div class='glass-card'><h4 style='margin-bottom:6px'>{feature} vs Production</h4></div>", unsafe_allow_html=True)
        target_col.plotly_chart(fig, use_container_width=True)

    # Depth chart full width (if column exists)
    if "Depth (feet)" in df.columns:
        st.markdown("<div class='glass-card'><h4>Depth (feet) vs Production</h4></div>", unsafe_allow_html=True)
        df['Depth_bin'] = pd.cut(df["Depth (feet)"], bins=10, duplicates='drop')
        binned_depth_df = df.groupby('Depth_bin', as_index=False)['Production (MMcfge)'].mean()
        binned_depth_df['bin_center'] = binned_depth_df['Depth_bin'].apply(lambda x: x.mid if x is not None else np.nan)
        binned_depth_df = binned_depth_df.dropna(subset=['bin_center', 'Production (MMcfge)']).sort_values('bin_center')

        fig_depth = px.line(
            binned_depth_df,
            x='bin_center',
            y='Production (MMcfge)',
            labels={'bin_center': 'Depth (feet)', 'Production (MMcfge)': 'Production (MMcfge)'},
            markers=True,
            title="Depth vs Production"
        )
        fig_depth.update_traces(line=dict(color='yellow', width=4), marker=dict(color='yellow', size=10))
        fig_depth.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Segoe UI', size=12),
            xaxis=dict(showgrid=False, title=dict(text='Depth (feet)', font=dict(color='white', size=13))),
            yaxis=dict(showgrid=False, title=dict(text='Production (MMcfge)', font=dict(color='white', size=13)))
        )
        st.plotly_chart(fig_depth, use_container_width=True)
    else:
        st.info("Depth (feet) column not found in data.", icon="ℹ️")

# ============================================
# RESERVOIR PREDICTION PAGE
# ============================================
elif page == "Reservoir Prediction":
    st.markdown("<div class='glass-card'><h1 style='text-align:center;'>Predict New Well Production</h1></div>", unsafe_allow_html=True)
    st.markdown("<p class='small-muted' style='text-align:center;'>Adjust parameters and predict production</p>", unsafe_allow_html=True)

    input_data = {}
    # Use sliders for prediction as before (yellow)
    for col in feature_cols:
        col_values = df[col].dropna() if col in df.columns else pd.Series([0.0])
        min_val, max_val, mean_val = float(col_values.min()), float(col_values.max()), float(col_values.mean())
        if min_val == max_val:
            max_val = min_val + 1.0
        # To avoid enormous sliders for very wide ranges, clamp sensible default step
        try:
            input_data[col] = st.slider(col, min_value=min_val, max_value=max_val, value=mean_val, key=f"pred_{col}")
        except Exception:
            # fallback to number_input if slider fails for column
            input_data[col] = st.number_input(col, min_value=min_val, max_value=max_val, value=mean_val, key=f"pred_num_{col}")

    if st.button("Predict Production"):
        with st.spinner("Predicting production..."):
            time.sleep(0.9)
            input_df = pd.DataFrame([input_data])
            # Ensure numeric_cols exist in input df before scaling
            available_numeric = [c for c in numeric_cols if c in input_df.columns]
            if available_numeric:
                input_df[available_numeric] = scaler.transform(input_df[available_numeric])
            pred = model.predict(input_df)[0]
            st.session_state.predicted_production = float(pred)
            st.markdown(f"<div class='glass-card'><h2 style='color:#ffd700; text-align:center; font-weight:bold;'>Predicted Production: {pred:.2f} MMcfge</h2></div>", unsafe_allow_html=True)

# ============================================
# PAGE 3: ECONOMIC ANALYSIS
# ============================================

elif page == "Economic Analysis":
    
    st.title("Economic Analysis")

    # -----------------------------
    # COST PARAMETERS (SLIDERS)
    # -----------------------------
    st.subheader("Adjust Cost Parameters")

    base_drilling_cost = st.slider("Base Drilling Cost ($/ft)", 500, 5000, 1000)
    base_completion_cost = st.slider("Base Completion Cost ($/ft)", 200, 2000, 500)

    proppant_cost_per_lb = st.slider("Proppant Cost ($/lb)", 0.01, 1.0, 0.10)
    water_cost_per_bbl = st.slider("Water Cost ($/bbl)", 0.5, 5.0, 1.5)
    additive_cost_per_bbl = st.slider("Additive Cost ($/bbl)", 0.5, 5.0, 2.0)

    base_maintenance_cost = st.slider("Maintenance Cost ($/year)", 10000, 100000, 30000)
    base_pump_cost = st.slider("Pump/Energy Cost ($/year)", 10000, 50000, 20000)

    gas_price = st.slider("Gas Price ($/MMcfge)", 1, 20, 5)

    # -----------------------------
    # CALCULATE REVENUE FOR EXISTING WELLS
    # -----------------------------
    df["Revenue"] = df["Production (MMcfge)"] * 1_000_000 * gas_price

    st.subheader("Revenue of Existing Wells")
    st.dataframe(df[['ID', 'Revenue']])

    # -----------------------------
    # REVENUE FOR PREDICTED WELL
    # -----------------------------
    if "predicted_production" in st.session_state:
        st.subheader("Revenue for Predicted Well")

        P = st.session_state.predicted_production
        new_revenue = P * 1_000_000 * gas_price
        
        st.write(f"Predicted Production: **{P:.2f} MMcfge**")
        st.write(f"Revenue: **${new_revenue:,.2f}**")
