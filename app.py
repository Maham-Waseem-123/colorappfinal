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
# ECONOMIC ANALYSIS PAGE
# ============================================
elif page == "Economic Analysis":
    st.markdown("<div class='glass-card'><h1 style='text-align:center;'>Economic Analysis</h1></div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'><h3 style='margin-bottom:8px;'>🛠️ Adjust Cost Parameters</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # --------------------------
    # COST INPUTS (SEGMENTED CONTROLS)
    # --------------------------
    with col1:
        # segmented_control may not exist on older Streamlit; we assume it's available
        try:
            base_drilling_cost = st.segmented_control(
                "Base Drilling Cost ($/ft)",
                options=[500, 1500, 3000, 5000],
                default=1500,
                key="seg_base_drill"
            )
            base_completion_cost = st.segmented_control(
                "Base Completion Cost ($/ft)",
                options=[200, 800, 1500, 2000],
                default=800,
                key="seg_base_comp"
            )
            proppant_cost = st.segmented_control(
                "Proppant Cost ($/lb)",
                options=[0.01, 0.25, 0.50, 1.00],
                default=0.25,
                format_func=lambda x: f"${x:.2f}",
                key="seg_proppant"
            )
            water_cost = st.segmented_control(
                "Water Cost ($/bbl)",
                options=[0.50, 1.50, 3.00, 5.00],
                default=1.50,
                format_func=lambda x: f"${x:.2f}",
                key="seg_water"
            )
        except Exception:
            # Fallback to number_input if segmented_control isn't available
            base_drilling_cost = st.number_input("Base Drilling Cost ($/ft)", min_value=500, max_value=5000, value=1500, step=50)
            base_completion_cost = st.number_input("Base Completion Cost ($/ft)", min_value=200, max_value=2000, value=800, step=50)
            proppant_cost = st.number_input("Proppant Cost ($/lb)", min_value=0.01, max_value=1.0, value=0.25, step=0.01, format="%.2f")
            water_cost = st.number_input("Water Cost ($/bbl)", min_value=0.5, max_value=5.0, value=1.5, step=0.1, format="%.2f")

    with col2:
        try:
            additive_cost = st.segmented_control(
                "Additive Cost ($/bbl)",
                options=[0.50, 1.50, 3.00, 5.00],
                default=1.50,
                format_func=lambda x: f"${x:.2f}",
                key="seg_add"
            )
            maintenance_cost = st.segmented_control(
                "Maintenance Cost ($/year)",
                options=[10000, 40000, 70000, 100000],
                default=40000,
                key="seg_maint"
            )
            pump_energy_cost = st.segmented_control(
                "Pump/Energy Cost ($/year)",
                options=[10000, 25000, 40000, 50000],
                default=25000,
                key="seg_pump"
            )
            gas_price = st.segmented_control(
                "Gas Price ($/MMcfge)",
                options=[1.00, 2.50, 4.00, 6.00],
                default=2.50,
                format_func=lambda x: f"${x:.2f}",
                key="seg_gas"
            )
        except Exception:
            additive_cost = st.number_input("Additive Cost ($/bbl)", min_value=0.5, max_value=5.0, value=1.5, step=0.1, format="%.2f")
            maintenance_cost = st.number_input("Maintenance Cost ($/year)", min_value=10000, max_value=100000, value=40000, step=1000)
            pump_energy_cost = st.number_input("Pump/Energy Cost ($/year)", min_value=10000, max_value=50000, value=25000, step=1000)
            gas_price = st.number_input("Gas Price ($/MMcfge)", min_value=0.01, max_value=100.0, value=2.5, step=0.1, format="%.2f")

    st.markdown("</div>", unsafe_allow_html=True)  # close Adjust Cost Parameters glass-card

    st.markdown("<div class='glass-card'><h3>🛢️ Well Parameters</h3>", unsafe_allow_html=True)
    # --------------------------
    # WELL PARAMETERS (NUMBER INPUTS styled as white)
    # --------------------------
    colA, colB = st.columns(2)
    with colA:
        lateral_length = st.number_input("Lateral Length (ft)", min_value=1000, max_value=15000, value=5003, step=1, key="lat_len")
        proppant_per_ft = st.number_input("Proppant per foot (lbs/ft)", min_value=200, max_value=3000, value=1200, step=1, key="prop_ft")
        water_per_ft = st.number_input("Water per foot (bbl/ft)", min_value=1, max_value=200, value=30, step=1, key="water_ft")
    with colB:
        additive_per_ft = st.number_input("Additive per foot (bbl/ft)", min_value=0, max_value=50, value=5, step=1, key="add_ft")
        # If prediction exists in session state use it as default in the number_input
        pred_default = st.session_state.get("predicted_production", 3.5)
        predicted_production = st.number_input("Predicted Gas Production (MMcfge)", min_value=0.0, value=float(pred_default), step=0.01, key="pred_gas")

    st.markdown("</div>", unsafe_allow_html=True)  # close Well Parameters glass-card

    st.divider()

    # ----------------------------------------------
    # COST CALCULATIONS
    # ----------------------------------------------
    # Convert segmented_control options which may come as strings to numeric if needed
    def to_float(v):
        try:
            return float(v)
        except Exception:
            return v

    base_drilling_cost = to_float(base_drilling_cost)
    base_completion_cost = to_float(base_completion_cost)
    proppant_cost = to_float(proppant_cost)
    water_cost = to_float(water_cost)
    additive_cost = to_float(additive_cost)
    maintenance_cost = to_float(maintenance_cost)
    pump_energy_cost = to_float(pump_energy_cost)
    gas_price = to_float(gas_price)
    predicted_production = float(predicted_production)

    # Drilling + Completion totals
    drilling_cost_total = base_drilling_cost * lateral_length
    completion_cost_total = base_completion_cost * lateral_length

    # Variable Costs (per-ft * length)
    proppant_total = proppant_cost * proppant_per_ft * lateral_length
    water_total = water_cost * water_per_ft * lateral_length
    additive_total = additive_cost * additive_per_ft * lateral_length

    # Annual fixed OPEX
    annual_opex = maintenance_cost + pump_energy_cost

    # Revenue -- be careful with units: keep consistent with earlier code
    # Here we follow previous approach: predicted_production (MMcfge) * gas_price ($/MMcfge) * 1 (no extra multiplier)
    gross_revenue = predicted_production * gas_price

    # Total CAPEX
    total_capex = drilling_cost_total + completion_cost_total + water_total + proppant_total + additive_total

    # Profitability
    net_cashflow = gross_revenue - annual_opex - total_capex

    # Breakeven Price (per MMcfge)
    breakeven_price = (annual_opex + total_capex) / max(predicted_production, 1e-6)

    # ----------------------------------------------
    # RESULTS DISPLAY (GLASS METRIC CARDS)
    # ----------------------------------------------
    st.markdown("<div class='glass-card'><h3 style='margin-bottom:6px;'>📊 Economic Summary</h3></div>", unsafe_allow_html=True)

    # three metrics row
    colR1, colR2, colR3 = st.columns(3)
    with colR1:
        st.markdown(f"<div class='glass-card'><h4 style='margin:0;'>Total CAPEX</h4><h2 style='margin:6px 0; color:white;'>${total_capex:,.0f}</h2></div>", unsafe_allow_html=True)
    with colR2:
        st.markdown(f"<div class='glass-card'><h4 style='margin:0;'>Annual OPEX</h4><h2 style='margin:6px 0; color:white;'>${annual_opex:,.0f}</h2></div>", unsafe_allow_html=True)
    with colR3:
        st.markdown(f"<div class='glass-card'><h4 style='margin:0;'>Gross Revenue</h4><h2 style='margin:6px 0; color:white;'>${gross_revenue:,.0f}</h2></div>", unsafe_allow_html=True)

    st.divider()

    colR4, colR5 = st.columns(2)
    with colR4:
        delta_label = "Positive" if net_cashflow > 0 else "Negative"
        delta_color = "normal"
        st.markdown(f"<div class='glass-card'><h4 style='margin:0;'>Net Cashflow</h4><h2 style='margin:6px 0; color:white;'>${net_cashflow:,.0f}</h2><div class='small-muted'>{delta_label}</div></div>", unsafe_allow_html=True)
    with colR5:
        st.markdown(f"<div class='glass-card'><h4 style='margin:0;'>Breakeven Gas Price</h4><h2 style='margin:6px 0; color:white;'>${breakeven_price:,.2f}/MMcfge</h2></div>", unsafe_allow_html=True)

    # Success box (styled)
    st.success("Economic analysis calculation completed successfully.")

    # Optionally show a table summary (existing wells) if data has those columns
    if set(["ID", "Production (MMcfge)"]).issubset(df.columns):
        df_summary = df.copy()
        # compute per-row CAPEX/OPEX/Revenue/Profit using selected costs (for demonstration)
        df_summary["CAPEX"] = base_drilling_cost * df_summary.get("Depth (feet)", lateral_length) + \
                              base_completion_cost * df_summary.get("Gross Perforated Interval (ft)", lateral_length) + \
                              proppant_cost * df_summary.get("Proppant per foot (lbs)", proppant_per_ft) * df_summary.get("Gross Perforated Interval (ft)", lateral_length) + \
                              water_cost * df_summary.get("Water per foot (bbls)", water_per_ft) * df_summary.get("Gross Perforated Interval (ft)", lateral_length) + \
                              additive_cost * df_summary.get("Additive per foot (bbls)", additive_per_ft) * df_summary.get("Gross Perforated Interval (ft)", lateral_length)

        df_summary["OPEX"] = annual_opex
        df_summary["Revenue"] = df_summary["Production (MMcfge)"] * gas_price
        df_summary["Profit"] = df_summary["Revenue"] - df_summary["CAPEX"] - df_summary["OPEX"]

        st.markdown("<div class='glass-card'><h4>Economic Metrics of Existing Wells</h4></div>", unsafe_allow_html=True)
        # show a subset for readability
        cols_to_show = ["ID", "CAPEX", "OPEX", "Revenue", "Profit"]
        for c in cols_to_show:
            if c not in df_summary.columns:
                df_summary[c] = np.nan
        st.dataframe(df_summary[cols_to_show].head(50))

# ============================================
# END APP
# ============================================
