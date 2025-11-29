# ============================================
# STREAMLIT RESERVOIR ENGINEERING APP
# DARK THEME, GLASSY SIDEBAR, YELLOW ACCENTS
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px
import time

# ============================================
# 1. PAGE CONFIG & BACKGROUND
# ============================================

st.set_page_config(page_title="Reservoir Engineering App", layout="wide")

# Background + global CSS
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
  background: linear-gradient(rgba(30,30,30,0.6), rgba(30,30,30,0.6)), 
              url("https://raw.githubusercontent.com/Maham-Waseem-123/colorappfinal/main/patrick-hendry-6xeDIZgoPaw-unsplash.jpg");
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
}
body, .block-container, .stText, .stMarkdown, .stDataFrame, .stButton, .stSlider {
    color: white !important;
    font-family: "Segoe UI", sans-serif;
}
h1,h2,h3,h4,h5 { 
    color: #ffd700 !important; 
    font-weight: bold !important; 
    text-shadow: 1px 1px 3px rgba(0,0,0,0.7);
}
.css-1d391kg { 
    background-color: rgba(255, 255, 255, 0.1) !important; 
    backdrop-filter: blur(15px); 
    color: white !important;
    transition: 0.3s;
}
.css-1d391kg:hover { background-color: rgba(255,255,255,0.2) !important; }

div.stButton > button {
    background-color: rgba(255, 255, 255, 0.2);
    color: white;
    font-weight: bold;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.3);
    transition: 0.3s;
}
div.stButton > button:hover { 
    background-color: rgba(255, 255, 255, 0.4); 
    transform: scale(1.05);
}

.stPlotlyChart { 
    background-color: rgba(0,0,0,0.35); 
    border-radius: 15px; 
    padding: 10px; 
}

.stDataFrame div.row_widget { 
    background-color: rgba(255, 255, 255, 0.1); 
    color: white; 
}

/* Sliders: bold white labels, yellow knob */
div.stSlider > label {
    color: white !important;
    font-weight: bold !important;
}
div.stSlider > div > div > div > div > div[role="slider"] {
    background-color: #ffd700 !important;
    border: 2px solid #ffd700 !important;
}
div.stSlider > div > div > div > div > div[role="slider"]:hover {
    box-shadow: 0 0 10px #ffd700;
}

/* Glass card for charts and outputs */
.glass-card {
    background-color: rgba(0,0,0,0.45);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.5);
    margin-bottom: 20px;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ============================================
# 2. LOAD DATA
# ============================================

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Maham-Waseem-123/Final_project/main/Shale_Test.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip().str.replace("\n", "").str.replace("\xa0", "")
    df = df.apply(pd.to_numeric, errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df

df = load_data()

# ============================================
# 3. TRAIN MODEL
# ============================================

@st.cache_resource
def train_model(df):
    target = "Production (MMcfge)"
    feature_cols = df.drop(columns=["ID", target]).columns.tolist()
    numeric_cols = feature_cols
    X = df[feature_cols].copy()
    y = df[target]

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
# 4. SIDEBAR NAVIGATION
# ============================================

st.sidebar.title("Pages")
page = st.sidebar.radio(
    "Select a Page:", 
    ["Reservoir Engineering Dashboard", "Reservoir Prediction", "Economic Analysis"],
    format_func=lambda x: "📊 " + x if x=="Reservoir Engineering Dashboard" else ("🛢️ "+x if x=="Reservoir Prediction" else "💰 "+x)
)

# ============================================
# 5. DASHBOARD PAGE
# ============================================

if page == "Reservoir Engineering Dashboard":
    st.markdown("<h1 style='text-align:center;'>Reservoir Engineering Dashboard</h1>", unsafe_allow_html=True)

    features_to_plot = [
        "Porosity",
        "Additive per foot (bbls)",
        "Water per foot (bbls)",
        "Proppant per foot (lbs)"
    ]

    col1, col2 = st.columns(2)

    for i, feature in enumerate(features_to_plot):
        target_col = col1 if i % 2 == 0 else col2
        df['bin'] = pd.cut(df[feature], bins=10, duplicates='drop')
        binned_df = df.groupby('bin', as_index=False)['Production (MMcfge)'].mean()
        binned_df['bin_center'] = binned_df['bin'].apply(lambda x: x.mid if x is not None else np.nan)
        binned_df = binned_df.dropna(subset=['bin_center', 'Production (MMcfge)'])
        binned_df = binned_df.sort_values('bin_center')

        fig = px.line(
            binned_df,
            x='bin_center',
            y='Production (MMcfge)',
            labels={'bin_center': feature, 'Production (MMcfge)': 'Production (MMcfge)'},
            markers=True
        )
        fig.update_traces(line=dict(color='yellow', width=3), marker=dict(color='yellow', size=8))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Segoe UI', size=12),
            xaxis=dict(showgrid=False, title=dict(text=feature, font=dict(color='white', size=14))),
            yaxis=dict(showgrid=False, title=dict(text='Production (MMcfge)', font=dict(color='white', size=14)))
        )
        target_col.markdown(f"<div class='glass-card'><h4>{feature} vs Production</h4></div>", unsafe_allow_html=True)
        target_col.plotly_chart(fig, use_container_width=True)

    # Depth chart full width
    st.markdown("<div class='glass-card'><h4>Depth (feet) vs Production</h4></div>", unsafe_allow_html=True)
    df['Depth_bin'] = pd.cut(df["Depth (feet)"], bins=10, duplicates='drop')
    binned_depth_df = df.groupby('Depth_bin', as_index=False)['Production (MMcfge)'].mean()
    binned_depth_df['bin_center'] = binned_depth_df['Depth_bin'].apply(lambda x: x.mid if x is not None else np.nan)
    binned_depth_df = binned_depth_df.dropna(subset=['bin_center', 'Production (MMcfge)'])
    binned_depth_df = binned_depth_df.sort_values('bin_center')

    fig_depth = px.line(
        binned_depth_df,
        x='bin_center',
        y='Production (MMcfge)',
        labels={'bin_center': 'Depth (feet)', 'Production (MMcfge)': 'Production (MMcfge)'},
        markers=True
    )
    fig_depth.update_traces(line=dict(color='yellow', width=4), marker=dict(color='yellow', size=10))
    fig_depth.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Segoe UI', size=12),
        xaxis=dict(showgrid=False, title=dict(text='Depth (feet)', font=dict(color='white', size=14))),
        yaxis=dict(showgrid=False, title=dict(text='Production (MMcfge)', font=dict(color='white', size=14)))
    )
    st.plotly_chart(fig_depth, use_container_width=True)

# ============================================
# 6. RESERVOIR PREDICTION PAGE
# ============================================

elif page == "Reservoir Prediction":
    st.markdown("<h1 style='text-align:center;'>Predict New Well Production</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:white; font-weight:bold;'>Adjust parameters and predict production</p>", unsafe_allow_html=True)

    input_data = {}
    for col in feature_cols:
        col_values = df[col].dropna()
        min_val, max_val, mean_val = float(col_values.min()), float(col_values.max()), float(col_values.mean())
        if min_val == max_val:
            max_val = min_val + 1.0
        input_data[col] = st.slider(col, min_val, max_val, mean_val, key=col)

    if st.button("Predict Production"):
        with st.spinner("Predicting production..."):
            time.sleep(1)
            input_df = pd.DataFrame([input_data])
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
            pred = model.predict(input_df)[0]
            st.session_state.predicted_production = pred
            st.markdown(f"<h2 style='color:#ffd700; text-align:center; font-weight:bold;'>Predicted Production: {pred:.2f} MMcfge</h2>", unsafe_allow_html=True)

# ============================================
# 7. ECONOMIC ANALYSIS PAGE
# ============================================

# ----------------------------------------------
# ECONOMIC ANALYSIS PAGE (FULL WORKING BLOCK)
# ----------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np

st.title("💰 Economic Analysis")

st.markdown("### 🛠️ Adjust Cost Parameters")

col1, col2 = st.columns(2)

# --------------------------
# COST INPUTS (SEGMENTED CONTROLS)
# --------------------------

with col1:
    base_drilling_cost = st.segmented_control(
        "Base Drilling Cost ($/ft)",
        options=[500, 1500, 3000, 5000],
        default=1500
    )

    base_completion_cost = st.segmented_control(
        "Base Completion Cost ($/ft)",
        options=[200, 800, 1500, 2000],
        default=800
    )

    proppant_cost = st.segmented_control(
        "Proppant Cost ($/lb)",
        options=[0.01, 0.25, 0.50, 1.00],
        default=0.25,
        format_func=lambda x: f"${x:.2f}"
    )

    water_cost = st.segmented_control(
        "Water Cost ($/bbl)",
        options=[0.50, 1.50, 3.00, 5.00],
        default=1.50,
        format_func=lambda x: f"${x:.2f}"
    )

with col2:
    additive_cost = st.segmented_control(
        "Additive Cost ($/bbl)",
        options=[0.50, 1.50, 3.00, 5.00],
        default=1.50,
        format_func=lambda x: f"${x:.2f}"
    )

    maintenance_cost = st.segmented_control(
        "Maintenance Cost ($/year)",
        options=[10000, 40000, 70000, 100000],
        default=40000
    )

    pump_energy_cost = st.segmented_control(
        "Pump/Energy Cost ($/year)",
        options=[10000, 25000, 40000, 50000],
        default=25000
    )

    gas_price = st.segmented_control(
        "Gas Price ($/MMcfge)",
        options=[1.00, 2.50, 4.00, 6.00],
        default=2.50,
        format_func=lambda x: f"${x:.2f}"
    )

st.divider()

# ----------------------------------------------
# USER WELL PARAMETERS
# ----------------------------------------------

st.markdown("### 🛢️ Well Parameters")

colA, colB = st.columns(2)

with colA:
    lateral_length = st.number_input("Lateral Length (ft)", min_value=1000, max_value=15000, value=5000)
    proppant_per_ft = st.number_input("Proppant per foot (lbs/ft)", min_value=200, max_value=3000, value=1200)
    water_per_ft = st.number_input("Water per foot (bbl/ft)", min_value=5, max_value=100, value=30)

with colB:
    additive_per_ft = st.number_input("Additive per foot (bbl/ft)", min_value=0, max_value=20, value=5)
    predicted_production = st.number_input("Predicted Gas Production (MMcfge)", min_value=0.0, value=3.5)

st.divider()

# ----------------------------------------------
# COST CALCULATIONS
# ----------------------------------------------

# Drilling + Completion
drilling_cost_total = base_drilling_cost * lateral_length
completion_cost_total = base_completion_cost * lateral_length

# Variable Costs
proppant_total = proppant_cost * proppant_per_ft * lateral_length
water_total = water_cost * water_per_ft * lateral_length
additive_total = additive_cost * additive_per_ft * lateral_length

# Annual fixed OPEX
annual_opex = maintenance_cost + pump_energy_cost

# Revenue
gross_revenue = predicted_production * gas_price * 1000  # Converted to $ (1 MMcfge * gas price * 1000)

# Total CAPEX
total_capex = drilling_cost_total + completion_cost_total + water_total + proppant_total + additive_total

# Profitability
net_cashflow = gross_revenue - annual_opex - total_capex

# Breakeven Price
breakeven_price = (annual_opex + total_capex) / max(predicted_production, 0.0001)

# ----------------------------------------------
# RESULTS DISPLAY
# ----------------------------------------------

st.markdown("## 📊 Economic Summary")

colR1, colR2, colR3 = st.columns(3)

colR1.metric("Total CAPEX", f"${total_capex:,.0f}")
colR2.metric("Annual OPEX", f"${annual_opex:,.0f}")
colR3.metric("Gross Revenue", f"${gross_revenue:,.0f}")

st.divider()

colR4, colR5 = st.columns(2)

colR4.metric("Net Cashflow", f"${net_cashflow:,.0f}",
             delta="Positive" if net_cashflow > 0 else "Negative",
             delta_color="normal")

colR5.metric("Breakeven Gas Price", f"${breakeven_price:.2f}/MMcfge")

st.success("Economic analysis calculation completed successfully.")

