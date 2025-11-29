# ============================================
# STREAMLIT RESERVOIR ENGINEERING APP (THEMED FIXED WITH WHITE TEXT & GLASS SIDEBAR)
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px

# ============================================
# BACKGROUND IMAGE & GLOBAL STYLING
# ============================================

page_bg_img = """
<style>
/* Full background image */
[data-testid="stAppViewContainer"] {
  background: url("https://raw.githubusercontent.com/Maham-Waseem-123/colorappfinal/main/patrick-hendry-6xeDIZgoPaw-unsplash.jpg");
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
}

/* Make all text white */
body, .block-container, .stText, .stMarkdown, .stDataFrame, .stButton, .stSlider {
    color: white !important;
    font-family: "Segoe UI", sans-serif;
}

/* Headings */
h1, h2, h3, h4, h5 {
    color: #ffffff !important;
}

/* Sidebar as glass/transparent */
.css-1d391kg { 
    background-color: rgba(0, 0, 0, 0.4) !important; 
    backdrop-filter: blur(10px); 
    color: white !important;
}

/* Buttons */
div.stButton > button {
    background-color: rgba(255, 255, 255, 0.2);
    color: white;
    font-weight: bold;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.3);
}
div.stButton > button:hover {
    background-color: rgba(255, 255, 255, 0.4);
}

/* Plotly chart background */
.stPlotlyChart { 
    background-color: rgba(0,0,0,0.3); 
    border-radius: 15px; 
    padding: 10px; 
}

/* Sliders text color */
div.stSlider > div { color: white; }

/* Dataframe text color */
.stDataFrame div.row_widget { background-color: rgba(255, 255, 255, 0.1); color: white; }
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ============================================
# 1. LOAD DATA
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
# 2. TRAIN MODEL
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
# 3. STREAMLIT PAGE CONFIG
# ============================================

st.set_page_config(page_title="Reservoir Engineering App", layout="wide")

# ============================================
# Sidebar Navigation
# ============================================

st.sidebar.title("Pages")
page = st.sidebar.radio("Select a Page:", [
    "Reservoir Engineering Dashboard",
    "Reservoir Prediction",
    "Economic Analysis"
])

# ============================================
# PAGE 1: RESERVOIR ENGINEERING DASHBOARD
# ============================================

if page == "Reservoir Engineering Dashboard":
    st.markdown("<h1 style='text-align:center;'>Reservoir Engineering Dashboard</h1>", unsafe_allow_html=True)

    features_to_plot = [
        "Porosity",
        "Additive per foot (bbls)",
        "Water per foot (bbls)",
        "Proppant per foot (lbs)"
    ]

    def make_binned_lineplot(xcol, bins=10):
        df['bin'] = pd.cut(df[xcol], bins=bins)
        binned_df = df.groupby('bin', as_index=False)['Production (MMcfge)'].mean()
        binned_df['bin_center'] = binned_df['bin'].apply(lambda x: x.mid)
        binned_df = binned_df.dropna(subset=['Production (MMcfge)'])
        binned_df = binned_df.sort_values("bin_center")

        fig = px.line(
            binned_df,
            x='bin_center',
            y='Production (MMcfge)',
            labels={'bin_center': xcol, 'Production (MMcfge)': 'Production (MMcfge)'},
            markers=True
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font=dict(color='#ffd700', size=20)
        )
        fig.update_xaxes(
            range=[binned_df['bin_center'].min(), binned_df['bin_center'].max()],
            dtick=(binned_df['bin_center'].max() - binned_df['bin_center'].min())/bins
        )
        fig.update_yaxes(title_text="Production (MMcfge)")
        st.subheader(f"Production vs {xcol}")
        st.plotly_chart(fig, use_container_width=True)

    for col in features_to_plot:
        make_binned_lineplot(col)

    # Depth vs Production
    st.subheader("Depth (feet) vs Production")
    df['Depth_bin'] = pd.cut(df["Depth (feet)"], bins=10)
    binned_depth_df = df.groupby('Depth_bin', as_index=False)['Production (MMcfge)'].mean()
    binned_depth_df['bin_center'] = binned_depth_df['Depth_bin'].apply(lambda x: x.mid)
    binned_depth_df = binned_depth_df.dropna(subset=['Production (MMcfge)'])
    binned_depth_df = binned_depth_df.sort_values("bin_center")

    fig = px.line(
        binned_depth_df,
        x='bin_center',
        y='Production (MMcfge)',
        labels={'bin_center': 'Depth (feet)', 'Production (MMcfge)': 'Production (MMcfge)'},
        markers=True
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='#ffd700', size=20)
    )
    fig.update_xaxes(
        range=[binned_depth_df['bin_center'].min(), binned_depth_df['bin_center'].max()],
        dtick=(binned_depth_df['bin_center'].max() - binned_depth_df['bin_center'].min())/10
    )
    fig.update_yaxes(title_text="Production (MMcfge)")
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# PAGE 2: RESERVOIR PREDICTION
# ============================================

elif page == "Reservoir Prediction":
    st.markdown("<h1 style='text-align:center;'>Predict New Well Production</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:white;'>Adjust parameters and predict production</p>", unsafe_allow_html=True)

    input_data = {}
    for col in feature_cols:
        col_values = df[col].dropna()
        min_val = float(col_values.min())
        max_val = float(col_values.max())
        mean_val = float(col_values.mean())
        if min_val == max_val:
            max_val = min_val + 1.0  # fix for slider crash
        input_data[col] = st.slider(col, min_val, max_val, mean_val, key=col)

    if st.button("Predict Production"):
        input_df = pd.DataFrame([input_data])
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
        pred = model.predict(input_df)[0]
        st.markdown(f"<h2 style='color:#eeff00; text-align:center;'>Predicted Production: {pred:.2f} MMcfge</h2>", unsafe_allow_html=True)
        st.session_state.predicted_production = pred

# ============================================
# PAGE 3: ECONOMIC ANALYSIS
# ============================================

elif page == "Economic Analysis":
    st.markdown("<h1 style='text-align:center;'>Economic Analysis</h1>", unsafe_allow_html=True)
    st.subheader("Adjust Cost Parameters")
    base_drilling_cost = st.slider("Base Drilling Cost ($/ft)", 500, 5000, 1000)
    base_completion_cost = st.slider("Base Completion Cost ($/ft)", 200, 2000, 500)
    proppant_cost_per_lb = st.slider("Proppant Cost ($/lb)", 0.01, 1.0, 0.1)
    water_cost_per_bbl = st.slider("Water Cost ($/bbl)", 0.5, 5.0, 1.5)
    additive_cost_per_bbl = st.slider("Additive Cost ($/bbl)", 0.5, 5.0, 2.0)
    base_maintenance_cost = st.slider("Maintenance Cost ($/year)", 10000, 100000, 30000)
    base_pump_cost = st.slider("Pump/Energy Cost ($/year)", 10000, 50000, 20000)
    gas_price = st.slider("Gas Price ($/MMcfge)", 1, 20, 5)

    df["CAPEX"] = (
        base_drilling_cost * df["Depth (feet)"] +
        base_completion_cost * df["Gross Perforated Interval (ft)"] +
        proppant_cost_per_lb * df["Proppant per foot (lbs)"] * df["Gross Perforated Interval (ft)"] +
        water_cost_per_bbl * df["Water per foot (bbls)"] * df["Gross Perforated Interval (ft)"] +
        additive_cost_per_bbl * df["Additive per foot (bbls)"] * df["Gross Perforated Interval (ft)"]
    )
    df["OPEX"] = base_maintenance_cost + base_pump_cost
    df["Revenue"] = df["Production (MMcfge)"] * gas_price
    df["Profit"] = df["Revenue"] - df["CAPEX"] - df["OPEX"]

    st.subheader("Economic Metrics of Existing Wells")
    st.dataframe(df[['ID', 'CAPEX', 'OPEX', 'Revenue', 'Profit']])

    if "predicted_production" in st.session_state:
        P = st.session_state.predicted_production
        new_capex = (
            base_drilling_cost * df["Depth (feet)"].mean() +
            base_completion_cost * df["Gross Perforated Interval (ft)"].mean() +
            proppant_cost_per_lb * df["Proppant per foot (lbs)"].mean() * df["Gross Perforated Interval (ft)"].mean() +
            water_cost_per_bbl * df["Water per foot (bbls)"].mean() * df["Gross Perforated Interval (ft)"].mean() +
            additive_cost_per_bbl * df["Additive per foot (bbls)"].mean() * df["Gross Perforated Interval (ft)"].mean()
        )
        new_opex = base_maintenance_cost + base_pump_cost
        new_revenue = P * gas_price
        new_profit = new_revenue - new_capex - new_opex

        st.markdown(f"""
        <div style='background-color:rgba(0,0,0,0.4); padding:15px; border-radius:10px; text-align:center;'>
            <h3 style='color:#ffd700;'>Predicted Production: {P:.2f} MMcfge</h3>
            <p>CAPEX: ${new_capex:,.2f}</p>
            <p>OPEX: ${new_opex:,.2f}</p>
            <p>Revenue: ${new_revenue:,.2f}</p>
            <p>Profit: ${new_profit:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)
