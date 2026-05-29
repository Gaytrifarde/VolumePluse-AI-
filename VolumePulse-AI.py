import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="VolumePulse AI",
    page_icon="📈",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* ============================= */
/* MAIN BACKGROUND */
/* ============================= */

.stApp {
    background: linear-gradient(to right, #0f172a, #1e293b);
    color: white;
}

/* ============================= */
/* SIDEBAR */
/* ============================= */

section[data-testid="stSidebar"] {
    background-color: white;
    border-right: 2px solid #e5e7eb;
    padding-top: 20px;
}

/* Sidebar text */

section[data-testid="stSidebar"] * {
    color: black !important;
}

/* ============================= */
/* INPUT BOXES */
/* ============================= */

/* Text Input */

.stTextInput input {
    background-color: #111827 !important;
    color: white !important;
    border: 2px solid #374151 !important;
    border-radius: 12px !important;
    padding: 10px !important;
}

/* Selectbox */

.stSelectbox div[data-baseweb="select"] {
    background-color: #111827 !important;
    border-radius: 12px !important;
    border: 2px solid #374151 !important;
}

/* Selected Text */

.stSelectbox div[data-baseweb="select"] span {
    color: white !important;
}

/* Dropdown menu */

div[role="listbox"] {
    background-color: #111827 !important;
    color: white !important;
}

/* ============================= */
/* TITLE */
/* ============================= */

.main-title {
    font-size: 42px;
    font-weight: bold;
    color: white;
}

.sub-title {
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 20px;
}

/* ============================= */
/* KPI CARDS */
/* ============================= */

.kpi-card {
    background: #374151;
    padding: 15px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid #4b5563;
    transition: 0.3s;
}

.kpi-card:hover {
    transform: translateY(-5px);
    box-shadow: 0px 4px 18px rgba(255,255,255,0.15);
}

.kpi-title {
    font-size: 16px;
    color: #d1d5db;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 28px;
    font-weight: bold;
    color: white;
}

/* ============================= */
/* TABLE */
/* ============================= */

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ============================= */
/* CHART TEXT */
/* ============================= */

.js-plotly-plot .plotly .gtitle {
    fill: white !important;
}

.js-plotly-plot .plotly .xtick text {
    fill: white !important;
}

.js-plotly-plot .plotly .ytick text {
    fill: white !important;
}

/* ============================= */
/* EXPANDER */
/* ============================= */

.streamlit-expanderHeader {
    color: white !important;
    background-color: #1f2937 !important;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================

st.markdown(
    '<div class="main-title">📈 VolumePulse AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Smart Stock Volume Spike Detection Dashboard</div>',
    unsafe_allow_html=True
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("⚙️ Dashboard Settings")

ticker = st.sidebar.text_input(
    "Stock Symbol",
    value="RELIANCE.NS"
)

period = st.sidebar.selectbox(
    "Select Period",
    ["1mo", "3mo", "6mo", "1y", "2y"]
)

interval = st.sidebar.selectbox(
    "Select Interval",
    ["1d", "1h"]
)

# =====================================================
# DOWNLOAD DATA
# =====================================================

with st.spinner("Fetching Live Market Data..."):

    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

# =====================================================
# CHECK DATA
# =====================================================

if data.empty:
    st.error("❌ Invalid stock symbol")
    st.stop()

# =====================================================
# FIX MULTI-INDEX COLUMNS
# =====================================================

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# =====================================================
# RESET INDEX
# =====================================================

data.reset_index(inplace=True)

# =====================================================
# FIX DATE COLUMN
# =====================================================

if "Datetime" in data.columns:
    data.rename(columns={"Datetime": "Date"}, inplace=True)

elif "index" in data.columns:
    data.rename(columns={"index": "Date"}, inplace=True)

if "Date" not in data.columns:
    data.rename(columns={data.columns[0]: "Date"}, inplace=True)

# =====================================================
# CLEAN DATA
# =====================================================

data.dropna(inplace=True)

# =====================================================
# FEATURE ENGINEERING
# =====================================================

data["Volume_MA"] = data["Volume"].rolling(5).mean()

data["Z_Score"] = (
    (data["Volume"] - data["Volume"].mean())
    / data["Volume"].std()
)

data["Z_Score"] = data["Z_Score"].fillna(0)

# =====================================================
# MACHINE LEARNING MODEL
# =====================================================

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

data["Anomaly"] = model.fit_predict(
    data[["Volume"]]
)

# =====================================================
# SPIKE DETECTION
# =====================================================

data["Spike"] = np.where(
    (data["Z_Score"] > 2) |
    (data["Anomaly"] == -1),
    1,
    0
)

# =====================================================
# KPI VALUES
# =====================================================

latest_volume = float(data["Volume"].iloc[-1])

avg_volume = float(data["Volume"].mean())

total_spikes = int(data["Spike"].sum())

latest_price = float(data["Close"].iloc[-1])

# =====================================================
# KPI CARDS
# =====================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">📊 Current Volume</div>
        <div class="kpi-value">{latest_volume:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">🚨 Total Spikes</div>
        <div class="kpi-value">{total_spikes}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">📈 Avg Volume</div>
        <div class="kpi-value">{avg_volume:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">💰 Current Price</div>
        <div class="kpi-value">₹{latest_price:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# CHART
# =====================================================

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.7, 0.3]
)

# =====================================================
# CANDLESTICK
# =====================================================

fig.add_trace(
    go.Candlestick(
        x=data["Date"],
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Price"
    ),
    row=1,
    col=1
)

# =====================================================
# VOLUME COLORS
# =====================================================

colors = [
    "#ef4444" if spike == 1 else "#3b82f6"
    for spike in data["Spike"]
]

# =====================================================
# VOLUME BARS
# =====================================================

fig.add_trace(
    go.Bar(
        x=data["Date"],
        y=data["Volume"],
        marker_color=colors,
        name="Volume"
    ),
    row=2,
    col=1
)

# =====================================================
# MOVING AVERAGE LINE
# =====================================================

fig.add_trace(
    go.Scatter(
        x=data["Date"],
        y=data["Volume_MA"],
        mode="lines",
        line=dict(
            color="yellow",
            width=2
        ),
        name="Volume MA"
    ),
    row=2,
    col=1
)

# =====================================================
# CHART LAYOUT
# =====================================================

fig.update_layout(
    template="plotly_dark",
    height=720,

    title={
        "text": f"{ticker} Live Volume Spike Analysis",
        "font": {
            "size": 24,
            "color": "white"
        }
    },

    xaxis_rangeslider_visible=False,
    hovermode="x unified",

    paper_bgcolor="#0f172a",
    plot_bgcolor="#111827",

    font=dict(
        color="white",
        size=14
    ),

    xaxis=dict(
        showgrid=False,
        color="white"
    ),

    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.1)",
        color="white"
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# DETECTED SPIKES
# =====================================================

st.subheader("🚨 Detected Volume Spikes")

spikes = data[data["Spike"] == 1][
    ["Date", "Close", "Volume", "Z_Score"]
]

if spikes.empty:

    st.success("✅ No unusual activity detected")

else:

    st.dataframe(
        spikes.sort_values(
            by="Z_Score",
            ascending=False
        ),
        use_container_width=True
    )

# =====================================================
# AI INSIGHTS
# =====================================================

st.subheader("🧠 AI Insights")

latest_spike = int(data["Spike"].iloc[-1])

latest_z = float(data["Z_Score"].iloc[-1])

if latest_spike == 1:

    st.error(f"""
🚨 High Trading Activity Detected

Current Z-Score: {latest_z:.2f}

Possible breakout or institutional activity.
""")

else:

    st.success(f"""
✅ Market Activity Normal

Current Z-Score: {latest_z:.2f}

Volume trend looks stable.
""")

# =====================================================
# RAW DATA
# =====================================================

with st.expander("📄 Show Raw Market Data"):
    st.dataframe(data, use_container_width=True)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "📈 Built using Streamlit • Plotly • Machine Learning • Live Market Data"
)
