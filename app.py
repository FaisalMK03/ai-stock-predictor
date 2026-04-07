"""
Saudi Stock Market Predictor (Tadawul)
=======================================
Production-grade stock price prediction app — Linear Regression.
Educational purposes only — not financial advice.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Tadawul Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# COMPANY REGISTRY
# ─────────────────────────────────────────────
TOP_COMPANIES = {
    "ACWA Power":                            "2082.SR",
    "Advanced Petrochemical":                "2330.SR",
    "Al Rajhi Bank":                         "1180.SR",
    "Alinma Bank":                           "1150.SR",
    "Almarai":                               "2280.SR",
    "Bupa Arabia":                           "8210.SR",
    "Elm Company":                           "7203.SR",
    "Jarir Marketing":                       "4190.SR",
    "Maaden (Mining Company)":               "1211.SR",
    "Mobily":                                "7020.SR",
    "Riyad Bank":                            "1010.SR",
    "SABIC":                                 "2010.SR",
    "Saudi Aramco":                          "2222.SR",
    "Saudi Electricity Company":             "5110.SR",
    "Saudi Investment Bank":                 "1030.SR",
    "Savola Group":                          "2050.SR",
    "STC (Saudi Telecom)":                   "7010.SR",
    "Sulaiman Al Habib Medical":             "4013.SR",
    "Yanbu National Petrochemical (Yansab)": "2290.SR",
}

LOGOS = {
    "ACWA Power":                            "https://logo.clearbit.com/acwapower.com",
    "Advanced Petrochemical":                "https://logo.clearbit.com/advanced-pc.com",
    "Al Rajhi Bank":                         "https://logo.clearbit.com/alrajhibank.com.sa",
    "Alinma Bank":                           "https://logo.clearbit.com/alinma.com",
    "Almarai":                               "https://logo.clearbit.com/almarai.com",
    "Bupa Arabia":                           "https://logo.clearbit.com/bupa.com.sa",
    "Elm Company":                           "https://logo.clearbit.com/elm.sa",
    "Jarir Marketing":                       "https://logo.clearbit.com/jarir.com",
    "Maaden (Mining Company)":               "https://logo.clearbit.com/maaden.com.sa",
    "Mobily":                                "https://logo.clearbit.com/mobily.com.sa",
    "Riyad Bank":                            "https://logo.clearbit.com/riyadbank.com",
    "SABIC":                                 "https://logo.clearbit.com/sabic.com",
    "Saudi Aramco":                          "https://logo.clearbit.com/aramco.com",
    "Saudi Electricity Company":             "https://logo.clearbit.com/se.com.sa",
    "Saudi Investment Bank":                 "https://logo.clearbit.com/saib.com.sa",
    "Savola Group":                          "https://logo.clearbit.com/savola.com",
    "STC (Saudi Telecom)":                   "https://logo.clearbit.com/stc.com.sa",
    "Sulaiman Al Habib Medical":             "https://logo.clearbit.com/hmg.com",
    "Yanbu National Petrochemical (Yansab)": "https://logo.clearbit.com/yansab.com",
}

# ─────────────────────────────────────────────
# GLOBAL STYLES — Bloomberg-inspired dark palette
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d1117;
    color: #e6edf3;
}
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* ── Brand bar ── */
.brand-bar {
    display: flex;
    align-items: center;
    gap: 16px;
    background: linear-gradient(90deg, #161b22 0%, #0d1117 100%);
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 24px;
}
.brand-logo {
    width: 68px;
    height: 68px;
    object-fit: contain;
    border-radius: 10px;
    background: #fff;
    padding: 6px;
    flex-shrink: 0;
}
.brand-logo-placeholder {
    width: 68px;
    height: 68px;
    border-radius: 10px;
    background: #21262d;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    flex-shrink: 0;
}
.brand-name    { font-size:1.55rem; font-weight:700; color:#f0f6fc; line-height:1.2; }
.brand-ticker  { font-size:0.85rem; color:#f78166; font-family:'Courier New',monospace;
                 letter-spacing:0.04em; margin-top:3px; }
.brand-sub     { font-size:0.75rem; color:#8b949e; margin-top:2px; }

/* ── Page subtitle ── */
.page-title {
    font-size:0.9rem; color:#8b949e; font-weight:600;
    letter-spacing:0.1em; text-transform:uppercase; margin-bottom:6px;
}

/* ── KPI grid ── */
.kpi-grid {
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:12px;
    margin:18px 0 6px 0;
}
.kpi-card {
    background:#161b22;
    border:1px solid #21262d;
    border-radius:10px;
    padding:16px 20px;
    position:relative;
    overflow:hidden;
}
.kpi-card::before {
    content:'';
    position:absolute; top:0; left:0; right:0; height:3px;
    background:var(--accent,#238636);
    border-radius:10px 10px 0 0;
}
.kpi-label  { font-size:0.7rem; color:#8b949e; text-transform:uppercase;
               letter-spacing:0.07em; margin-bottom:7px; }
.kpi-value  { font-size:1.45rem; font-weight:700; color:#f0f6fc; line-height:1; }
.kpi-delta  { font-size:0.8rem; margin-top:6px; font-weight:600; }
.kpi-delta.up      { color:#3fb950; }
.kpi-delta.down    { color:#f85149; }
.kpi-delta.neutral { color:#8b949e; }

/* ── Section headers ── */
.section-hdr {
    font-size:0.72rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.1em; color:#8b949e;
    border-bottom:1px solid #21262d;
    padding-bottom:7px; margin:28px 0 14px 0;
}

/* ── Disclaimer ── */
.disclaimer {
    font-size:0.74rem; color:#6e7681;
    border-top:1px solid #21262d;
    margin-top:32px; padding-top:14px; line-height:1.7;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#0d1117; }
::-webkit-scrollbar-thumb { background:#30363d; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🇸🇦 Tadawul Predictor")
    st.markdown("---")

    company_names    = sorted(TOP_COMPANIES.keys())
    selected_company = st.selectbox(
        "Select Company",
        options=company_names,
        index=company_names.index("Saudi Aramco"),
    )
    ticker = TOP_COMPANIES[selected_company]

    st.markdown("---")
    forecast_days = st.slider("Forecast horizon (days)", min_value=3, max_value=30, value=10)
    start_date    = st.date_input("Historical data from", value=datetime(2020, 1, 1))

    st.markdown("---")
    st.caption("Data · Yahoo Finance\nModel · Linear Regression\nTadawul · Saudi Exchange")

# ─────────────────────────────────────────────
# BRAND BAR
# ─────────────────────────────────────────────
st.markdown('<div class="page-title">📈 Saudi Stock Market Predictor</div>',
            unsafe_allow_html=True)

logo_url  = LOGOS.get(selected_company, "")
logo_html = (
    f'<img class="brand-logo" src="{logo_url}" '
    f'onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\'" />'
    f'<div class="brand-logo-placeholder" style="display:none">🏢</div>'
    if logo_url else
    '<div class="brand-logo-placeholder">🏢</div>'
)

st.markdown(f"""
<div class="brand-bar">
    {logo_html}
    <div>
        <div class="brand-name">{selected_company}</div>
        <div class="brand-ticker">{ticker}</div>
        <div class="brand-sub">Tadawul — Saudi Exchange &nbsp;·&nbsp; Prices in SAR</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_data(tkr: str, start: str) -> pd.DataFrame:
    """Download historical OHLCV data from Yahoo Finance."""
    return yf.download(tkr, start=start, progress=False, auto_adjust=True)

# ─────────────────────────────────────────────
# FEATURE ENGINEERING  ← unchanged
# ─────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df[["Close"]].copy()
    data.columns = ["Close"]
    data["MA10"]   = data["Close"].rolling(10).mean()
    data["MA50"]   = data["Close"].rolling(50).mean()
    data["Return"] = data["Close"].pct_change()
    data["Lag1"]   = data["Close"].shift(1)
    data["Target"] = data["Close"].shift(-1)
    data.dropna(inplace=True)
    return data

# ─────────────────────────────────────────────
# MODEL TRAINING & EVALUATION  ← unchanged
# ─────────────────────────────────────────────
def train_model(data: pd.DataFrame):
    feature_cols = ["Close", "MA10", "MA50", "Return", "Lag1"]
    X = data[feature_cols].values
    y = data["Target"].values
    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
    act_cl  = data["Close"].values[split:]
    dir_acc = np.mean(((y_test - act_cl) > 0) == ((y_pred - act_cl) > 0)) * 100
    return model, feature_cols, X_test, y_test, y_pred, rmse, dir_acc, split

# ─────────────────────────────────────────────
# NEXT-DAY PREDICTION  ← unchanged
# ─────────────────────────────────────────────
def predict_next_day(model, data: pd.DataFrame, feature_cols: list) -> float:
    latest = data[feature_cols].iloc[-1].values.reshape(1, -1)
    return float(model.predict(latest)[0])

# ─────────────────────────────────────────────
# MULTI-DAY ITERATIVE FORECAST  ← unchanged
# ─────────────────────────────────────────────
def iterative_forecast(model, data: pd.DataFrame,
                        feature_cols: list, n_days: int) -> list:
    recent_closes = list(data["Close"].values[-50:])
    last_row      = data[feature_cols].iloc[-1].copy()
    preds = []
    for _ in range(n_days):
        pred = float(model.predict(last_row.values.reshape(1, -1))[0])
        preds.append(pred)
        recent_closes.append(pred)
        last_row = pd.Series({
            "Close" : pred,
            "MA10"  : np.mean(recent_closes[-10:]),
            "MA50"  : np.mean(recent_closes[-50:]),
            "Return": (pred - last_row["Close"]) / last_row["Close"]
                       if last_row["Close"] != 0 else 0,
            "Lag1"  : last_row["Close"],
        })
    return preds

# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────
C = {
    "bg"       : "#0d1117",
    "ax"       : "#161b22",
    "grid"     : "#21262d",
    "text"     : "#8b949e",
    "actual"   : "#58a6ff",
    "predicted": "#f78166",
    "forecast" : "#3fb950",
    "border"   : "#30363d",
}

def _dark(fig, ax):
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["ax"])
    ax.tick_params(colors=C["text"], labelsize=8)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(C["border"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_color(C["text"])
    ax.yaxis.label.set_color(C["text"])
    ax.title.set_color("#f0f6fc")
    ax.grid(True, color=C["grid"], linewidth=0.7, linestyle="--")

def plot_actual_vs_predicted(dates_test, y_test, y_pred, company: str):
    fig, ax = plt.subplots(figsize=(13, 4))
    _dark(fig, ax)
    ax.plot(dates_test, y_test,
            label="Actual Close", color=C["actual"], linewidth=1.8, zorder=3)
    ax.plot(dates_test, y_pred,
            label="Predicted",    color=C["predicted"],
            linewidth=1.5, linestyle="--", alpha=0.9, zorder=2)
    ax.fill_between(dates_test, y_test, y_pred,
                    alpha=0.07, color=C["predicted"])
    ax.set_title(f"{company}  ·  Actual vs Predicted (Test Set)",
                 fontsize=11, fontweight="600", pad=12)
    ax.set_ylabel("Price (SAR)", fontsize=9)
    ax.legend(fontsize=9, framealpha=0, labelcolor="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout(pad=1.5)
    return fig

def plot_forecast(data: pd.DataFrame, forecast_prices: list,
                  company: str, n_days: int):
    history       = data["Close"].iloc[-90:]
    last_date     = history.index[-1]
    f_dates       = [last_date + timedelta(days=i + 1) for i in range(n_days)]
    forecast_x    = [last_date] + f_dates
    forecast_y    = [float(history.values[-1])] + forecast_prices

    fig, ax = plt.subplots(figsize=(13, 4))
    _dark(fig, ax)

    ax.plot(history.index, history.values,
            label="Historical Close", color=C["actual"], linewidth=2, zorder=3)
    ax.scatter([last_date], [history.values[-1]],
               color=C["actual"], zorder=6, s=55)
    ax.plot(forecast_x, forecast_y,
            label=f"{n_days}-Day Forecast", color=C["forecast"],
            linewidth=2, linestyle="--", marker="o", markersize=5, zorder=4)
    ax.axvspan(last_date, f_dates[-1], alpha=0.06, color=C["forecast"], zorder=1)
    ax.axvline(last_date, color=C["border"], linewidth=1, linestyle=":")

    ax.set_title(f"{company}  ·  {n_days}-Day Price Forecast",
                 fontsize=11, fontweight="600", pad=12)
    ax.set_ylabel("Price (SAR)", fontsize=9)
    ax.legend(fontsize=9, framealpha=0, labelcolor="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout(pad=1.5)
    return fig

# ─────────────────────────────────────────────
# RUN MODEL
# ─────────────────────────────────────────────
with st.spinner(f"Fetching market data for **{selected_company}** …"):
    raw_df = fetch_data(ticker, str(start_date))

if raw_df.empty:
    st.error(
        f"❌ No data found for **{ticker}** ({selected_company}). "
        "Try a different company or an earlier start date."
    )
    st.stop()

# Flatten MultiIndex columns if yfinance returns them
if isinstance(raw_df.columns, pd.MultiIndex):
    raw_df.columns = raw_df.columns.get_level_values(0)

data = build_features(raw_df)

if len(data) < 60:
    st.warning("⚠️ Insufficient historical data. "
               "Select a different company or move the start date earlier.")
    st.stop()

(model, feature_cols, X_test, y_test,
 y_pred, rmse, dir_acc, split) = train_model(data)

next_day_price  = predict_next_day(model, data, feature_cols)
forecast_prices = iterative_forecast(model, data, feature_cols, forecast_days)
dates_test      = data.index[split:]
last_close      = float(data["Close"].iloc[-1])
delta_pct       = ((next_day_price - last_close) / last_close) * 100
delta_cls       = "up" if delta_pct > 0 else "down"
delta_arrow     = "▲" if delta_pct > 0 else "▼"

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
st.markdown('<div class="section-hdr">Model Performance Overview</div>',
            unsafe_allow_html=True)

def kpi(label, value, delta="", dcls="neutral", accent="#238636"):
    d = f'<div class="kpi-delta {dcls}">{delta}</div>' if delta else ""
    return (f'<div class="kpi-card" style="--accent:{accent}">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>{d}</div>')

st.markdown(f"""
<div class="kpi-grid">
    {kpi("Last Close (SAR)",        f"{last_close:,.2f}",
         accent="#58a6ff")}
    {kpi("Next-Day Forecast (SAR)", f"{next_day_price:,.2f}",
         f"{delta_arrow} {abs(delta_pct):.2f}%", delta_cls,
         "#3fb950" if delta_pct > 0 else "#f85149")}
    {kpi("RMSE (SAR)",              f"{rmse:,.2f}",
         accent="#f78166")}
    {kpi("Direction Accuracy",      f"{dir_acc:.1f}%",
         "↑ Trend calls correct" if dir_acc >= 55 else "— Moderate accuracy",
         "up" if dir_acc >= 55 else "neutral",
         "#a371f7")}
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ACTUAL VS PREDICTED CHART
# ─────────────────────────────────────────────
st.markdown('<div class="section-hdr">Actual vs Predicted — Test Set</div>',
            unsafe_allow_html=True)
st.pyplot(plot_actual_vs_predicted(dates_test, y_test, y_pred, selected_company))

# ─────────────────────────────────────────────
# FUTURE FORECAST CHART
# ─────────────────────────────────────────────
st.markdown(
    f'<div class="section-hdr">Future Price Forecast — Next {forecast_days} Trading Days</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Iterative forecast: each predicted price seeds the next step. "
    "Uncertainty compounds — treat longer horizons as indicative direction only."
)
st.pyplot(plot_forecast(data, forecast_prices, selected_company, forecast_days))

# Forecast table
f_dates = [data.index[-1] + timedelta(days=i + 1) for i in range(forecast_days)]
forecast_df = pd.DataFrame({
    "Day":             range(1, forecast_days + 1),
    "Date":            [d.strftime("%a, %d %b %Y") for d in f_dates],
    "Forecast (SAR)":  [f"{p:,.2f}" for p in forecast_prices],
    "Δ vs Last Close": [
        f"{'▲' if p > last_close else '▼'} {abs((p - last_close)/last_close*100):.2f}%"
        for p in forecast_prices
    ],
})
st.dataframe(forecast_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# RAW FEATURE EXPANDER
# ─────────────────────────────────────────────
with st.expander("🔬 Raw Feature Data (last 30 rows)"):
    st.dataframe(
        data[["Close","MA10","MA50","Return","Lag1","Target"]].tail(30)
            .style.format("{:.4f}"),
        use_container_width=True,
    )

# ─────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Disclaimer:</strong> This application is strictly for
    <strong>educational and portfolio demonstration purposes only</strong>.
    All predictions are generated by a simple Linear Regression model trained on
    historical price data. They <strong>do not constitute financial advice</strong>,
    investment recommendations, or solicitations to buy or sell any security.
    Past performance is not indicative of future results.
    Always consult a licensed financial advisor before making investment decisions.
    &nbsp;·&nbsp; Data provided by Yahoo Finance via yfinance.
</div>
""", unsafe_allow_html=True)
