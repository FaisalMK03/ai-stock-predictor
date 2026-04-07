"""
Saudi Stock Market Predictor (Tadawul)
=======================================
Production-grade stock price prediction app.
Models: Linear Regression + Random Forest Regressor.
Educational purposes only — not financial advice.
"""

import warnings
warnings.filterwarnings("ignore")

import logging
import time
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tadawul")

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
TOP_COMPANIES: dict[str, str] = {
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

LOGOS: dict[str, str] = {
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
# COLOR PALETTE — Bloomberg-inspired dark
# ─────────────────────────────────────────────
C = {
    "bg":        "#0d1117",
    "ax":        "#161b22",
    "grid":      "#21262d",
    "text":      "#8b949e",
    "actual":    "#58a6ff",
    "lr":        "#f78166",
    "rf":        "#a371f7",
    "forecast":  "#3fb950",
    "border":    "#30363d",
}

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d1117;
    color: #e6edf3;
}
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* Brand bar */
.brand-bar {
    display:flex; align-items:center; gap:16px;
    background:linear-gradient(90deg,#161b22 0%,#0d1117 100%);
    border:1px solid #21262d; border-radius:12px;
    padding:16px 24px; margin-bottom:24px;
}
.brand-logo {
    width:68px; height:68px; object-fit:contain;
    border-radius:10px; background:#fff; padding:6px; flex-shrink:0;
}
.brand-logo-placeholder {
    width:68px; height:68px; border-radius:10px; background:#21262d;
    display:flex; align-items:center; justify-content:center;
    font-size:2rem; flex-shrink:0;
}
.brand-name   { font-size:1.55rem; font-weight:700; color:#f0f6fc; line-height:1.2; }
.brand-ticker { font-size:0.85rem; color:#f78166; font-family:'Courier New',monospace;
                letter-spacing:0.04em; margin-top:3px; }
.brand-sub    { font-size:0.75rem; color:#8b949e; margin-top:2px; }

.page-title {
    font-size:0.9rem; color:#8b949e; font-weight:600;
    letter-spacing:0.1em; text-transform:uppercase; margin-bottom:6px;
}

/* KPI grid */
.kpi-grid {
    display:grid; grid-template-columns:repeat(5,1fr);
    gap:12px; margin:18px 0 6px 0;
}
.kpi-card {
    background:#161b22; border:1px solid #21262d;
    border-radius:10px; padding:16px 20px;
    position:relative; overflow:hidden;
}
.kpi-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background:var(--accent,#238636); border-radius:10px 10px 0 0;
}
.kpi-label  { font-size:0.7rem; color:#8b949e; text-transform:uppercase;
              letter-spacing:0.07em; margin-bottom:7px; }
.kpi-value  { font-size:1.45rem; font-weight:700; color:#f0f6fc; line-height:1; }
.kpi-delta  { font-size:0.8rem; margin-top:6px; font-weight:600; }
.kpi-delta.up      { color:#3fb950; }
.kpi-delta.down    { color:#f85149; }
.kpi-delta.neutral { color:#8b949e; }

/* Model comparison table */
.model-table {
    width:100%; border-collapse:collapse; font-size:0.82rem;
    margin:12px 0;
}
.model-table th {
    background:#21262d; color:#8b949e; text-transform:uppercase;
    letter-spacing:0.07em; font-size:0.68rem; padding:10px 14px;
    text-align:left; border-bottom:1px solid #30363d;
}
.model-table td {
    padding:10px 14px; border-bottom:1px solid #21262d; color:#e6edf3;
}
.model-table tr:hover td { background:#161b22; }
.badge {
    display:inline-block; padding:2px 8px; border-radius:4px;
    font-size:0.7rem; font-weight:700; letter-spacing:0.05em;
}
.badge-lr  { background:#2d1a14; color:#f78166; border:1px solid #f78166; }
.badge-rf  { background:#1e1428; color:#a371f7; border:1px solid #a371f7; }
.badge-win { background:#122b1a; color:#3fb950; border:1px solid #3fb950; }

.section-hdr {
    font-size:0.72rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.1em; color:#8b949e;
    border-bottom:1px solid #21262d;
    padding-bottom:7px; margin:28px 0 14px 0;
}
.disclaimer {
    font-size:0.74rem; color:#6e7681;
    border-top:1px solid #21262d;
    margin-top:32px; padding-top:14px; line-height:1.7;
}
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#0d1117; }
::-webkit-scrollbar-thumb { background:#30363d; border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA FETCHING WITH RETRY / FALLBACK LOGIC
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_data(tkr: str, start: str) -> tuple[pd.DataFrame, str]:
    """
    Download OHLCV data from Yahoo Finance with retry and fallback logic.

    Strategy:
      1. Try the exact start date supplied by the user.
      2. If that returns empty, retry with a shorter 2-year window.
      3. If still empty, try the '5y' period string.

    Returns
    -------
    (df, source_label) where source_label describes which attempt succeeded,
    or (empty_df, "") if all attempts fail.
    """
    attempts = [
        {"start": start,                                       "label": f"from {start}"},
        {"start": (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d"),
                                                               "label": "last 2 years (fallback)"},
        {"period": "5y",                                       "label": "5-year period (fallback)"},
    ]

    for attempt in attempts:
        try:
            log.info("Fetching %s — %s", tkr, attempt.get("label", ""))
            if "period" in attempt:
                df = yf.download(tkr, period=attempt["period"],
                                 progress=False, auto_adjust=True)
            else:
                df = yf.download(tkr, start=attempt["start"],
                                 progress=False, auto_adjust=True)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if not df.empty and len(df) > 60:
                log.info("Success: %d rows", len(df))
                return df, attempt["label"]

            log.warning("Attempt returned %d rows — retrying…", len(df))
            time.sleep(0.4)

        except Exception as exc:
            log.error("Fetch error: %s", exc)
            time.sleep(0.4)

    return pd.DataFrame(), ""


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a rich feature set from raw OHLCV data.

    Features
    --------
    - Close          : raw closing price
    - MA10, MA20, MA50: rolling means (momentum proxies)
    - EMA12, EMA26   : exponential MAs (trend)
    - MACD           : EMA12 - EMA26 (trend-change signal)
    - Return         : daily % change
    - Volatility10   : 10-day rolling std of returns (risk proxy)
    - RSI14          : Relative Strength Index (overbought/oversold)
    - Lag1, Lag2, Lag3: lagged close prices (autoregressive features)
    - Target         : next-day close (what we predict)
    """
    data = df[["Close"]].copy()
    data.columns = ["Close"]

    # Moving averages
    data["MA10"]  = data["Close"].rolling(10).mean()
    data["MA20"]  = data["Close"].rolling(20).mean()
    data["MA50"]  = data["Close"].rolling(50).mean()

    # Exponential MAs & MACD
    data["EMA12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA26"] = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"]  = data["EMA12"] - data["EMA26"]

    # Returns & volatility
    data["Return"]      = data["Close"].pct_change()
    data["Volatility10"] = data["Return"].rolling(10).std()

    # RSI (14-period)
    delta = data["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    data["RSI14"] = 100 - (100 / (1 + rs))

    # Lagged closes
    data["Lag1"] = data["Close"].shift(1)
    data["Lag2"] = data["Close"].shift(2)
    data["Lag3"] = data["Close"].shift(3)

    # Target: next-day close
    data["Target"] = data["Close"].shift(-1)

    data.dropna(inplace=True)
    return data


FEATURE_COLS = [
    "Close", "MA10", "MA20", "MA50",
    "EMA12", "EMA26", "MACD",
    "Return", "Volatility10", "RSI14",
    "Lag1", "Lag2", "Lag3",
]


# ─────────────────────────────────────────────
# MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    actual_close: np.ndarray) -> dict:
    """Compute RMSE, MAE, R², and directional accuracy."""
    rmse    = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae     = float(mean_absolute_error(y_true, y_pred))
    r2      = float(r2_score(y_true, y_pred))
    dir_acc = float(np.mean(
        ((y_true - actual_close) > 0) == ((y_pred - actual_close) > 0)
    ) * 100)
    return {"rmse": rmse, "mae": mae, "r2": r2, "dir_acc": dir_acc}


def train_models(data: pd.DataFrame) -> dict:
    """
    Train Linear Regression and Random Forest on an 80/20 train-test split.

    Returns a dict with everything needed for display and forecasting.
    """
    X = data[FEATURE_COLS].values
    y = data["Target"].values

    split   = int(len(X) * 0.80)
    X_train = X[:split];  X_test = X[split:]
    y_train = y[:split];  y_test = y[split:]
    actual_close_test = data["Close"].values[split:]

    results = {}

    # ── Linear Regression ──
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    results["lr"] = {
        "model":   lr,
        "y_pred":  lr_pred,
        "metrics": compute_metrics(y_test, lr_pred, actual_close_test),
    }

    # ── Random Forest ──
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["rf"] = {
        "model":   rf,
        "y_pred":  rf_pred,
        "metrics": compute_metrics(y_test, rf_pred, actual_close_test),
    }

    results["y_test"]            = y_test
    results["actual_close_test"] = actual_close_test
    results["split"]             = split
    results["dates_test"]        = data.index[split:]
    return results


# ─────────────────────────────────────────────
# PREDICTION UTILITIES
# ─────────────────────────────────────────────
def predict_next_day(model, data: pd.DataFrame) -> float:
    """Predict the next trading day's closing price from the latest row."""
    latest = data[FEATURE_COLS].iloc[-1].values.reshape(1, -1)
    return float(model.predict(latest)[0])


def iterative_forecast(model, data: pd.DataFrame, n_days: int) -> list[float]:
    """
    Roll a multi-step forecast by feeding each prediction back as the next input.

    MA/EMA features are approximated from a rolling buffer of predicted closes.
    Uncertainty compounds with horizon — treat as directional, not exact.
    """
    recent = list(data["Close"].values[-50:])
    row    = data[FEATURE_COLS].iloc[-1].copy()
    preds  = []

    for _ in range(n_days):
        pred = float(model.predict(row.values.reshape(1, -1))[0])
        preds.append(pred)
        recent.append(pred)

        returns_buf = [
            (recent[i] - recent[i - 1]) / recent[i - 1]
            for i in range(max(1, len(recent) - 14), len(recent))
            if recent[i - 1] != 0
        ]
        vol_buf = [
            (recent[i] - recent[i - 1]) / recent[i - 1]
            for i in range(max(1, len(recent) - 10), len(recent))
            if recent[i - 1] != 0
        ]
        ret   = (pred - row["Close"]) / row["Close"] if row["Close"] != 0 else 0
        ema12 = pred * (2 / 13) + row["EMA12"] * (11 / 13)
        ema26 = pred * (2 / 27) + row["EMA26"] * (25 / 27)

        gains   = [r for r in returns_buf if r > 0]
        losses  = [-r for r in returns_buf if r < 0]
        avg_g   = np.mean(gains)  if gains  else 1e-9
        avg_l   = np.mean(losses) if losses else 1e-9
        rsi     = 100 - (100 / (1 + avg_g / avg_l))

        row = pd.Series({
            "Close":        pred,
            "MA10":         np.mean(recent[-10:]),
            "MA20":         np.mean(recent[-20:]),
            "MA50":         np.mean(recent[-50:]),
            "EMA12":        ema12,
            "EMA26":        ema26,
            "MACD":         ema12 - ema26,
            "Return":       ret,
            "Volatility10": float(np.std(vol_buf)) if len(vol_buf) > 1 else 0.0,
            "RSI14":        rsi,
            "Lag1":         row["Close"],
            "Lag2":         row["Lag1"],
            "Lag3":         row["Lag2"],
        })

    return preds


# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────
def _style_axes(fig: plt.Figure, ax: plt.Axes) -> None:
    """Apply dark Bloomberg-style theme to a matplotlib figure/axes pair."""
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


def plot_actual_vs_predicted(
    dates_test: pd.DatetimeIndex,
    y_test: np.ndarray,
    lr_pred: np.ndarray,
    rf_pred: np.ndarray,
    company: str,
) -> plt.Figure:
    """
    Overlay actual vs. Linear Regression vs. Random Forest predictions
    on a single chart (test-set period).
    """
    fig, ax = plt.subplots(figsize=(13, 4))
    _style_axes(fig, ax)

    ax.plot(dates_test, y_test,   label="Actual Close",      color=C["actual"], lw=2,   zorder=4)
    ax.plot(dates_test, lr_pred,  label="Linear Regression", color=C["lr"],     lw=1.5,
            linestyle="--", alpha=0.85, zorder=3)
    ax.plot(dates_test, rf_pred,  label="Random Forest",     color=C["rf"],     lw=1.5,
            linestyle=":",  alpha=0.85, zorder=3)

    ax.fill_between(dates_test, y_test, lr_pred, alpha=0.05, color=C["lr"])
    ax.fill_between(dates_test, y_test, rf_pred, alpha=0.05, color=C["rf"])

    ax.set_title(f"{company}  ·  Actual vs Predicted — Test Set",
                 fontsize=11, fontweight="600", pad=12)
    ax.set_ylabel("Price (SAR)", fontsize=9)
    ax.legend(fontsize=9, framealpha=0, labelcolor="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout(pad=1.5)
    return fig


def plot_forecast(
    data: pd.DataFrame,
    lr_preds: list[float],
    rf_preds: list[float],
    company: str,
    n_days: int,
) -> plt.Figure:
    """Plot 90-day historical close plus dual-model n-day forecasts."""
    history   = data["Close"].iloc[-90:]
    last_date = history.index[-1]
    f_dates   = [last_date + timedelta(days=i + 1) for i in range(n_days)]
    fx        = [last_date] + f_dates
    last_val  = float(history.values[-1])

    fig, ax = plt.subplots(figsize=(13, 4))
    _style_axes(fig, ax)

    ax.plot(history.index, history.values, label="Historical Close",
            color=C["actual"], lw=2, zorder=3)
    ax.scatter([last_date], [last_val], color=C["actual"], zorder=6, s=55)

    ax.plot(fx, [last_val] + lr_preds, label=f"LR {n_days}-Day Forecast",
            color=C["lr"], lw=2, linestyle="--", marker="o", markersize=4, zorder=4)
    ax.plot(fx, [last_val] + rf_preds, label=f"RF {n_days}-Day Forecast",
            color=C["rf"], lw=2, linestyle=":", marker="s", markersize=4, zorder=4)

    ax.fill_between(fx,
                    [last_val] + lr_preds,
                    [last_val] + rf_preds,
                    alpha=0.08, color="#ffffff", label="Model uncertainty band")

    ax.axvspan(last_date, f_dates[-1], alpha=0.04, color=C["forecast"], zorder=1)
    ax.axvline(last_date, color=C["border"], lw=1, linestyle=":")
    ax.set_title(f"{company}  ·  {n_days}-Day Price Forecast",
                 fontsize=11, fontweight="600", pad=12)
    ax.set_ylabel("Price (SAR)", fontsize=9)
    ax.legend(fontsize=9, framealpha=0, labelcolor="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout(pad=1.5)
    return fig


def plot_residuals(
    dates_test: pd.DatetimeIndex,
    y_test: np.ndarray,
    lr_pred: np.ndarray,
    rf_pred: np.ndarray,
) -> plt.Figure:
    """Plot prediction residuals for both models over the test period."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.5), sharey=False)

    for ax, pred, label, color in zip(
        axes,
        [lr_pred, rf_pred],
        ["Linear Regression", "Random Forest"],
        [C["lr"], C["rf"]],
    ):
        _style_axes(fig, ax)
        residuals = y_test - pred
        ax.bar(dates_test, residuals, color=color, alpha=0.55, width=1.5, zorder=3)
        ax.axhline(0, color=C["border"], lw=1)
        ax.set_title(f"{label} — Residuals", fontsize=10, fontweight="600", pad=10)
        ax.set_ylabel("Error (SAR)", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.autofmt_xdate(rotation=30)

    fig.tight_layout(pad=1.5)
    return fig


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
def kpi_card(label: str, value: str, delta: str = "",
             dcls: str = "neutral", accent: str = "#238636") -> str:
    """Return HTML for a single KPI card."""
    d = f'<div class="kpi-delta {dcls}">{delta}</div>' if delta else ""
    return (
        f'<div class="kpi-card" style="--accent:{accent}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{d}'
        f'</div>'
    )


def brand_bar_html(company: str, ticker: str, logo_url: str) -> str:
    """Return HTML for the company brand bar."""
    if logo_url:
        logo = (
            f'<img class="brand-logo" src="{logo_url}" '
            f'onerror="this.style.display=\'none\';'
            f'this.nextElementSibling.style.display=\'flex\'" />'
            f'<div class="brand-logo-placeholder" style="display:none">🏢</div>'
        )
    else:
        logo = '<div class="brand-logo-placeholder">🏢</div>'

    return f"""
    <div class="brand-bar">
        {logo}
        <div>
            <div class="brand-name">{company}</div>
            <div class="brand-ticker">{ticker}</div>
            <div class="brand-sub">Tadawul — Saudi Exchange  ·  Prices in SAR</div>
        </div>
    </div>"""


def model_comparison_table(lr_m: dict, rf_m: dict) -> str:
    """Return an HTML table comparing LR vs RF metrics side-by-side."""
    lr_wins = sum([
        lr_m["rmse"]    < rf_m["rmse"],
        lr_m["mae"]     < rf_m["mae"],
        lr_m["r2"]      > rf_m["r2"],
        lr_m["dir_acc"] > rf_m["dir_acc"],
    ])
    rf_wins = 4 - lr_wins

    def cell(lr_val, rf_val, fmt, lower_better=True):
        lr_better = (lr_val < rf_val) if lower_better else (lr_val > rf_val)
        lr_cls = "badge-win" if lr_better else "badge-lr"
        rf_cls = "badge-win" if not lr_better else "badge-rf"
        return (
            f'<td><span class="badge {lr_cls}">{fmt.format(lr_val)}</span></td>'
            f'<td><span class="badge {rf_cls}">{fmt.format(rf_val)}</span></td>'
        )

    rows = [
        ("RMSE (SAR)",         cell(lr_m["rmse"],    rf_m["rmse"],    "{:.3f}", True)),
        ("MAE (SAR)",          cell(lr_m["mae"],     rf_m["mae"],     "{:.3f}", True)),
        ("R²",                 cell(lr_m["r2"],      rf_m["r2"],      "{:.4f}", False)),
        ("Direction Accuracy", cell(lr_m["dir_acc"], rf_m["dir_acc"], "{:.1f}%", False)),
    ]

    body = "".join(
        f"<tr><td>{name}</td>{cells}</tr>"
        for name, cells in rows
    )

    return f"""
    <table class="model-table">
        <thead>
            <tr>
                <th>Metric</th>
                <th><span class="badge badge-lr">Linear Regression</span></th>
                <th><span class="badge badge-rf">Random Forest</span></th>
            </tr>
        </thead>
        <tbody>
            {body}
            <tr>
                <td style="color:#8b949e;font-size:0.7rem">Wins</td>
                <td><span class="badge badge-lr">{lr_wins}/4</span></td>
                <td><span class="badge badge-rf">{rf_wins}/4</span></td>
            </tr>
        </tbody>
    </table>"""


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
    active_model = st.radio(
        "Primary forecast model",
        options=["Linear Regression", "Random Forest", "Both"],
        index=2,
        horizontal=False,
    )

    st.markdown("---")
    st.caption("Data · Yahoo Finance\nModels · LR + Random Forest\nTadawul · Saudi Exchange")


# ─────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="page-title">📈 Saudi Stock Market Predictor</div>',
            unsafe_allow_html=True)

st.markdown(
    brand_bar_html(selected_company, ticker, LOGOS.get(selected_company, "")),
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────
with st.spinner(f"Fetching market data for **{selected_company}** ({ticker}) …"):
    raw_df, data_source = fetch_data(ticker, str(start_date))

if raw_df.empty:
    st.error(
        f"❌ Could not retrieve data for **{ticker}** ({selected_company}) "
        "after multiple attempts. This ticker may be temporarily unavailable "
        "on Yahoo Finance. Please try a different company or check back later."
    )
    st.info(
        "💡 **Tip:** Tadawul tickers sometimes have limited coverage on free data "
        "providers. Saudi Aramco (2222.SR) and other large caps generally have the "
        "most reliable data."
    )
    st.stop()

if data_source and "fallback" in data_source:
    st.info(
        f"ℹ️ Data loaded via fallback ({data_source}). "
        "Your requested start date returned insufficient rows; "
        "results now cover the available history."
    )

# ─────────────────────────────────────────────
# FEATURE ENGINEERING & MODEL TRAINING
# ─────────────────────────────────────────────
data = build_features(raw_df)

if len(data) < 80:
    st.warning(
        "⚠️ Insufficient historical data after feature engineering. "
        "Select a different company or move the start date earlier."
    )
    st.stop()

with st.spinner("Training models …"):
    results      = train_models(data)
    lr_metrics   = results["lr"]["metrics"]
    rf_metrics   = results["rf"]["metrics"]
    lr_pred_test = results["lr"]["y_pred"]
    rf_pred_test = results["rf"]["y_pred"]
    y_test       = results["y_test"]
    dates_test   = results["dates_test"]

    lr_next = predict_next_day(results["lr"]["model"], data)
    rf_next = predict_next_day(results["rf"]["model"], data)
    lr_forecast = iterative_forecast(results["lr"]["model"], data, forecast_days)
    rf_forecast = iterative_forecast(results["rf"]["model"], data, forecast_days)

last_close  = float(data["Close"].iloc[-1])

# Best model by RMSE
best_model  = "lr" if lr_metrics["rmse"] < rf_metrics["rmse"] else "rf"
best_next   = lr_next if best_model == "lr" else rf_next
delta_pct   = (best_next - last_close) / last_close * 100
delta_cls   = "up"   if delta_pct > 0 else "down"
delta_arrow = "▲"    if delta_pct > 0 else "▼"


# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
st.markdown('<div class="section-hdr">Market Snapshot & Model Overview</div>',
            unsafe_allow_html=True)

data_rows   = len(raw_df)
train_rows  = results["split"]
test_rows   = len(y_test)

st.markdown(f"""
<div class="kpi-grid">
    {kpi_card("Last Close (SAR)",       f"{last_close:,.2f}",     accent="#58a6ff")}
    {kpi_card("Next-Day Forecast (SAR)",
              f"{best_next:,.2f}",
              f"{delta_arrow} {abs(delta_pct):.2f}% · best model",
              delta_cls,
              "#3fb950" if delta_pct > 0 else "#f85149")}
    {kpi_card("LR RMSE / MAE",
              f"{lr_metrics['rmse']:,.2f}",
              f"MAE {lr_metrics['mae']:,.2f}  ·  R² {lr_metrics['r2']:.3f}",
              "neutral", "#f78166")}
    {kpi_card("RF RMSE / MAE",
              f"{rf_metrics['rmse']:,.2f}",
              f"MAE {rf_metrics['mae']:,.2f}  ·  R² {rf_metrics['r2']:.3f}",
              "neutral", "#a371f7")}
    {kpi_card("Dataset",
              f"{data_rows:,} rows",
              f"Train {train_rows:,}  ·  Test {test_rows:,}",
              "neutral", "#238636")}
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL COMPARISON TABLE
# ─────────────────────────────────────────────
st.markdown('<div class="section-hdr">Model Comparison</div>', unsafe_allow_html=True)
st.markdown(model_comparison_table(lr_metrics, rf_metrics), unsafe_allow_html=True)
st.caption(
    "Green highlights indicate the better-performing model for each metric. "
    "RMSE and MAE are lower-better; R² and Direction Accuracy are higher-better."
)


# ─────────────────────────────────────────────
# ACTUAL vs PREDICTED CHART
# ─────────────────────────────────────────────
st.markdown('<div class="section-hdr">Actual vs Predicted — Test Set</div>',
            unsafe_allow_html=True)
st.pyplot(plot_actual_vs_predicted(
    dates_test, y_test, lr_pred_test, rf_pred_test, selected_company
))


# ─────────────────────────────────────────────
# RESIDUALS CHART
# ─────────────────────────────────────────────
with st.expander("📊 Residual Analysis — Test Set"):
    st.pyplot(plot_residuals(dates_test, y_test, lr_pred_test, rf_pred_test))
    st.caption(
        "Residuals = Actual − Predicted. Bars close to zero indicate accurate predictions. "
        "Systematic patterns (trends, clusters) suggest the model is missing structure in the data."
    )


# ─────────────────────────────────────────────
# FUTURE FORECAST CHART
# ─────────────────────────────────────────────
st.markdown(
    f'<div class="section-hdr">Future Price Forecast — Next {forecast_days} Trading Days</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Iterative forecast: each predicted price seeds the next step. "
    "The shaded band between LR and RF lines represents model disagreement — "
    "wider band → higher uncertainty. Treat longer horizons as directional guidance only."
)
st.pyplot(plot_forecast(
    data, lr_forecast, rf_forecast, selected_company, forecast_days
))

# ── Forecast table ──
f_dates = [data.index[-1] + timedelta(days=i + 1) for i in range(forecast_days)]
forecast_df = pd.DataFrame({
    "Day":          range(1, forecast_days + 1),
    "Date":         [d.strftime("%a, %d %b %Y") for d in f_dates],
    "LR Forecast":  [f"{p:,.2f}" for p in lr_forecast],
    "RF Forecast":  [f"{p:,.2f}" for p in rf_forecast],
    "LR Δ vs Last": [
        f"{'▲' if p > last_close else '▼'} {abs((p - last_close)/last_close*100):.2f}%"
        for p in lr_forecast
    ],
    "RF Δ vs Last": [
        f"{'▲' if p > last_close else '▼'} {abs((p - last_close)/last_close*100):.2f}%"
        for p in rf_forecast
    ],
})
st.dataframe(forecast_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE (RF only)
# ─────────────────────────────────────────────
with st.expander("🌲 Random Forest — Feature Importances"):
    rf_model   = results["rf"]["model"]
    importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=False)

    fig_fi, ax_fi = plt.subplots(figsize=(10, 3.5))
    _style_axes(fig_fi, ax_fi)
    ax_fi.barh(importances.index[::-1], importances.values[::-1],
               color=C["rf"], alpha=0.8)
    ax_fi.set_title("Feature Importance — Random Forest", fontsize=10,
                    fontweight="600", pad=10, color="#f0f6fc")
    ax_fi.set_xlabel("Importance", fontsize=9)
    fig_fi.tight_layout(pad=1.5)
    st.pyplot(fig_fi)
    st.caption(
        "Higher importance = the feature contributes more to split decisions in the Random Forest. "
        "Lagged prices (Lag1–3) and moving averages typically dominate in short-term price prediction."
    )


# ─────────────────────────────────────────────
# RAW FEATURE DATA
# ─────────────────────────────────────────────
with st.expander("🔬 Raw Feature Data (last 30 rows)"):
    display_cols = ["Close", "MA10", "MA20", "MA50", "EMA12", "EMA26",
                    "MACD", "Return", "Volatility10", "RSI14",
                    "Lag1", "Lag2", "Lag3", "Target"]
    st.dataframe(
        data[display_cols].tail(30).style.format("{:.4f}"),
        use_container_width=True,
    )


# ─────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Disclaimer:</strong> This application is strictly for
    <strong>educational and portfolio demonstration purposes only</strong>.
    All predictions are generated by simple machine-learning models (Linear Regression
    and Random Forest) trained on historical price data. Past performance is
    <em>not</em> indicative of future results. This tool does <strong>not</strong>
    constitute financial, investment, or trading advice. Always consult a qualified
    financial advisor before making any investment decisions.
    <br><br>
    Data provided by Yahoo Finance via <code>yfinance</code>. Models retrained on each
    session load. No user data is stored or transmitted.
</div>
""", unsafe_allow_html=True)
