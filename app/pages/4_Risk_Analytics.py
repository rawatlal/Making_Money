import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.backtest.performance import PerformanceAnalytics

st.set_page_config(page_title="Risk Analytics", layout="wide")
st.title("🛡️ Risk Analytics")

if "backtest_report" not in st.session_state:
    st.info("Run a backtest first on the Backtest Results page.")
    st.stop()

report = st.session_state["backtest_report"]
returns = report.returns_series
fundamentals = st.session_state.get("fundamentals", pd.DataFrame())

# --- Rolling Metrics ---
st.subheader("Rolling Metrics")

window = st.slider("Rolling Window (days)", 20, 120, 60)

rolling_vol = returns.rolling(window).std() * np.sqrt(252)
rolling_sharpe = (returns.rolling(window).mean() * 252) / (returns.rolling(window).std() * np.sqrt(252))

col1, col2 = st.columns(2)

with col1:
    fig = px.line(x=rolling_vol.index, y=rolling_vol.values, title=f"Rolling {window}-Day Volatility (Annualized)")
    fig.update_layout(xaxis_title="Date", yaxis_title="Volatility", yaxis_tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.line(x=rolling_sharpe.index, y=rolling_sharpe.values, title=f"Rolling {window}-Day Sharpe Ratio")
    fig.update_layout(xaxis_title="Date", yaxis_title="Sharpe Ratio")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

# --- Rolling Beta ---
if not report.benchmark_returns.empty:
    st.subheader("Rolling Beta vs Benchmark")
    aligned = pd.concat([returns, report.benchmark_returns], axis=1).dropna()
    if len(aligned) > window:
        aligned.columns = ["portfolio", "benchmark"]
        rolling_cov = aligned["portfolio"].rolling(window).cov(aligned["benchmark"])
        rolling_var = aligned["benchmark"].rolling(window).var()
        rolling_beta = rolling_cov / rolling_var

        fig = px.line(x=rolling_beta.index, y=rolling_beta.values, title=f"Rolling {window}-Day Beta")
        fig.update_layout(xaxis_title="Date", yaxis_title="Beta")
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

# --- Sector Exposure Over Time ---
st.markdown("---")
st.subheader("Sector Exposure Over Time")

weights_history = report.weights_history
if not weights_history.empty and "sector" in fundamentals.columns:
    sector_map = fundamentals["sector"].to_dict()
    sector_weights = {}

    for date, row in weights_history.iterrows():
        sector_w = {}
        for ticker, weight in row.items():
            if weight > 0:
                sector = sector_map.get(ticker, "Unknown")
                sector_w[sector] = sector_w.get(sector, 0) + weight
        sector_weights[date] = sector_w

    sector_df = pd.DataFrame(sector_weights).T.fillna(0)

    fig = px.area(sector_df, title="Sector Weights Over Time")
    fig.update_layout(xaxis_title="Date", yaxis_title="Weight", yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No sector data available for exposure analysis.")

# --- Concentration (HHI) ---
st.markdown("---")
st.subheader("Portfolio Concentration (HHI)")

if not weights_history.empty:
    hhi = (weights_history ** 2).sum(axis=1)
    fig = px.line(x=hhi.index, y=hhi.values, title="Herfindahl-Hirschman Index Over Time")
    fig.update_layout(xaxis_title="Date", yaxis_title="HHI")
    fig.add_hline(y=1.0 / 50, line_dash="dash", line_color="green", annotation_text="Equal Weight (50 stocks)")
    st.plotly_chart(fig, use_container_width=True)

# --- Return Distribution ---
st.markdown("---")
st.subheader("Daily Return Distribution")

fig = go.Figure()
fig.add_trace(go.Histogram(x=returns.values, nbinsx=100, name="Portfolio"))
if not report.benchmark_returns.empty:
    fig.add_trace(go.Histogram(x=report.benchmark_returns.values, nbinsx=100, name="Benchmark", opacity=0.5))
fig.update_layout(
    title="Daily Returns Distribution",
    xaxis_title="Daily Return",
    yaxis_title="Frequency",
    barmode="overlay",
    xaxis_tickformat=".1%",
)
st.plotly_chart(fig, use_container_width=True)

# --- Summary Table ---
st.markdown("---")
st.subheader("Key Risk Metrics")

stats = report.summary_stats
risk_data = {
    "Metric": ["Max Drawdown", "Annualized Volatility", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"],
    "Value": [
        f"{stats.get('max_drawdown', 0):.2%}",
        f"{stats.get('annualized_volatility', 0):.2%}",
        f"{stats.get('sharpe_ratio', 0):.2f}",
        f"{stats.get('sortino_ratio', 0):.2f}",
        f"{stats.get('calmar_ratio', 0):.2f}",
    ],
}
if "beta" in stats:
    risk_data["Metric"].extend(["Beta", "Alpha"])
    risk_data["Value"].extend([f"{stats['beta']:.2f}", f"{stats['alpha']:.2%}"])

st.table(pd.DataFrame(risk_data))
