import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.backtest.broker import SimulatedBroker
from src.backtest.engine import BacktestEngine
from src.backtest.performance import PerformanceAnalytics
from src.factors.composite import CompositeScorer
from src.factors.registry import get_factor
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.constructor import PortfolioConstructor
from src.portfolio.optimizer import PortfolioOptimizer
from src.portfolio.rebalancer import Rebalancer

st.set_page_config(page_title="Backtest Results", layout="wide")
st.title("📊 Backtest Results")

if "prices" not in st.session_state:
    st.info("No data loaded. Please download data on the Portfolio Builder page first.")
    st.stop()

prices = st.session_state["prices"]
fundamentals = st.session_state["fundamentals"]

# --- Backtest Settings ---
st.sidebar.header("Backtest Settings")
start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp("2022-01-01"))
end_date = st.sidebar.date_input("End Date", value=prices.index[-1])
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=1_000_000, step=100_000)
rebal_freq = st.sidebar.selectbox("Rebalance Frequency", ["monthly", "quarterly"])
opt_method = st.sidebar.selectbox("Optimizer", ["max_sharpe", "min_variance", "risk_parity"])

if st.button("Run Backtest", type="primary"):
    with st.spinner("Running backtest... This may take a moment."):
        # Build components
        factor_weights = [
            (get_factor("pe_ratio"), 0.20),
            (get_factor("pb_ratio"), 0.15),
            (get_factor("ev_ebitda"), 0.15),
            (get_factor("momentum_3m"), 0.15),
            (get_factor("momentum_6m"), 0.20),
            (get_factor("momentum_12m"), 0.15),
        ]
        scorer = CompositeScorer(factor_weights)
        optimizer = PortfolioOptimizer(method=opt_method)
        constraints = PortfolioConstraints(max_position_weight=0.05, max_holdings=50)
        constructor = PortfolioConstructor(scorer, optimizer, constraints, top_n=50)
        rebalancer = Rebalancer(frequency=rebal_freq)
        broker = SimulatedBroker()

        engine = BacktestEngine(
            constructor=constructor,
            rebalancer=rebalancer,
            broker=broker,
            initial_capital=initial_capital,
        )

        report = engine.run(prices, fundamentals, str(start_date), str(end_date))
        st.session_state["backtest_report"] = report

if "backtest_report" not in st.session_state:
    st.info("Configure settings and click 'Run Backtest' to see results.")
    st.stop()

report = st.session_state["backtest_report"]

# --- Summary Stats ---
st.markdown("---")
st.subheader("Summary Statistics")
stats = report.summary_stats

col1, col2, col3, col4 = st.columns(4)
col1.metric("CAGR", f"{stats.get('cagr', 0):.2%}")
col2.metric("Sharpe Ratio", f"{stats.get('sharpe_ratio', 0):.2f}")
col3.metric("Max Drawdown", f"{stats.get('max_drawdown', 0):.2%}")
col4.metric("Final Value", f"${report.final_value:,.0f}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Annualized Vol", f"{stats.get('annualized_volatility', 0):.2%}")
col2.metric("Sortino Ratio", f"{stats.get('sortino_ratio', 0):.2f}")
col3.metric("Calmar Ratio", f"{stats.get('calmar_ratio', 0):.2f}")
col4.metric("Total Return", f"{stats.get('total_return', 0):.2%}")

if "alpha" in stats:
    col1, col2, _, _ = st.columns(4)
    col1.metric("Alpha", f"{stats['alpha']:.2%}")
    col2.metric("Beta", f"{stats.get('beta', 0):.2f}")

# --- Equity Curve ---
st.markdown("---")
st.subheader("Equity Curve")

cumulative = (1 + report.returns_series).cumprod() * report.initial_capital
fig = go.Figure()
fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative.values, name="Portfolio", line=dict(color="blue")))

if not report.benchmark_returns.empty:
    bench_cum = (1 + report.benchmark_returns).cumprod() * report.initial_capital
    fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values, name="Benchmark (SPY)", line=dict(color="gray", dash="dash")))

fig.update_layout(title="Portfolio Value Over Time", xaxis_title="Date", yaxis_title="Value ($)", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# --- Drawdown ---
st.subheader("Drawdown")
perf = PerformanceAnalytics()
dd = perf.drawdown_series(report.returns_series)
fig = px.area(x=dd.index, y=dd.values, title="Underwater Chart")
fig.update_layout(xaxis_title="Date", yaxis_title="Drawdown", yaxis_tickformat=".0%")
fig.update_traces(line_color="red", fillcolor="rgba(255,0,0,0.2)")
st.plotly_chart(fig, use_container_width=True)

# --- Monthly Returns Heatmap ---
st.subheader("Monthly Returns")
monthly_table = perf.monthly_returns_table(report.returns_series)
if not monthly_table.empty:
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    monthly_table.columns = [month_names.get(c, c) for c in monthly_table.columns]
    fig = px.imshow(
        monthly_table.values * 100,
        x=monthly_table.columns.tolist(),
        y=monthly_table.index.tolist(),
        text_auto=".1f",
        color_continuous_scale="RdYlGn",
        title="Monthly Returns (%)",
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Turnover ---
if not report.turnover_history.empty:
    st.subheader("Portfolio Turnover")
    fig = px.bar(x=report.turnover_history.index, y=report.turnover_history.values, title="Turnover per Rebalance")
    fig.update_layout(xaxis_title="Date", yaxis_title="Turnover", yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

# --- Transaction Costs ---
if not report.transaction_costs.empty:
    st.subheader("Cumulative Transaction Costs")
    cum_costs = report.transaction_costs.cumsum()
    fig = px.line(x=cum_costs.index, y=cum_costs.values, title="Cumulative Transaction Costs ($)")
    fig.update_layout(xaxis_title="Date", yaxis_title="Costs ($)")
    st.plotly_chart(fig, use_container_width=True)
