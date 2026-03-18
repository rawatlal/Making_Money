import streamlit as st

st.set_page_config(
    page_title="Making Money - Factor Investing",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Making Money")
st.subheader("Quantitative Factor Investing Platform")

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Asset Universe", "Russell 1000")
with col2:
    st.metric("Factors", "Value + Momentum")
with col3:
    st.metric("Optimization", "Multi-Strategy")

st.markdown("---")

st.markdown("""
### How It Works

1. **Factor Explorer** — Analyze individual factor performance, distributions, and correlations
2. **Portfolio Builder** — Configure factor weights, optimization method, and constraints
3. **Backtest Results** — Run historical backtests with realistic transaction costs
4. **Risk Analytics** — Monitor drawdowns, sector exposure, and rolling risk metrics

### Getting Started

Use the sidebar to navigate between pages. Start with the **Factor Explorer** to understand
each factor's behavior, then build and backtest a portfolio.

### Factors Available

| Factor | Type | Signal |
|--------|------|--------|
| P/E Ratio | Value | Lower is better (cheap stocks) |
| P/B Ratio | Value | Lower is better |
| EV/EBITDA | Value | Lower is better |
| 3-Month Momentum | Momentum | Higher is better (trending up) |
| 6-Month Momentum | Momentum | Higher is better |
| 12-Month Momentum | Momentum | Higher is better |
""")
