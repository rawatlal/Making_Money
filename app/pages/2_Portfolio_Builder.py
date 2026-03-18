import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.data.downloader import YFinanceDownloader
from src.data.cleaner import DataCleaner
from src.data.storage import DataStore
from src.universe.provider import Russell1000Provider
from src.factors.registry import get_factor
from src.factors.composite import CompositeScorer
from src.portfolio.optimizer import PortfolioOptimizer
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.constructor import PortfolioConstructor

st.set_page_config(page_title="Portfolio Builder", layout="wide")
st.title("🏗️ Portfolio Builder")

# --- Data Download Section ---
st.sidebar.header("Data Management")
if st.sidebar.button("Download / Refresh Data"):
    with st.spinner("Downloading data..."):
        provider = Russell1000Provider()
        tickers = provider.get_tickers()

        # Use a smaller subset for faster demo
        use_subset = st.sidebar.checkbox("Use subset (50 tickers)", value=True)
        if use_subset:
            tickers = tickers[:50]

        downloader = YFinanceDownloader(chunk_size=50)
        store = DataStore()

        prices_raw = downloader.download_prices(tickers, years=5)
        fundamentals = downloader.download_fundamentals(tickers)

        cleaner = DataCleaner()
        prices = cleaner.clean_prices(prices_raw)
        fundamentals = cleaner.clean_fundamentals(fundamentals)

        store.save_processed(prices, "clean_prices")
        store.save_fundamentals(fundamentals)

        st.session_state["prices"] = prices
        st.session_state["fundamentals"] = fundamentals
        st.success(f"Downloaded data for {len(prices.columns)} tickers")

# Load cached data if available
if "prices" not in st.session_state:
    store = DataStore()
    prices = store.load_processed("clean_prices")
    fundamentals = store.load_fundamentals()
    if prices is not None and fundamentals is not None:
        st.session_state["prices"] = prices
        st.session_state["fundamentals"] = fundamentals

if "prices" not in st.session_state:
    st.info("Click 'Download / Refresh Data' in the sidebar to get started.")
    st.stop()

prices = st.session_state["prices"]
fundamentals = st.session_state["fundamentals"]

st.markdown("---")

# --- Factor Weights ---
st.subheader("Factor Weights")
col1, col2, col3 = st.columns(3)
with col1:
    w_pe = st.slider("P/E Ratio", 0.0, 1.0, 0.20, 0.05)
    w_pb = st.slider("P/B Ratio", 0.0, 1.0, 0.15, 0.05)
with col2:
    w_ev = st.slider("EV/EBITDA", 0.0, 1.0, 0.15, 0.05)
    w_m3 = st.slider("Momentum 3M", 0.0, 1.0, 0.15, 0.05)
with col3:
    w_m6 = st.slider("Momentum 6M", 0.0, 1.0, 0.20, 0.05)
    w_m12 = st.slider("Momentum 12M", 0.0, 1.0, 0.15, 0.05)

total_weight = w_pe + w_pb + w_ev + w_m3 + w_m6 + w_m12
st.metric("Total Weight", f"{total_weight:.2f}", delta=f"{total_weight - 1.0:.2f}" if total_weight != 1.0 else None)

# --- Optimizer Settings ---
st.markdown("---")
st.subheader("Optimization Settings")
col1, col2 = st.columns(2)
with col1:
    opt_method = st.radio("Optimizer", ["max_sharpe", "min_variance", "risk_parity", "max_score"])
    top_n = st.number_input("Number of Holdings", min_value=5, max_value=200, value=50)
with col2:
    max_pos = st.slider("Max Position Weight", 0.01, 0.20, 0.05, 0.01)
    max_sector = st.slider("Max Sector Weight", 0.10, 0.50, 0.30, 0.05)

# --- Build Portfolio ---
st.markdown("---")
if st.button("Build Portfolio", type="primary"):
    with st.spinner("Constructing portfolio..."):
        # Normalize weights
        factor_weights = [
            (get_factor("pe_ratio"), w_pe / total_weight),
            (get_factor("pb_ratio"), w_pb / total_weight),
            (get_factor("ev_ebitda"), w_ev / total_weight),
            (get_factor("momentum_3m"), w_m3 / total_weight),
            (get_factor("momentum_6m"), w_m6 / total_weight),
            (get_factor("momentum_12m"), w_m12 / total_weight),
        ]
        factor_weights = [(f, w) for f, w in factor_weights if w > 0]

        scorer = CompositeScorer(factor_weights)
        optimizer = PortfolioOptimizer(method=opt_method)
        constraints = PortfolioConstraints(
            max_position_weight=max_pos,
            max_sector_weight=max_sector,
            max_holdings=top_n,
        )
        constructor = PortfolioConstructor(scorer, optimizer, constraints, top_n=top_n)

        date = prices.index[-1]
        weights = constructor.construct(prices, fundamentals, date)

        if weights:
            st.session_state["portfolio_weights"] = weights
            st.session_state["scorer"] = scorer
            st.session_state["optimizer"] = optimizer
            st.session_state["constraints"] = constraints

            # Display results
            st.subheader(f"Portfolio: {len(weights)} holdings")

            weights_df = pd.DataFrame.from_dict(weights, orient="index", columns=["Weight"])
            weights_df = weights_df.sort_values("Weight", ascending=False)

            if "sector" in fundamentals.columns:
                weights_df["Sector"] = weights_df.index.map(
                    lambda t: fundamentals.loc[t, "sector"] if t in fundamentals.index else "Unknown"
                )

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(weights_df.style.format({"Weight": "{:.3f}"}))
            with col2:
                fig = px.bar(weights_df.reset_index(), x="index", y="Weight", title="Portfolio Weights")
                fig.update_layout(xaxis_title="Ticker", xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

            if "Sector" in weights_df.columns:
                sector_weights = weights_df.groupby("Sector")["Weight"].sum()
                fig = px.pie(values=sector_weights.values, names=sector_weights.index, title="Sector Allocation")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Failed to construct portfolio. Check data availability.")
