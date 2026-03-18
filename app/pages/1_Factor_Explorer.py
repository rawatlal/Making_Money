import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.factors.registry import get_factor, list_factors

st.set_page_config(page_title="Factor Explorer", layout="wide")
st.title("🔍 Factor Explorer")

# Session state for data
if "prices" not in st.session_state:
    st.info("No data loaded. Please download data first using the Portfolio Builder page or run the data pipeline.")
    st.stop()

prices = st.session_state["prices"]
fundamentals = st.session_state.get("fundamentals", pd.DataFrame())

# Factor selection
factor_names = list_factors()
selected_factor = st.selectbox("Select Factor", factor_names)

# Date selection
dates = prices.index
date_idx = st.slider(
    "Select Date",
    min_value=0,
    max_value=len(dates) - 1,
    value=len(dates) - 1,
)
selected_date = dates[date_idx]
st.write(f"**Date:** {selected_date.date()}")

st.markdown("---")

# Compute factor values
factor = get_factor(selected_factor)
values = factor.compute(prices, fundamentals, selected_date)

if values.empty:
    st.warning("No factor values computed for this date/factor combination.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution")
    fig = px.histogram(values, nbins=50, title=f"{selected_factor} Distribution")
    fig.update_layout(xaxis_title=selected_factor, yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Top & Bottom 10")
    top = values.nlargest(10)
    bottom = values.nsmallest(10)

    tab1, tab2 = st.tabs(["Top 10", "Bottom 10"])
    with tab1:
        st.dataframe(top.to_frame(name=selected_factor).style.format("{:.2f}"))
    with tab2:
        st.dataframe(bottom.to_frame(name=selected_factor).style.format("{:.2f}"))

# Factor correlation matrix
st.markdown("---")
st.subheader("Factor Correlation Matrix")

corr_data = {}
for name in factor_names:
    f = get_factor(name)
    vals = f.compute(prices, fundamentals, selected_date)
    if not vals.empty:
        corr_data[name] = vals

if len(corr_data) > 1:
    corr_df = pd.DataFrame(corr_data).dropna()
    corr_matrix = corr_df.corr(method="spearman")
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Spearman Rank Correlation Between Factors",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Need at least 2 factors with data to show correlations.")
