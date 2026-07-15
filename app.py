"""
app.py
======
Streamlit dashboard for the Intelligent Stock Screener & Portfolio Builder
("Bloomberg Lite").

Run with:
    streamlit run app.py

Views
-----
1. Global Market Screen  - sortable/searchable table: cluster, anomaly, buy prob.
2. Single Ticker Deep-Dive - KPI cards, price/MA chart, peer table, SHAP chart.
3. Portfolio Builder      - MPT / Risk-Parity optimizer with pie chart + metrics.
"""

from __future__ import annotations

import logging
import traceback
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_pipeline import DataPipeline
from ml_engine import MLEngine
from portfolio import PortfolioOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# ----------------------------------------------------------------------------
# Page config & dark institutional theme
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Volatika | Intelligent Stock Screener and Portfolio Builder",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-card: #1c2128;
    --accent-amber: #f0b90b;
    --accent-green: #26a69a;
    --accent-red: #ef5350;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --border-color: #30363d;
}
.stApp { background-color: var(--bg-primary); color: var(--text-primary); }
section[data-testid="stSidebar"] { background-color: var(--bg-secondary); border-right: 1px solid var(--border-color); }
h1, h2, h3 { color: var(--accent-amber) !important; font-family: 'Consolas', monospace; letter-spacing: 0.5px; }
div[data-testid="stMetric"] {
    background-color: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 14px 16px;
}
div[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; }
div[data-testid="stMetricValue"] { color: var(--accent-amber) !important; font-family: 'Consolas', monospace; }
.stDataFrame { border: 1px solid var(--border-color); border-radius: 6px; }
.stButton>button {
    background-color: var(--accent-amber); color: #0d1117; font-weight: 700;
    border: none; border-radius: 4px;
}
.stButton>button:hover { background-color: #ffca28; color: #0d1117; }
hr { border-color: var(--border-color); }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

DEFAULT_UNIVERSES = {
    "NIFTY 50 (sample)": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
        "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    ],
    "S&P 500 (sample)": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM",
        "V", "UNH", "JNJ", "PG", "HD", "MA", "XOM",
    ],
    "Custom": [],
}

BENCHMARK_FOR_UNIVERSE = {
    "NIFTY 50 (sample)": "^NSEI",
    "S&P 500 (sample)": "^GSPC",
    "Custom": "^GSPC",
}


# ----------------------------------------------------------------------------
# Session state helpers
# ----------------------------------------------------------------------------

def _init_state():
    defaults = {
        "pipeline_result": None,
        "feature_table": None,
        "ml_engine": None,
        "pca_result": None,
        "cluster_result": None,
        "classifier_result": None,
        "risk_result": None,
        "anomaly_result": None,
        "training_panel": None,
        "last_run_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


# ----------------------------------------------------------------------------
# Core pipeline execution (cached per ticker-set to avoid re-downloading)
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _run_data_pipeline(tickers: tuple, benchmark: str):
    pipeline = DataPipeline(benchmark_ticker=benchmark)
    result = pipeline.run(list(tickers))
    training_panel = pipeline.build_training_panel(result.price_histories)
    return result, training_panel


def run_full_analysis(tickers: List[str], benchmark: str):
    with st.spinner("Downloading market data & engineering features..."):
        try:
            result, training_panel = _run_data_pipeline(tuple(tickers), benchmark)
        except Exception as exc:  # noqa: BLE001
            st.session_state.last_run_error = f"Data pipeline failed: {exc}"
            logger.error(traceback.format_exc())
            return

    if result.feature_table.empty:
        st.session_state.last_run_error = (
            "No tickers returned usable data. Check symbols and try again."
        )
        return

    engine = MLEngine()
    feature_table = result.feature_table
    step_errors: List[str] = []

    def _run_step(label: str, fn):
        """Run a single ML step in isolation so one failure can't skip
        the steps that come after it."""
        try:
            with st.spinner(label):
                fn()
        except Exception as exc:  # noqa: BLE001
            step_errors.append(f"{label} — {exc}")
            logger.error("%s failed:\n%s", label, traceback.format_exc())

    _run_step("Running PCA...", lambda: engine.run_pca(feature_table, n_components=5))
    _run_step("Running clustering...", lambda: engine.run_clustering(feature_table, n_clusters=4))

    if training_panel.empty:
        step_errors.append(
            "Classifier & risk model skipped — training panel is empty "
            "(need at least one ticker with a completed forward-return window; "
            "try a longer lookback or fewer/larger-cap tickers)."
        )
    else:
        _run_step(
            "Training outperformance classifier (XGBoost / LightGBM / RandomForest / "
            "LogisticRegression bake-off)...",
            lambda: engine.train_classifier(training_panel, feature_table),
        )
        _run_step(
            "Training downside-risk regressor...",
            lambda: engine.train_risk_model(training_panel, feature_table),
        )

    _run_step("Building peer-finder...", lambda: engine.build_peer_finder(feature_table))
    _run_step("Running anomaly detection...", lambda: engine.run_anomaly_detection(feature_table))

    st.session_state.pipeline_result = result
    st.session_state.feature_table = feature_table
    st.session_state.training_panel = training_panel
    st.session_state.ml_engine = engine

    if step_errors:
        st.session_state.last_run_error = (
            "Some steps failed and were skipped:\n- " + "\n- ".join(step_errors)
        )
    else:
        st.session_state.last_run_error = None


# ----------------------------------------------------------------------------
# Sidebar - universe selection & run control
# ----------------------------------------------------------------------------

st.sidebar.title("📊 Quantora")
st.sidebar.caption("Intelligent Stock Screener & Portfolio Builder")
st.sidebar.divider()

universe_choice = st.sidebar.selectbox("Ticker Universe", list(DEFAULT_UNIVERSES.keys()))

if universe_choice == "Custom":
    custom_input = st.sidebar.text_area(
        "Enter tickers (comma-separated)", value="AAPL, MSFT, GOOGL, AMZN, NVDA"
    )
    tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
else:
    tickers = DEFAULT_UNIVERSES[universe_choice]
    st.sidebar.caption(f"{len(tickers)} tickers loaded")

benchmark = st.sidebar.text_input(
    "Benchmark index", value=BENCHMARK_FOR_UNIVERSE.get(universe_choice, "^GSPC")
)

run_clicked = st.sidebar.button("🚀 Run Full Analysis", use_container_width=True)

if run_clicked:
    if not tickers:
        st.sidebar.error("Please provide at least one ticker.")
    else:
        run_full_analysis(tickers, benchmark)

if st.session_state.last_run_error:
    st.sidebar.error(st.session_state.last_run_error)

if st.session_state.pipeline_result is not None:
    failed = st.session_state.pipeline_result.failed_tickers
    if failed:
        st.sidebar.warning(f"{len(failed)} ticker(s) failed to load: {', '.join(failed)}")

st.sidebar.divider()
st.sidebar.caption(
    "⚠️ Educational tool only. Not investment advice. Data via Yahoo Finance "
    "may be delayed or incomplete."
)

# ----------------------------------------------------------------------------
# Main content
# ----------------------------------------------------------------------------

st.title("📈 Intelligent Stock Screener & Portfolio Builder")
st.caption("Quantora — screening, ML analytics, and portfolio construction in one workspace")

if st.session_state.feature_table is None:
    st.info(
        "👈 Select a ticker universe in the sidebar and click **Run Full Analysis** "
        "to fetch data, engineer features, and train the ML models."
    )
    st.stop()

feature_table: pd.DataFrame = st.session_state.feature_table
engine: MLEngine = st.session_state.ml_engine
pipeline_result = st.session_state.pipeline_result

tab_screen, tab_deep_dive, tab_portfolio = st.tabs(
    ["🌐 Global Market Screen", "🔍 Single Ticker Deep-Dive", "🧮 Portfolio Builder"]
)

# ============================================================================
# TAB 1 — Global Market Screen
# ============================================================================

with tab_screen:
    st.subheader("Global Market Screen")

    display_df = feature_table.copy()

    if engine.cluster_result is not None:
        display_df["Cluster"] = engine.cluster_result.labels.map(
            engine.cluster_result.archetype_names
        )
        # Per-stock tags computed universe-wide (independent of which
        # cluster a stock landed in) — surfaces e.g. a high-yield stock
        # that got clustered as "Growth" but still screens as a dividend
        # candidate on its own percentile ranking.
        tags_series = engine.cluster_result.stock_tags.get("tags")
        display_df["Also Screens As"] = (
            tags_series.reindex(display_df.index) if tags_series is not None else "N/A"
        )
    else:
        display_df["Cluster"] = "N/A"
        display_df["Also Screens As"] = "N/A"

    if engine.anomaly_result is not None:
        display_df["Anomaly"] = engine.anomaly_result.is_anomaly.map(
            {True: "⚠️ Flagged", False: "Normal"}
        )
    else:
        display_df["Anomaly"] = "N/A"

    if engine.classifier_result is not None:
        display_df["Buy Probability"] = engine.classifier_result.probabilities
    else:
        display_df["Buy Probability"] = np.nan

    if engine.risk_result is not None:
        display_df["Expected Downside Risk"] = engine.risk_result.predictions
    else:
        display_df["Expected Downside Risk"] = np.nan

    search = st.text_input("🔎 Search ticker", "")
    view_cols = [
        "Cluster", "Also Screens As", "Anomaly", "Buy Probability", "Expected Downside Risk",
        "last_price", "pe_ratio", "pb_ratio", "roe", "beta", "dividend_yield",
    ]
    view_cols = [c for c in view_cols if c in display_df.columns]
    filtered = display_df[view_cols]
    if search:
        filtered = filtered[filtered.index.str.contains(search.upper())]

    st.dataframe(
        filtered.sort_values("Buy Probability", ascending=False)
        if "Buy Probability" in filtered.columns else filtered,
        use_container_width=True,
        height=480,
        column_config={
            "Buy Probability": st.column_config.ProgressColumn(
                "Buy Probability", min_value=0, max_value=1, format="%.2f"
            ),
        },
    )

    if engine.classifier_result is not None and not np.isnan(engine.classifier_result.mean_auc):
        cr = engine.classifier_result
        model_label = cr.model_name.replace("_", " ").title()
        st.caption(
            f"Outperformance classifier — winner: **{model_label}** — "
            f"mean time-series-CV AUC: **{cr.mean_auc:.3f}** "
            f"({len(cr.auc_scores)} folds)"
        )
        if cr.candidate_scores:
            with st.expander("Model bake-off — AUC by candidate"):
                bake_off_df = pd.DataFrame(
                    {
                        "model": [k.replace("_", " ").title() for k in cr.candidate_scores],
                        "mean_cv_auc": list(cr.candidate_scores.values()),
                    }
                ).sort_values("mean_cv_auc", ascending=False)
                st.dataframe(
                    bake_off_df.style.format({"mean_cv_auc": "{:.3f}"}),
                    use_container_width=True,
                    hide_index=True,
                )

    if engine.pca_result is not None:
        with st.expander("PCA — Explained Variance"):
            var_df = pd.DataFrame({
                "component": [f"PC{i+1}" for i in range(len(engine.pca_result.explained_variance_ratio))],
                "explained_variance": engine.pca_result.explained_variance_ratio,
                "cumulative": engine.pca_result.cumulative_variance,
            })
            st.bar_chart(var_df.set_index("component")["explained_variance"])
            st.caption(f"Cumulative variance explained: {engine.pca_result.cumulative_variance[-1]*100:.1f}%")

# ============================================================================
# TAB 2 — Single Ticker Deep-Dive
# ============================================================================

with tab_deep_dive:
    st.subheader("Single Ticker Deep-Dive")

    ticker_options = list(feature_table.index)
    selected_ticker = st.selectbox("Select ticker", ticker_options)

    if selected_ticker:
        row = feature_table.loc[selected_ticker]
        price_history = pipeline_result.price_histories.get(selected_ticker, pd.DataFrame())

        buy_prob = (
            engine.classifier_result.probabilities.get(selected_ticker, np.nan)
            if engine.classifier_result is not None else np.nan
        )
        is_anomaly = (
            engine.anomaly_result.is_anomaly.get(selected_ticker, False)
            if engine.anomaly_result is not None else False
        )

        kpi_cols = st.columns(6)
        kpi_cols[0].metric("Price", f"{row.get('last_price', float('nan')):.2f}")
        kpi_cols[1].metric("P/E", f"{row.get('pe_ratio', float('nan')):.2f}")
        kpi_cols[2].metric("ROE", f"{row.get('roe', float('nan')):.2%}" if pd.notna(row.get('roe')) else "N/A")
        kpi_cols[3].metric("Beta", f"{row.get('beta', float('nan')):.2f}")
        kpi_cols[4].metric("Buy Probability", f"{buy_prob:.2%}" if pd.notna(buy_prob) else "N/A")
        kpi_cols[5].metric("Anomaly", "⚠️ Yes" if is_anomaly else "No")

        cluster_label = (
            engine.cluster_result.archetype_names.get(engine.cluster_result.labels.get(selected_ticker), "N/A")
            if engine.cluster_result is not None else "N/A"
        )
        st.markdown(f"**Cluster:** {cluster_label}")

        if engine.cluster_result is not None:
            tags_series = engine.cluster_result.stock_tags.get("tags")
            ticker_tags = tags_series.get(selected_ticker, "—") if tags_series is not None else "—"
            if ticker_tags and ticker_tags != "—":
                st.markdown(
                    f"**Also screens as:** {ticker_tags} "
                    f"_(top-percentile on this axis vs. the whole universe, "
                    f"independent of its assigned cluster)_"
                )

        st.markdown("#### Price vs Moving Averages")
        if not price_history.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=price_history.index, y=price_history["Close"], name="Close", line=dict(color="#f0b90b")))
            for ma_col, color in [("ma_20", "#26a69a"), ("ma_50", "#42a5f5"), ("ma_200", "#ef5350")]:
                if ma_col in price_history.columns:
                    fig.add_trace(go.Scatter(
                        x=price_history.index, y=price_history[ma_col],
                        name=ma_col.upper().replace("_", " "), line=dict(width=1.2, color=color),
                    ))
            fig.update_layout(
                template="plotly_dark", height=420, margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                legend=dict(orientation="h", y=1.05),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No price history available for this ticker.")

        st.markdown("#### Peer Analysis (Top 5 Similar Stocks)")
        try:
            peers = engine.find_peers(selected_ticker, top_n=5)
            if peers:
                peer_df = pd.DataFrame(peers, columns=["Ticker", "Distance (lower = more similar)"])
                st.dataframe(peer_df, use_container_width=True, hide_index=True)
            else:
                st.info("No peers found (universe may be too small).")
        except Exception as exc:  # noqa: BLE001
            st.info(f"Peer analysis unavailable: {exc}")

        st.markdown("#### Explainable AI — Why this prediction?")
        if engine.classifier_result is not None:
            try:
                contributions = engine.explain_ticker(selected_ticker, feature_table)
                top_contrib = contributions.head(10)
                fig_shap = go.Figure(go.Bar(
                    x=top_contrib.values,
                    y=top_contrib.index,
                    orientation="h",
                    marker_color=["#26a69a" if v > 0 else "#ef5350" for v in top_contrib.values],
                ))
                fig_shap.update_layout(
                    template="plotly_dark", height=380, margin=dict(l=10, r=10, t=20, b=10),
                    paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                    xaxis_title="SHAP value (impact on outperformance probability)",
                )
                st.plotly_chart(fig_shap, use_container_width=True)
                st.caption(f"Explaining the **{engine.classifier_result.model_name.replace('_', ' ').title()}** model's prediction.")
            except Exception as exc:  # noqa: BLE001
                st.info(f"SHAP explanation unavailable: {exc}")
        else:
            st.info("Train the classifier (Run Full Analysis) to see explanations.")

# ============================================================================
# TAB 3 — Portfolio Builder
# ============================================================================

with tab_portfolio:
    st.subheader("Portfolio Builder")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        selection_mode = st.radio(
            "Ticker selection", ["Auto-select top 5 by Buy Probability", "Custom subset"],
            horizontal=True,
        )
        if selection_mode == "Custom subset":
            portfolio_tickers = st.multiselect(
                "Choose tickers", list(feature_table.index), default=list(feature_table.index[:5])
            )
        else:
            if engine.classifier_result is not None:
                portfolio_tickers = PortfolioOptimizer.auto_select_top_n(
                    engine.classifier_result.probabilities, top_n=5
                )
                st.write("Auto-selected:", ", ".join(portfolio_tickers))
            else:
                portfolio_tickers = list(feature_table.index[:5])
                st.info("Classifier not trained — defaulting to first 5 tickers.")

    with col_b:
        method = st.selectbox("Optimization method", ["Maximum Sharpe Ratio", "Risk Parity"])
        risk_free = st.number_input("Risk-free rate", value=0.06, step=0.005, format="%.3f")

    if st.button("⚙️ Generate Optimized Portfolio"):
        if len(portfolio_tickers) < 2:
            st.error("Select at least 2 tickers to build a portfolio.")
        else:
            optimizer = PortfolioOptimizer(risk_free_rate=risk_free)
            try:
                if method == "Maximum Sharpe Ratio":
                    port_result = optimizer.optimize_max_sharpe(
                        pipeline_result.price_histories, portfolio_tickers
                    )
                else:
                    port_result = optimizer.optimize_risk_parity(
                        pipeline_result.price_histories, portfolio_tickers
                    )

                st.session_state["portfolio_result"] = port_result
            except Exception as exc:  # noqa: BLE001
                st.error(f"Portfolio optimization failed: {exc}")
                logger.error(traceback.format_exc())

    port_result = st.session_state.get("portfolio_result")
    if port_result is not None:
        m1, m2, m3 = st.columns(3)
        m1.metric("Expected Return (annualized)", f"{port_result.expected_return:.2%}")
        m2.metric("Expected Volatility (annualized)", f"{port_result.expected_volatility:.2%}")
        m3.metric("Sharpe Ratio", f"{port_result.sharpe_ratio:.2f}")

        if port_result.tickers_dropped:
            st.warning(f"Dropped (insufficient data): {', '.join(port_result.tickers_dropped)}")

        fig_pie = go.Figure(data=[go.Pie(
            labels=port_result.weights.index,
            values=port_result.weights.values,
            hole=0.45,
            marker=dict(colors=[
                "#f0b90b", "#26a69a", "#42a5f5", "#ef5350", "#ab47bc", "#66bb6a", "#ffa726"
            ]),
        )])
        fig_pie.update_layout(
            template="plotly_dark", height=420, margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            title=f"Asset Allocation — {port_result.method}",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.dataframe(
            port_result.weights.to_frame("Weight").style.format({"Weight": "{:.2%}"}),
            use_container_width=True,
        )
    else:
        st.info("Configure your selection above and click **Generate Optimized Portfolio**.")