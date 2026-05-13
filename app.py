from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from vol_surface.analytics import build_surface_grid, densify_surface_grid
from vol_surface.data import load_symbol_surface


st.set_page_config(
    page_title="Implied Volatility Surface Builder",
    page_icon="📈",
    layout="wide",
)


def format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.2%}"


def render_glossary_page() -> None:
    st.title("Glossary")
    st.caption("Key terms used in the volatility surface dashboard and what they mean in practice.")

    glossary_items = [
        ("Underlying", "The stock or ETF whose listed options are being analyzed, such as `AAPL` or `SPY`."),
        ("Ticker", "The market symbol entered into the dashboard to request a fresh option chain."),
        ("Option Chain", "The full set of listed calls and puts across strikes and expirations for one underlying."),
        ("Call Option", "An option that gives the holder the right, but not the obligation, to buy the underlying at the strike price."),
        ("Put Option", "An option that gives the holder the right, but not the obligation, to sell the underlying at the strike price."),
        ("Strike", "The fixed price at which the option can be exercised."),
        ("Expiration", "The date on which the option contract expires."),
        ("Days to Expiry (DTE)", "The number of calendar days remaining until the option expires."),
        ("Time to Expiry", "Days to expiry converted into years, which is the form used by Black-Scholes."),
        ("Spot Price", "The current price of the underlying used as the anchor for moneyness and IV calculations."),
        ("Bid", "The highest displayed price buyers are currently willing to pay for an option."),
        ("Ask", "The lowest displayed price sellers are currently willing to accept for an option."),
        ("Mid Price", "The average of bid and ask. This app prefers the midpoint when the spread looks sane."),
        ("Market Price", "The option price fed into the IV solver. It is usually the midpoint, otherwise the last traded price."),
        ("Last Price", "The most recent trade price recorded for the option contract."),
        ("Open Interest", "The number of currently open option contracts that have not been closed or exercised."),
        ("Volume", "The number of contracts traded during the current session."),
        ("Risk-Free Rate", "The rate used in Black-Scholes to discount the strike and solve implied volatility."),
        ("Dividend Yield", "The underlying’s cash-yield assumption used when pricing options on dividend-paying names."),
        ("Black-Scholes Model", "The option-pricing model used here to back out implied volatility from observed option prices."),
        ("Implied Volatility (IV)", "The volatility level that makes the model price match the observed market price of the option."),
        ("Provider IV", "Any IV field already supplied by the data source. The app uses it only as a fallback when the solver cannot stabilize."),
        ("Raw IV", "The direct implied-volatility result from the chosen market price before smoothing or clipping."),
        ("Surface IV", "The cleaned and smoothed IV used to draw the display surface."),
        ("Moneyness", "A measure of strike relative to spot. In this app it is shown as percentage distance from the current spot price."),
        ("Moneyness (%)", "Computed as `(strike / spot - 1) * 100`, so positive values are above spot and negative values are below spot."),
        ("OTM", "Out of the money. Calls with strikes above spot and puts with strikes below spot. The app prefers OTM contracts for the surface."),
        ("ATM", "At the money. A strike very close to the current spot price."),
        ("ITM", "In the money. Calls with strikes below spot and puts with strikes above spot."),
        ("Skew", "How implied volatility changes across strikes or moneyness for a single expiry."),
        ("Term Structure", "How implied volatility changes across expirations for a chosen moneyness region."),
        ("Volatility Surface", "The 3D relationship between moneyness, time to expiry, and implied volatility."),
        ("Interpolation", "The process of filling the gaps between observed contracts so the plotted surface looks continuous."),
        ("Outlier Clipping", "Trimming extreme IV values that are likely caused by stale quotes or illiquid contracts."),
        ("Per-Expiry Smoothing", "Applying a rolling median within each expiration to reduce jagged spikes before plotting the surface."),
        ("Liquidity Filter", "The rule set that removes contracts with weak or suspicious quotes before building the surface."),
        ("Deep Wing", "Very far OTM or ITM strikes that often have unreliable or thinly traded quotes."),
    ]

    for term, meaning in glossary_items:
        st.markdown(f"**{term}**")
        st.write(meaning)

    st.subheader("How The App Uses These Terms")
    st.markdown(
        "- The dashboard pulls an option chain for the selected ticker.\n"
        "- It chooses tradable-looking contracts, solves or falls back to IV, then smooths the result into `surface_iv`.\n"
        "- The main surface shows `surface_iv` across moneyness and days to expiry.\n"
        "- The skew chart fixes one expiration and looks sideways across moneyness.\n"
        "- The term-structure chart fixes one moneyness region and looks forward across expirations."
    )


def build_surface_figure(grid: pd.DataFrame, surface_df: pd.DataFrame) -> go.Figure:
    dense_grid = densify_surface_grid(grid, x_points=81, y_points=61)
    z_values = dense_grid.to_numpy()
    z_min = float(surface_df["surface_iv"].quantile(0.03))
    z_max = float(surface_df["surface_iv"].quantile(0.97))

    fig = go.Figure(
        data=[
            go.Surface(
                x=dense_grid.columns.tolist(),
                y=dense_grid.index.tolist(),
                z=z_values,
                colorscale="Turbo",
                cmin=z_min,
                cmax=z_max,
                opacity=0.98,
                lighting={
                    "ambient": 0.7,
                    "diffuse": 0.9,
                    "roughness": 0.55,
                    "specular": 0.25,
                    "fresnel": 0.08,
                },
                lightposition={"x": -400, "y": -250, "z": 420},
                colorbar={"title": "IV", "tickformat": ".0%"},
                contours={
                    "x": {"show": True, "color": "rgba(255,255,255,0.18)", "width": 1},
                    "y": {"show": True, "color": "rgba(255,255,255,0.18)", "width": 1},
                    "z": {"show": True, "color": "rgba(255,255,255,0.28)", "width": 1},
                },
                hovertemplate=(
                    "Moneyness: %{x:.1f}%<br>"
                    "Days to expiry: %{y:.0f}<br>"
                    "Surface IV: %{z:.2%}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title=f"{surface_df['symbol'].iloc[0]} Volatility Surface",
        height=780,
        paper_bgcolor="#05070b",
        plot_bgcolor="#05070b",
        font={"family": "Menlo, Monaco, monospace", "color": "#f4f7fb"},
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        scene={
            "bgcolor": "#05070b",
            "aspectratio": {"x": 1.35, "y": 1.05, "z": 0.78},
            "camera": {"eye": {"x": 1.45, "y": -1.6, "z": 0.92}},
            "xaxis": {
                "title": "Moneyness (%)",
                "backgroundcolor": "#05070b",
                "gridcolor": "rgba(255,255,255,0.18)",
                "showbackground": True,
                "zerolinecolor": "rgba(255,255,255,0.30)",
            },
            "yaxis": {
                "title": "Days to Expiry",
                "backgroundcolor": "#05070b",
                "gridcolor": "rgba(255,255,255,0.18)",
                "showbackground": True,
                "zerolinecolor": "rgba(255,255,255,0.30)",
            },
            "zaxis": {
                "title": "Implied Volatility",
                "backgroundcolor": "#05070b",
                "gridcolor": "rgba(255,255,255,0.18)",
                "showbackground": True,
                "zerolinecolor": "rgba(255,255,255,0.30)",
                "tickformat": ".0%",
                "range": [z_min, z_max],
            },
        },
    )
    return fig


def build_skew_figure(df: pd.DataFrame, selected_expiry: str) -> go.Figure:
    slice_df = df[df["expiration"] == selected_expiry].sort_values("moneyness_pct")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=slice_df["moneyness_pct"],
            y=slice_df["surface_iv"],
            mode="lines",
            line={"shape": "spline", "smoothing": 0.8, "color": "#38bdf8", "width": 3},
            name="Surface skew",
            hovertemplate="Moneyness: %{x:.1f}%<br>IV: %{y:.2%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=slice_df["moneyness_pct"],
            y=slice_df["implied_volatility"],
            mode="markers",
            marker={"size": 5, "color": "rgba(244,247,251,0.42)"},
            name="Raw contracts",
            hovertemplate="Moneyness: %{x:.1f}%<br>Raw IV: %{y:.2%}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.45)")
    fig.update_layout(
        title=f"Skew Slice | {selected_expiry}",
        height=420,
        paper_bgcolor="#05070b",
        plot_bgcolor="#05070b",
        font={"family": "Menlo, Monaco, monospace", "color": "#f4f7fb"},
        margin={"l": 8, "r": 8, "t": 40, "b": 8},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        xaxis_title="Moneyness (%)",
        yaxis_title="Implied Volatility",
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.12)", tickformat=".0%")
    return fig


def build_term_structure_figure(df: pd.DataFrame, target_moneyness: float) -> go.Figure:
    tolerance = 2.5
    term_df = (
        df.assign(distance=(df["moneyness_pct"] - target_moneyness).abs())
        .query("distance <= @tolerance")
        .sort_values(["expiration", "distance", "open_interest"], ascending=[True, True, False])
        .groupby("expiration", as_index=False)
        .first()
        .sort_values("days_to_expiry")
    )

    if term_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"Term Structure | {target_moneyness:+.1f}% Moneyness",
            height=420,
            paper_bgcolor="#05070b",
            plot_bgcolor="#05070b",
            font={"family": "Menlo, Monaco, monospace", "color": "#f4f7fb"},
            annotations=[
                {
                    "text": "No contracts landed near the selected moneyness.",
                    "showarrow": False,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "font": {"color": "#f4f7fb"},
                }
            ],
        )
        return fig

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=term_df["days_to_expiry"],
            y=term_df["surface_iv"],
            mode="lines+markers",
            line={"shape": "spline", "smoothing": 0.7, "color": "#f59e0b", "width": 3},
            marker={"size": 7, "color": "#fde68a", "line": {"color": "#f59e0b", "width": 1}},
            name="Term structure",
            hovertemplate="Days: %{x:.0f}<br>IV: %{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Term Structure | {target_moneyness:+.1f}% Moneyness",
        height=420,
        paper_bgcolor="#05070b",
        plot_bgcolor="#05070b",
        font={"family": "Menlo, Monaco, monospace", "color": "#f4f7fb"},
        margin={"l": 8, "r": 8, "t": 40, "b": 8},
        xaxis_title="Days to Expiry",
        yaxis_title="Implied Volatility",
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.12)", tickformat=".0%")
    return fig


st.title("Implied Volatility Surface Builder")
st.caption(
    "Type any U.S. stock or ETF ticker to pull its option chain, compute Black-Scholes implied volatility, "
    "and visualize a cleaner tradable surface."
)

with st.sidebar:
    page = st.radio("Page", ["Surface Builder", "Glossary"])
    st.header("Controls")
    symbol = st.text_input("Ticker", value="AAPL", max_chars=10).strip().upper()
    max_expiries = st.slider("Max expiries", min_value=4, max_value=40, value=16)
    risk_free_rate = st.slider("Risk-free rate", min_value=0.0, max_value=0.10, value=0.04, step=0.0025)
    dividend_yield_override = st.number_input(
        "Dividend yield override",
        min_value=0.0,
        max_value=0.20,
        value=0.0,
        step=0.0025,
        help="Leave at 0 to use Yahoo Finance's reported yield when available.",
    )
    st.markdown(
        "Data source: `yfinance` / Yahoo Finance\n\n"
        "Surface construction: cleaned OTM chain, Black-Scholes IV, per-expiry smoothing, outlier clipping."
    )

if page == "Glossary":
    render_glossary_page()
    st.stop()

if not symbol:
    st.info("Enter a ticker in the sidebar to build a volatility surface.")
    st.stop()

with st.spinner(f"Pulling option chain for {symbol} and computing implied volatilities..."):
    try:
        surface_df, metadata = load_symbol_surface(
            symbol=symbol,
            max_expiries=max_expiries,
            risk_free_rate=risk_free_rate,
            dividend_yield_override=dividend_yield_override,
        )
    except Exception as exc:
        st.error(f"Unable to build a surface for {symbol}: {exc}")
        st.stop()

if surface_df.empty:
    st.error("No usable option contracts were returned for that ticker.")
    st.stop()

surface_grid = build_surface_grid(surface_df, iv_column="surface_iv")
if surface_grid.empty or surface_grid.shape[0] < 2 or surface_grid.shape[1] < 3:
    st.error("Not enough stable option data was available to build a smooth surface for that ticker.")
    st.stop()

atm_slice = surface_df.iloc[(surface_df["moneyness_pct"]).abs().argsort()[:1]]
atm_iv = atm_slice["surface_iv"].iloc[0] if not atm_slice.empty else None

metric_spot, metric_contracts, metric_expiries, metric_iv = st.columns(4)
metric_spot.metric("Spot", f"{metadata['spot_price']:.2f}")
metric_contracts.metric("Contracts Used", f"{len(surface_df):,}")
metric_expiries.metric("Expiries", f"{surface_df['expiration'].nunique()}")
metric_iv.metric("ATM IV", format_percent(atm_iv))

tab_surface, tab_slices, tab_data = st.tabs(["Surface", "Slices", "Data"])

with tab_surface:
    st.plotly_chart(build_surface_figure(surface_grid, surface_df), use_container_width=True)

with tab_slices:
    skew_col, term_col = st.columns(2)
    expiries = sorted(surface_df["expiration"].unique().tolist(), key=lambda value: pd.Timestamp(value))
    selected_expiry = skew_col.selectbox(
        "Expiry for skew slice",
        expiries,
        index=min(2, len(expiries) - 1),
    )
    moneyness_bounds = surface_df["moneyness_pct"].clip(-40, 40)
    target_moneyness = term_col.slider(
        "Target moneyness for term structure (%)",
        min_value=math.floor(moneyness_bounds.min()),
        max_value=math.ceil(moneyness_bounds.max()),
        value=0,
        step=1,
    )
    skew_col.plotly_chart(build_skew_figure(surface_df, selected_expiry), use_container_width=True)
    term_col.plotly_chart(build_term_structure_figure(surface_df, target_moneyness), use_container_width=True)

with tab_data:
    st.subheader("Surface-ready option dataset")
    st.dataframe(
        surface_df[
            [
                "expiration",
                "days_to_expiry",
                "strike",
                "option_type",
                "market_price",
                "implied_volatility",
                "surface_iv",
                "moneyness_pct",
                "open_interest",
                "volume",
            ]
        ].sort_values(["days_to_expiry", "strike"]),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown(
        "- Near-expiry noise and deep-wing outliers are filtered before the surface is built.\n"
        "- The display surface uses smoothed per-expiry IV (`surface_iv`) rather than plotting raw spikes directly.\n"
        "- If the numerical solver cannot stabilize a contract, the app falls back to the provider IV only when it is in a sane range."
    )
