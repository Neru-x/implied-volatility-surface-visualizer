from __future__ import annotations

from datetime import datetime, time, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from vol_surface.analytics import implied_volatility


def _coalesce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    output = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in columns:
        if column in frame.columns:
            output = output.fillna(pd.to_numeric(frame[column], errors="coerce"))
    return output


def _extract_spot_price(ticker: yf.Ticker) -> float:
    fast_info = getattr(ticker, "fast_info", {}) or {}
    for key in ("lastPrice", "last_price", "regularMarketPrice", "previousClose"):
        value = fast_info.get(key)
        if value:
            return float(value)

    history = ticker.history(period="5d", auto_adjust=False)
    if history.empty:
        raise ValueError("Unable to determine spot price.")
    return float(history["Close"].dropna().iloc[-1])


def _extract_dividend_yield(ticker: yf.Ticker, override: float) -> float:
    if override > 0:
        return float(override)

    info = ticker.info or {}
    for key in ("dividendYield", "trailingAnnualDividendYield"):
        value = info.get(key)
        if value is not None:
            return float(value)
    return 0.0


def _market_price(frame: pd.DataFrame) -> pd.Series:
    bid = _coalesce_numeric(frame, ["bid"])
    ask = _coalesce_numeric(frame, ["ask"])
    last_price = _coalesce_numeric(frame, ["lastPrice", "last_price"])

    midpoint = (bid + ask) / 2.0
    spread_ratio = (ask - bid) / midpoint.replace(0, np.nan)
    midpoint = midpoint.where((bid > 0) & (ask > 0) & (spread_ratio <= 0.35))
    return midpoint.fillna(last_price)


def _prepare_contract_frame(
    symbol: str,
    option_type: str,
    frame: pd.DataFrame,
    expiration: str,
    spot_price: float,
) -> pd.DataFrame:
    output = frame.copy()
    output["symbol"] = symbol
    output["option_type"] = option_type
    output["expiration"] = expiration
    output["strike"] = pd.to_numeric(output["strike"], errors="coerce")
    output["bid"] = _coalesce_numeric(output, ["bid"])
    output["ask"] = _coalesce_numeric(output, ["ask"])
    output["lastPrice"] = _coalesce_numeric(output, ["lastPrice", "last_price"])
    output["provider_iv"] = _coalesce_numeric(output, ["impliedVolatility", "implied_volatility"])
    output["volume"] = _coalesce_numeric(output, ["volume"]).fillna(0)
    output["openInterest"] = _coalesce_numeric(output, ["openInterest", "open_interest"]).fillna(0)
    output["market_price"] = _market_price(output)
    output["moneyness_pct"] = ((output["strike"] / spot_price) - 1.0) * 100.0
    output["is_otm"] = np.where(
        option_type == "call",
        output["strike"] >= spot_price,
        output["strike"] <= spot_price,
    )
    return output


def _calculate_days_to_expiry(expiration: str) -> float:
    expiry_date = datetime.combine(pd.Timestamp(expiration).date(), time(hour=16), tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return max((expiry_date - now).total_seconds() / 86400.0, 0.0)


@st.cache_data(ttl=900, show_spinner=False)
def load_symbol_surface(
    symbol: str,
    max_expiries: int,
    risk_free_rate: float,
    dividend_yield_override: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    ticker = yf.Ticker(symbol)
    expiries = list(ticker.options or [])
    if not expiries:
        raise ValueError(f"No listed option expiries found for {symbol}.")

    selected_expiries = expiries[:max_expiries]
    spot_price = _extract_spot_price(ticker)
    dividend_yield = _extract_dividend_yield(ticker, dividend_yield_override)

    prepared_frames: list[pd.DataFrame] = []
    for expiration in selected_expiries:
        chain = ticker.option_chain(expiration)
        prepared_frames.append(
            _prepare_contract_frame(symbol, "call", chain.calls, expiration, spot_price)
        )
        prepared_frames.append(
            _prepare_contract_frame(symbol, "put", chain.puts, expiration, spot_price)
        )

    raw_df = pd.concat(prepared_frames, ignore_index=True)
    raw_df["days_to_expiry"] = raw_df["expiration"].map(_calculate_days_to_expiry)
    raw_df["time_to_expiry"] = raw_df["days_to_expiry"] / 365.25
    raw_df = raw_df.replace([np.inf, -np.inf], np.nan)
    raw_df = raw_df.dropna(subset=["strike", "market_price", "time_to_expiry"])
    raw_df = raw_df.loc[(raw_df["market_price"] > 0) & (raw_df["time_to_expiry"] > 0)]
    raw_df = raw_df.loc[(raw_df["days_to_expiry"] >= 3) & (raw_df["days_to_expiry"] <= 730)]
    raw_df = raw_df.loc[raw_df["moneyness_pct"].between(-55, 55)]
    raw_df = raw_df.loc[
        (raw_df["openInterest"] > 0)
        | (raw_df["volume"] > 0)
        | (raw_df["provider_iv"].between(0.01, 3.0))
    ]

    raw_df["surface_priority"] = np.where(raw_df["is_otm"], 0, 1)
    raw_df = raw_df.sort_values(
        ["expiration", "strike", "surface_priority", "openInterest", "volume"],
        ascending=[True, True, True, False, False],
    )

    surface_df = raw_df.drop_duplicates(subset=["expiration", "strike"], keep="first").copy()
    surface_df["implied_volatility"] = surface_df.apply(
        lambda row: implied_volatility(
            market_price=float(row["market_price"]),
            spot=float(spot_price),
            strike=float(row["strike"]),
            time_to_expiry=float(row["time_to_expiry"]),
            rate=float(risk_free_rate),
            option_type=str(row["option_type"]),
            dividend_yield=float(dividend_yield),
        ),
        axis=1,
    )
    surface_df["implied_volatility"] = surface_df["implied_volatility"].fillna(surface_df["provider_iv"])

    surface_df = surface_df.dropna(subset=["implied_volatility"]).copy()
    surface_df["implied_volatility"] = pd.to_numeric(surface_df["implied_volatility"], errors="coerce")
    surface_df = surface_df.loc[
        (surface_df["implied_volatility"] >= 0.01) & (surface_df["implied_volatility"] <= 3.0)
    ].copy()
    surface_df = surface_df.sort_values(["expiration", "moneyness_pct", "openInterest"], ascending=[True, True, False])
    surface_df["surface_iv"] = (
        surface_df.groupby("expiration")["implied_volatility"]
        .transform(lambda series: series.rolling(window=5, center=True, min_periods=1).median())
    )
    expiry_counts = surface_df.groupby("expiration")["strike"].transform("count")
    surface_df = surface_df.loc[expiry_counts >= 5].copy()
    low_cut = surface_df.groupby("expiration")["surface_iv"].transform(
        lambda series: series.quantile(0.03)
    )
    high_cut = surface_df.groupby("expiration")["surface_iv"].transform(
        lambda series: series.quantile(0.97)
    )
    surface_df["surface_iv"] = surface_df["surface_iv"].clip(lower=low_cut, upper=high_cut)
    surface_df["open_interest"] = surface_df["openInterest"]
    surface_df["moneyness_bucket"] = np.round(surface_df["moneyness_pct"] / 2.5) * 2.5

    return surface_df, {
        "spot_price": spot_price,
        "dividend_yield": dividend_yield,
    }
