from __future__ import annotations

from math import erf, exp, log, sqrt

import numpy as np
import pandas as pd


def norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    sigma: float,
    option_type: str,
    dividend_yield: float = 0.0,
) -> float:
    if time_to_expiry <= 0:
        intrinsic = max(0.0, spot - strike) if option_type == "call" else max(0.0, strike - spot)
        return intrinsic

    sigma = max(sigma, 1e-8)
    sqrt_t = sqrt(time_to_expiry)
    d1 = (
        log(spot / strike)
        + (rate - dividend_yield + 0.5 * sigma * sigma) * time_to_expiry
    ) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    discounted_spot = spot * exp(-dividend_yield * time_to_expiry)
    discounted_strike = strike * exp(-rate * time_to_expiry)

    if option_type == "call":
        return discounted_spot * norm_cdf(d1) - discounted_strike * norm_cdf(d2)
    return discounted_strike * norm_cdf(-d2) - discounted_spot * norm_cdf(-d1)


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: str,
    dividend_yield: float = 0.0,
    lower_bound: float = 1e-4,
    upper_bound: float = 5.0,
    tolerance: float = 1e-5,
    max_iterations: int = 120,
) -> float | None:
    if market_price <= 0 or spot <= 0 or strike <= 0 or time_to_expiry <= 0:
        return None

    discounted_spot = spot * exp(-dividend_yield * time_to_expiry)
    discounted_strike = strike * exp(-rate * time_to_expiry)
    intrinsic = max(0.0, discounted_spot - discounted_strike) if option_type == "call" else max(0.0, discounted_strike - discounted_spot)
    max_price = discounted_spot if option_type == "call" else discounted_strike

    if market_price < intrinsic - tolerance or market_price > max_price + tolerance:
        return None

    low = lower_bound
    high = upper_bound
    low_price = black_scholes_price(spot, strike, time_to_expiry, rate, low, option_type, dividend_yield)
    high_price = black_scholes_price(spot, strike, time_to_expiry, rate, high, option_type, dividend_yield)

    if market_price < low_price - tolerance or market_price > high_price + tolerance:
        return None

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        mid_price = black_scholes_price(spot, strike, time_to_expiry, rate, mid, option_type, dividend_yield)

        if abs(mid_price - market_price) < tolerance:
            return mid

        if mid_price > market_price:
            high = mid
        else:
            low = mid

    return 0.5 * (low + high)


def build_surface_grid(
    surface_df: pd.DataFrame,
    bucket_size: float = 2.5,
    iv_column: str = "surface_iv",
) -> pd.DataFrame:
    if surface_df.empty:
        return pd.DataFrame()

    grid_source = surface_df.copy()
    grid_source["moneyness_bucket"] = (
        np.round(grid_source["moneyness_pct"] / bucket_size) * bucket_size
    )

    surface_grid = (
        grid_source.groupby(["days_to_expiry", "moneyness_bucket"], as_index=False)[iv_column]
        .median()
        .pivot(index="days_to_expiry", columns="moneyness_bucket", values=iv_column)
        .sort_index()
        .sort_index(axis=1)
    )

    surface_grid = surface_grid.interpolate(axis=1, limit_direction="both")
    surface_grid = surface_grid.interpolate(axis=0, limit_direction="both")
    return surface_grid


def densify_surface_grid(
    surface_grid: pd.DataFrame,
    x_points: int = 61,
    y_points: int = 55,
) -> pd.DataFrame:
    if surface_grid.empty:
        return pd.DataFrame()

    x_min = float(surface_grid.columns.min())
    x_max = float(surface_grid.columns.max())
    y_min = float(surface_grid.index.min())
    y_max = float(surface_grid.index.max())

    dense_columns = np.linspace(x_min, x_max, x_points)
    dense_index = np.linspace(y_min, y_max, y_points)

    expanded = surface_grid.reindex(surface_grid.index.union(dense_index)).sort_index()
    expanded = expanded.interpolate(method="index", axis=0, limit_direction="both")
    expanded = expanded.reindex(columns=surface_grid.columns.union(dense_columns)).sort_index(axis=1)
    expanded = expanded.interpolate(method="index", axis=1, limit_direction="both")

    dense_grid = expanded.reindex(index=dense_index, columns=dense_columns)
    dense_grid = dense_grid.interpolate(method="index", axis=0, limit_direction="both")
    dense_grid = dense_grid.interpolate(method="index", axis=1, limit_direction="both")
    return dense_grid
