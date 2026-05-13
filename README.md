# Implied Volatility Surface Builder

This project is a Python dashboard that pulls an option chain for any ticker, computes Black-Scholes implied volatility across strikes and expiries, and visualizes:

- A 3D implied volatility surface
- Volatility skew slices by expiry
- Term structure slices at a chosen moneyness
- A 2D heatmap and surface-ready table

## Why this data source

I checked the currently available free options-chain sources before wiring the app:

- `yfinance` / Yahoo Finance: free, no API key, supports expiries and per-expiry option chains, but it is unofficial and intended for personal/research use.
- Alpha Vantage: has official options endpoints, but current documentation marks realtime options as premium and the standard free tier is limited.
- Polygon: current free options tier exists, but snapshot, Greeks, and IV-rich workflow features are paid on higher tiers.
- Cboe delayed quote pages are not a good automation target because their site explicitly prohibits auto-extraction.

For a no-key dashboard that can accept any ticker, `yfinance` is the most practical default.

## Files

- `app.py`: Streamlit dashboard
- `vol_surface/data.py`: option-chain retrieval and surface dataset preparation
- `vol_surface/analytics.py`: Black-Scholes pricing, implied volatility solver, and surface grid builder
- `.vscode/launch.json`: VS Code run configuration

## Run it

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the dashboard:

```bash
streamlit run app.py
```

Or in VS Code, run the launch configuration named `Streamlit: IV Surface Dashboard`.

## Notes

- The dashboard recomputes IV from option prices instead of just reading any provider-supplied IV field.
- It prefers out-of-the-money contracts per strike/expiry when building the surface so the surface is less duplicated by both call and put listings.
- Quotes from free sources can contain stale or crossed markets; the app falls back from bid/ask midpoint to last traded price when needed.

## Research sources

- [yfinance on PyPI](https://pypi.org/project/yfinance/)
- [Alpha Vantage API documentation](https://www.alphavantage.co/documentation/)
- [Polygon options pricing](https://polygon.io/pricing?product=options)
- [Cboe delayed quotes](https://www.cboe.com/delayed_quotes/main/quote_table)
