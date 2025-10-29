"""Predefined ticker collections for quick experiments."""

from __future__ import annotations

TOP_50_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "TSLA",
    "META",
    "BRK.B",
    "LLY",
    "JPM",
    "V",
    "UNH",
    "MA",
    "AVGO",
    "JNJ",
    "WMT",
    "XOM",
    "PG",
    "ORCL",
    "HD",
    "COST",
    "MRK",
    "KO",
    "PEP",
    "BAC",
    "ADBE",
    "CRM",
    "CSCO",
    "TMO",
    "NFLX",
    "ABT",
    "ACN",
    "LIN",
    "DHR",
    "MCD",
    "DIS",
    "NKE",
    "INTC",
    "PFE",
    "AMD",
    "HON",
    "CMCSA",
    "TXN",
    "VZ",
    "IBM",
    "CAT",
    "UPS",
    "PM",
    "BMY",
    "MS",
]
"""A list of 50 large-cap U.S. equities for broad market coverage."""

POPULAR_ETFS = [
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "VTI",
    "XLK",
    "ARKK",
]
"""A handful of diversified ETFs to complement the equity universe."""

DEFAULT_TICKERS = TOP_50_TICKERS + POPULAR_ETFS
"""A convenient combination of equities and ETFs for running simulations."""

__all__ = ["TOP_50_TICKERS", "POPULAR_ETFS", "DEFAULT_TICKERS"]
