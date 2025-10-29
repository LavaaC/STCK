"""Utilities for downloading historical market data for simulations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Sequence, TYPE_CHECKING

from .data import HistoricalData

if TYPE_CHECKING:  # pragma: no cover - import used only for type checking
    import pandas as pd


@dataclass(frozen=True)
class DownloadResult:
    """Container for the data returned by :func:`fetch_historical_data`."""

    data: HistoricalData
    start_date: datetime
    end_date: datetime


def _normalize_ticker(ticker: str) -> str:
    """Convert tickers with dots to the Yahoo Finance compatible format."""

    return ticker.replace(".", "-")


def fetch_historical_data(
    tickers: Sequence[str],
    *,
    years: int = 6,
    interval: str = "1d",
) -> DownloadResult:
    """Download adjusted close prices for ``tickers`` covering ``years`` years.

    Parameters
    ----------
    tickers:
        Iterable of ticker symbols to download. Symbols containing dots are
        automatically translated to Yahoo Finance's dash notation to ensure
        Berkshire Hathaway style tickers resolve correctly.
    years:
        Number of years of trailing data to request. Defaults to six years.
    interval:
        Sampling interval passed directly to :func:`yfinance.download`.

    Returns
    -------
    DownloadResult
        A dataclass containing the synchronized historical data along with the
        start and end timestamps of the downloaded window.
    """

    if not tickers:
        raise ValueError("At least one ticker must be provided")

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("fetch_historical_data requires the 'pandas' package") from exc
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("fetch_historical_data requires the 'yfinance' package") from exc

    normalized: Dict[str, str] = {ticker: _normalize_ticker(ticker) for ticker in tickers}
    unique_symbols = sorted(set(normalized.values()))

    download = yf.download(
        tickers=unique_symbols,
        period=f"{years}y",
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if download.empty:
        raise RuntimeError("No historical price data was returned by yfinance")

    if isinstance(download.columns, pd.MultiIndex):
        adj_close = download.loc[:, (slice(None), "Adj Close")]
        adj_close.columns = [col[0] for col in adj_close.columns]
    else:
        if "Adj Close" not in download.columns:
            raise RuntimeError("Downloaded data does not contain 'Adj Close' prices")
        adj_close = download[["Adj Close"]]
        adj_close.columns = unique_symbols

    # Ensure we have monotonically increasing timestamps and drop rows with any gaps.
    adj_close = adj_close.sort_index()
    adj_close = adj_close.dropna(how="all")
    adj_close = adj_close.ffill().dropna()

    missing_symbols = [symbol for symbol in unique_symbols if symbol not in adj_close.columns]
    if missing_symbols:
        joined = ", ".join(missing_symbols)
        raise RuntimeError(f"Missing adjusted close data for symbols: {joined}")

    prices: Dict[str, Sequence[float]] = {}
    for original, normalized_symbol in normalized.items():
        series = adj_close[normalized_symbol]
        if series.isna().any():
            raise RuntimeError(f"Ticker {original} contains missing values after cleaning")
        prices[original] = series.tolist()

    historical = HistoricalData(prices)
    start_date = adj_close.index[0].to_pydatetime()
    end_date = adj_close.index[-1].to_pydatetime()
    return DownloadResult(data=historical, start_date=start_date, end_date=end_date)


__all__ = ["DownloadResult", "fetch_historical_data"]
