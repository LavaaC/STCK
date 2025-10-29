"""Evolutionary trading formulas and backtesting tools."""

from .data import HistoricalData, PriceHistory
from .formulas import TradingFormula, FormulaFactory
from .portfolio import PortfolioState, TickerAllocation, PortfolioBacktester, BacktestResult
from .evolution import (
    EvolutionEngine,
    EvolutionConfig,
    GenerationReport,
    PopulationMetrics,
    PopulationTransitionSummary,
)
from .ui import EvolutionConsoleUI, EvolutionTkUI
from .data_fetcher import DownloadResult, fetch_historical_data
from .tickers import TOP_50_TICKERS, POPULAR_ETFS, DEFAULT_TICKERS

__all__ = [
    "HistoricalData",
    "PriceHistory",
    "TradingFormula",
    "FormulaFactory",
    "PortfolioState",
    "TickerAllocation",
    "PortfolioBacktester",
    "BacktestResult",
    "EvolutionEngine",
    "EvolutionConfig",
    "GenerationReport",
    "PopulationMetrics",
    "PopulationTransitionSummary",
    "EvolutionConsoleUI",
    "EvolutionTkUI",
    "DownloadResult",
    "fetch_historical_data",
    "TOP_50_TICKERS",
    "POPULAR_ETFS",
    "DEFAULT_TICKERS",
]
