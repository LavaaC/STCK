"""Entry point for launching the STCK evolutionary trading simulator."""

from __future__ import annotations

import argparse
import sys
from typing import List

from .data_fetcher import fetch_historical_data
from .evolution import EvolutionConfig, EvolutionEngine
from .tickers import DEFAULT_TICKERS
from .ui import EvolutionTkUI


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the STCK evolutionary simulator")
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Number of generations to run (defaults to configuration value).",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=None,
        help="Population size per ticker (overrides configuration).",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=None,
        help="Initial cash value for each backtest portfolio.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    config = EvolutionConfig()
    if args.generations is not None:
        config.generations = max(0, args.generations)
    if args.population is not None:
        config.population_size = args.population
    if args.cash is not None:
        config.initial_cash = args.cash

    download = fetch_historical_data(DEFAULT_TICKERS, years=6)
    engine = EvolutionEngine(download.data, config=config)

    ui = EvolutionTkUI(engine=engine, tickers=list(download.data.tickers))
    if args.generations is not None:
        max_generations = config.generations
    elif config.generations > 0:
        max_generations = config.generations
    else:
        max_generations = None
    ui.start(max_generations=max_generations)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:  # pragma: no cover - graceful shutdown for CLI
        pass
