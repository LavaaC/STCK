# STCK Evolutionary Trading Simulator

This project provides a small framework for experimenting with evolutionary trading formulas. A population of formulas is generated for each ticker, backtested over historical price data, and evolved by removing the worst performers while mutating the best ones.

Key features:

- **Formula trees** composed of indicators and arithmetic operators.
- **Portfolio simulation** with an uninvested cash pool and per-ticker priorities that control how capital is reallocated.
- **Backtesting** over user-supplied historical price data to evaluate equity curves, drawdowns, and trade allocations across multiple historical windows.
- **Evolution engine** that eliminates poorly performing formulas, protects the strongest 10%, and probabilistically retires the middle performers while mutating survivors to explore new strategies.
- **Console dashboard** that displays generation-by-generation performance, supports pausing, saving, and plots how the best performers improve over time.
- **Order tracking** that rebalances once per ticker at the end of each trading day so you can replay the generated trades on future price data.

The framework is intentionally lightweight so that users can plug in their own datasets, extend the indicator library, or integrate alternative fitness scores.

## Getting started

Install the project in editable mode (optionally with the testing extras):

```bash
pip install -e .[test]
```

Run the unit tests:

```bash
pytest
```

Refer to the documentation in the `stck` package for details on using the `HistoricalData`, `PortfolioBacktester`, and `EvolutionEngine` classes.

## Evaluating formulas over multiple time periods

Each trading formula now carries a set of evaluation windows (for example 30, 90, and 180 trading days). During evolution, formulas are backtested on the trailing slices of each window and their scores are averaged. The windows mutate along with the formula tree so that individual strategies can specialize in the time horizons where they perform best.

## Interactive console dashboard

The `EvolutionConsoleUI` class provides a simple console dashboard that tracks generations and exposes pause/resume controls. A quick example:

```python
from stck.data import HistoricalData
from stck.evolution import EvolutionConfig, EvolutionEngine
from stck.ui import EvolutionConsoleUI

data = HistoricalData({"AAPL": [...], "MSFT": [...]})
engine = EvolutionEngine(data, EvolutionConfig(population_size=20, generations=50))
ui = EvolutionConsoleUI(engine, ["AAPL", "MSFT"])
ui.interactive(plot=True, show_plot=False)
```

While the simulation is running you can press:

- `p` to pause after the current generation
- `r` to resume
- `s` (while paused) to save a strategy from the latest generation
- `q` to stop immediately

Each generation prints the best performer, along with the average equity for the top stock, top 10%, top 20%, and the whole population. When the run completes a performance chart is saved (by default as `performance.png`). Install `matplotlib` to enable plotting:

```bash
pip install matplotlib
```

## Saving strategies and replaying them on new data

Pause the dashboard once a generation finishes and press `s` to store any ticker's strategy as JSON. The prompt shows the top performers for the selected ticker so you can choose which one to keep. Saved files include the generation, formula definition, evaluation windows, and the training window size so they can be reapplied later.

To inspect what that strategy would have done after more price history becomes available, load the file and call `future_orders` with an updated `HistoricalData` instance:

```python
from pathlib import Path

from stck.data import HistoricalData
from stck.strategies import load_strategy, future_orders

saved = load_strategy(Path("strategy_AAPL_gen5.json"))
expanded_data = HistoricalData({"AAPL": existing_prices + new_prices})
orders = future_orders(saved, expanded_data)
for day, trades in enumerate(orders, start=saved.training_length + 1):
    print(f"Day {day} orders:")
    for trade in trades:
        print(f"  {trade.action} {trade.shares:.2f} shares at {trade.price:.2f}")
```

The returned orders align with the simulator's end-of-day trading rule and contain at most one buy or sell per ticker per day.
