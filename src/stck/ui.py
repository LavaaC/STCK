"""Console utilities for running and visualizing the evolution process."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .evolution import EvolutionEngine, GenerationReport, PopulationMetrics
from .formulas import TradingFormula
from .strategies import SavedStrategy, save_strategy


@dataclass
class EvolutionConsoleUI:
    """Interactive console runner for the evolutionary trading simulator."""

    engine: EvolutionEngine
    tickers: List[str]
    population: dict[str, List[TradingFormula]] = field(init=False)
    generation: int = field(default=0, init=False)
    history: List[PopulationMetrics] = field(default_factory=list, init=False)
    latest_report: Optional[GenerationReport] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.population = self.engine.initialize_population(self.tickers)
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._stop_event = threading.Event()

    # Control helpers -----------------------------------------------------------------

    def pause(self) -> None:
        """Pause the simulation after the current generation completes."""

        self._pause_event.clear()
        print("Simulation paused. Press 'r' to resume or 'q' to stop.")

    def resume(self) -> None:
        """Resume the simulation if it is currently paused."""

        if not self._pause_event.is_set():
            self._pause_event.set()
            print("Resuming simulation...")

    def toggle_pause(self) -> None:
        """Toggle between paused and running states."""

        if self._pause_event.is_set():
            self.pause()
        else:
            self.resume()

    def stop(self) -> None:
        """Stop the simulation gracefully."""

        self._stop_event.set()
        self._pause_event.set()
        print("Stopping simulation...")

    @property
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    # Running -------------------------------------------------------------------------

    def run(
        self,
        max_generations: Optional[int] = None,
        *,
        plot: bool = False,
        show_plot: bool = False,
        save_path: Optional[str] = None,
    ) -> None:
        """Run the evolutionary loop and optionally plot performance metrics."""

        total_generations = max_generations or self.engine.config.generations
        for gen in range(total_generations):
            if self._stop_event.is_set():
                break
            self._wait_if_paused()
            if self._stop_event.is_set():
                break
            self.population, report = self.engine.evolve(self.population, generation=gen)
            self._record_generation(report)
            self._render_report(report)
            self.generation = gen + 1
        if plot:
            self.plot_history(show=show_plot, save_path=save_path)

    def interactive(
        self,
        max_generations: Optional[int] = None,
        *,
        plot: bool = True,
        show_plot: bool = False,
        save_path: Optional[str] = "performance.png",
    ) -> None:
        """Launch the simulation and accept pause/resume commands from stdin."""

        thread = threading.Thread(
            target=self.run,
            kwargs={
                "max_generations": max_generations,
                "plot": False,
                "show_plot": show_plot,
                "save_path": save_path,
            },
            daemon=True,
        )
        thread.start()
        try:
            while thread.is_alive():
                command = input("[p]ause/[r]esume/[s]ave/[q]uit> ").strip().lower()
                if command in {"p", "pause"}:
                    self.pause()
                elif command in {"r", "resume"}:
                    self.resume()
                elif command in {"s", "save"}:
                    self._prompt_save_strategy()
                elif command in {"q", "quit"}:
                    self.stop()
                    break
        except KeyboardInterrupt:
            self.stop()
        finally:
            self.resume()
            self._stop_event.set()
            thread.join()
            if plot:
                self.plot_history(show=show_plot, save_path=save_path)

    # Visualization -------------------------------------------------------------------

    def plot_history(self, *, show: bool = True, save_path: Optional[str] = None) -> None:
        """Plot the stored metrics by generation using matplotlib if available."""

        if not self.history:
            print("No historical metrics available to plot.")
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover - optional dependency
            print("matplotlib is not installed; skipping performance plot.")
            return

        generations = [m.generation + 1 for m in self.history]
        top_values = [m.top_equity for m in self.history]
        top10_values = [m.top10_mean for m in self.history]
        top20_values = [m.top20_mean for m in self.history]

        plt.figure(figsize=(8, 4.5))
        plt.plot(generations, top_values, label="Top Formula")
        plt.plot(generations, top10_values, label="Top 10% Avg")
        plt.plot(generations, top20_values, label="Top 20% Avg")
        plt.xlabel("Generation")
        plt.ylabel("Final Equity")
        plt.title("Evolution Performance by Generation")
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Saved performance plot to {save_path}.")
        if show:
            plt.show()
        else:
            plt.close()

    # Internal utilities ---------------------------------------------------------------

    def _wait_if_paused(self) -> None:
        while not self._pause_event.is_set() and not self._stop_event.is_set():
            time.sleep(0.1)

    def _record_generation(self, report: GenerationReport) -> None:
        self.history.append(report.metrics)
        self.latest_report = report

    def _render_report(self, report: GenerationReport) -> None:
        metrics = report.metrics
        best = report.best_performance
        windows = ", ".join(str(w.window) for w in best.windows)
        print("=" * 60)
        print(f"Generation {report.generation + 1}")
        print(f"Top final equity: {metrics.top_equity:,.2f}")
        print(f"Top 10% average equity: {metrics.top10_mean:,.2f}")
        print(f"Top 20% average equity: {metrics.top20_mean:,.2f}")
        print(f"Population average equity: {metrics.average_equity:,.2f}")
        print(f"Best formula ticker: {best.ticker}")
        print(f"Evaluation windows: {windows}")
        print(f"Formula description: {best.formula.describe()}")
        print("-" * 60)

    # Saving -------------------------------------------------------------------------

    def _prompt_save_strategy(self) -> None:
        if not self.is_paused:
            print("Pause the simulation before saving a strategy.")
            return
        if self.latest_report is None:
            print("No generation has completed yet; nothing to save.")
            return

        report = self.latest_report
        ticker = input("Ticker to save from (default best ticker)> ").strip().upper() or report.best_performance.ticker
        if ticker not in report.ticker_performances:
            print(f"Ticker '{ticker}' was not evaluated in the latest generation.")
            return

        performances = sorted(
            report.ticker_performances[ticker], key=lambda p: p.average_final_equity, reverse=True
        )
        for idx, perf in enumerate(performances[:10]):
            print(
                f"[{idx}] equity={perf.average_final_equity:,.2f} drawdown={perf.average_max_drawdown:.2%}"
                f" formula={perf.formula.describe()}"
            )
        selection = input("Select formula index to save (default 0)> ").strip()
        choice = 0
        if selection:
            try:
                choice = max(0, min(int(selection), len(performances) - 1))
            except ValueError:
                print("Invalid selection; defaulting to best formula.")
                choice = 0

        selected = performances[choice]
        default_name = f"strategy_{ticker}_gen{report.generation + 1}.json"
        path_input = input(f"Save path (default {default_name})> ").strip()
        path = Path(path_input) if path_input else Path(default_name)

        strategy = SavedStrategy(
            ticker=ticker,
            generation=report.generation,
            formula=selected.formula.clone(),
            training_length=len(self.engine.data),
            initial_cash=self.engine.config.initial_cash,
        )
        location = save_strategy(path, strategy)
        print(f"Saved strategy for {ticker} to {location}")
