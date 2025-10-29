"""Console utilities for running and visualizing the evolution process."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .evolution import EvolutionEngine, GenerationReport, PopulationMetrics
from .formulas import TradingFormula


@dataclass
class EvolutionConsoleUI:
    """Interactive console runner for the evolutionary trading simulator."""

    engine: EvolutionEngine
    tickers: List[str]
    population: dict[str, List[TradingFormula]] = field(init=False)
    generation: int = field(default=0, init=False)
    history: List[PopulationMetrics] = field(default_factory=list, init=False)

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
                command = input("[p]ause/[r]esume/[q]uit> ").strip().lower()
                if command in {"p", "pause"}:
                    self.pause()
                elif command in {"r", "resume"}:
                    self.resume()
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
