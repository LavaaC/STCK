"""Console utilities for running and visualizing the evolution process."""

from __future__ import annotations

import itertools
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

try:  # pragma: no cover - optional dependency in test environments
    import tkinter as tk
except ImportError:  # pragma: no cover - tkinter may be unavailable on some systems
    tk = None

from .evolution import (
    EvolutionEngine,
    GenerationReport,
    PopulationMetrics,
    PortfolioMember,
)


def determine_generation_limit(
    engine_limit: int, requested_limit: Optional[int]
) -> Optional[int]:
    """Resolve the number of generations to run.

    A limit of ``0`` or ``None`` indicates the simulation should run indefinitely
    until manually stopped.
    """

    if requested_limit is not None:
        return max(0, requested_limit)
    if engine_limit and engine_limit > 0:
        return engine_limit
    return None


def generation_sequence(limit: Optional[int]) -> Iterable[int]:
    """Yield generation counters respecting ``limit`` if provided."""

    if limit is None:
        return itertools.count()
    return range(limit)


@dataclass
class EvolutionConsoleUI:
    """Interactive console runner for the evolutionary trading simulator."""

    engine: EvolutionEngine
    tickers: List[str]
    population: List[PortfolioMember] = field(init=False)
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

        limit = determine_generation_limit(self.engine.config.generations, max_generations)
        for gen in generation_sequence(limit):
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
        top_values = [m.top_percent_gain for m in self.history]
        top10_values = [m.top10_mean_percent for m in self.history]
        top20_values = [m.top20_mean_percent for m in self.history]

        plt.figure(figsize=(8, 4.5))
        plt.plot(generations, top_values, label="Top Member")
        plt.plot(generations, top10_values, label="Top 10% Avg")
        plt.plot(generations, top20_values, label="Top 20% Avg")
        plt.xlabel("Generation")
        plt.ylabel("Percent Gain (%)")
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
        best = report.best_member
        if best is None:
            print("No performance data available.")
            return
        print("=" * 60)
        print(f"Generation {report.generation + 1}")
        print(f"Top percent gain: {metrics.top_percent_gain:.2f}%")
        print(f"Top 10% average gain: {metrics.top10_mean_percent:.2f}%")
        print(f"Top 20% average gain: {metrics.top20_mean_percent:.2f}%")
        print(f"Population average gain: {metrics.average_percent_gain:.2f}%")
        print(f"Best member final equity: {best.final_equity:,.2f}")
        print(f"Best member percent gain: {best.percent_gain:.2f}%")
        print("Member gains (sorted by performance):")
        for idx, performance in enumerate(report.performances, start=1):
            print(
                f"  #{idx}: {performance.percent_gain:.2f}% | Final Equity {performance.final_equity:,.2f}"
            )
        print("Best member formulas:")
        for line in best.member.describe_formulas():
            print(f"  {line}")
        print("-" * 60)


@dataclass
class EvolutionTkUI:
    """Tkinter-based UI showing live updates of the top performing formula."""

    engine: EvolutionEngine
    tickers: List[str]
    population: List[PortfolioMember] = field(init=False)
    generation: int = field(default=0, init=False)
    history: List[PopulationMetrics] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if tk is None:
            raise RuntimeError("tkinter is required for EvolutionTkUI but is not available")
        self.population = self.engine.initialize_population(self.tickers)
        self._stop_event = threading.Event()
        self._report_queue: queue.Queue[GenerationReport | None] = queue.Queue()
        self._latest_report: GenerationReport | None = None
        self._equity_points: List[tuple[int, float, float]] = []

        self.root = tk.Tk()
        self.root.title("STCK Evolution Monitor")
        self.root.configure(padx=12, pady=12)
        self.status_text = tk.StringVar()
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_text,
            justify="left",
            anchor="w",
            font=("Courier", 10),
            padx=8,
            pady=8,
        )
        self.status_label.pack(fill="x", expand=False)
        self._plot_enabled = self._init_chart()
        message = "Initializing simulation..."
        if not self._plot_enabled:
            message += "\nInstall matplotlib for live performance charts."
        self.status_text.set(message)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def start(self, max_generations: Optional[int] = None) -> None:
        """Start the simulation and open the Tkinter window."""

        worker = threading.Thread(
            target=self._run_loop,
            kwargs={"max_generations": max_generations},
            daemon=True,
        )
        worker.start()
        self.root.after(200, self._process_reports)
        self.root.mainloop()

    # Internal helpers ---------------------------------------------------------------

    def _init_chart(self) -> bool:
        try:  # pragma: no cover - optional dependency
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except ImportError:  # pragma: no cover - optional dependency
            self._figure = None
            self._axis = None
            self._canvas = None
            self._line_top = None
            self._line_avg = None
            return False

        self._figure = Figure(figsize=(6, 3), dpi=100)
        self._axis = self._figure.add_subplot(111)
        self._axis.set_title("Top percent gain by generation")
        self._axis.set_xlabel("Generation")
        self._axis.set_ylabel("Percent gain (%)")
        (self._line_top,) = self._axis.plot([], [], color="#1f77b4", label="Top gain")
        (self._line_avg,) = self._axis.plot([], [], color="#ff7f0e", label="Population avg")
        self._axis.legend(loc="upper left")
        self._canvas = FigureCanvasTkAgg(self._figure, master=self.root)
        widget = self._canvas.get_tk_widget()
        widget.pack(fill="both", expand=True, pady=(12, 0))
        self._canvas.draw()
        return True

    def _run_loop(self, max_generations: Optional[int]) -> None:
        limit = determine_generation_limit(self.engine.config.generations, max_generations)
        for gen in generation_sequence(limit):
            if self._stop_event.is_set():
                break
            self.population, report = self.engine.evolve(self.population, generation=gen)
            self.history.append(report.metrics)
            self._report_queue.put(report)
            self.generation = gen + 1
        self._report_queue.put(None)

    def _process_reports(self) -> None:
        try:
            while True:
                report = self._report_queue.get_nowait()
                if report is None:
                    self.status_text.set(self.status_text.get() + "\nSimulation complete.")
                    self._stop_event.set()
                    self.root.after(500, self.root.destroy)
                    return
                self._latest_report = report
                self.status_text.set(self._format_report(report))
                self._record_equity(report)
                self._update_plot()
        except queue.Empty:
            pass
        if not self._stop_event.is_set():
            self.root.after(200, self._process_reports)

    def _format_report(self, report: GenerationReport) -> str:
        metrics = report.metrics
        best = report.best_member
        if best is None:
            return "No performance data yet."
        lines = [
            f"Generation: {report.generation + 1}",
            f"Top percent gain: {metrics.top_percent_gain:.2f}%",
            f"Top 10% average gain: {metrics.top10_mean_percent:.2f}%",
            f"Top 20% average gain: {metrics.top20_mean_percent:.2f}%",
            f"Population average gain: {metrics.average_percent_gain:.2f}%",
            f"Best member final equity: {best.final_equity:,.2f}",
            f"Best member percent gain: {best.percent_gain:.2f}%",
            "",
            "Member gains:",
        ]
        for idx, performance in enumerate(report.performances, start=1):
            lines.append(
                f"#{idx}: {performance.percent_gain:.2f}% | Final {performance.final_equity:,.2f}"
            )
        lines.extend(
            [
                "",
                "Best Member Allocations:",
                *best.member.describe_formulas(),
            ]
        )
        return "\n".join(lines)

    def _record_equity(self, report: GenerationReport) -> None:
        if not self._plot_enabled:
            return
        metrics = report.metrics
        self._equity_points.append(
            (
                report.generation + 1,
                metrics.top_percent_gain,
                metrics.average_percent_gain,
            )
        )

    def _update_plot(self) -> None:
        if not self._plot_enabled:
            return
        xs = [point[0] for point in self._equity_points]
        top_values = [point[1] for point in self._equity_points]
        average_values = [point[2] for point in self._equity_points]
        self._line_top.set_data(xs, top_values)
        self._line_avg.set_data(xs, average_values)
        self._axis.relim()
        self._axis.autoscale_view()
        self._canvas.draw_idle()

    def _on_close(self) -> None:
        self._stop_event.set()
        self.root.destroy()
