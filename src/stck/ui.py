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
    format_member_portfolio,
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


def _pluralize(count: int, noun: str) -> str:
    suffix = "s" if count != 1 else ""
    return f"{count} {noun}{suffix}"


def _build_generation_lines(report: GenerationReport) -> List[str]:
    metrics = report.metrics
    best = report.best_member
    if best is None:
        return ["No performance data yet."]

    lines = [
        f"Generation {report.generation + 1}",
        f"Top final cash: {metrics.top_final_cash:,.2f}",
        f"Top 10% average cash: {metrics.top10_mean_final_cash:,.2f}",
        f"Top 20% average cash: {metrics.top20_mean_final_cash:,.2f}",
        f"Population average cash: {metrics.average_final_cash:,.2f}",
        f"Best member final cash: {best.final_cash:,.2f}",
        f"Best member cash gain: {best.cash_percent_gain:.2f}%",
    ]

    if report.transition is not None:
        summary = report.transition
        lines.extend(
            [
                "",
                "Transition Summary:",
                (
                    f"  Preserved top 10%: "
                    f"{_pluralize(summary.top_preserved_count, 'member')}"
                ),
                (
                    f"  Culled bottom 10%: "
                    f"{_pluralize(summary.bottom_culled_count, 'member')}"
                ),
                f"  Middle cohort: {summary.middle_strategy}",
                (
                    f"  Survivors carried forward: "
                    f"{_pluralize(summary.survivor_count, 'member')}"
                ),
                (
                    f"  Mutated clones created: "
                    f"{_pluralize(summary.clones_created, 'clone')}"
                ),
            ]
        )

    lines.extend(["", "Top performers (sorted by final cash):"])
    for idx, performance in enumerate(report.performances[:5], start=1):
        lines.append(
            (
                f"  #{idx}: Final cash {performance.final_cash:,.2f} | "
                f"Cash gain {performance.cash_percent_gain:.2f}% | "
                f"Final equity {performance.final_equity:,.2f}"
            )
        )
    if len(report.performances) > 5:
        lines.append(
            f"  ... and {len(report.performances) - 5} more members."
        )

    lines.append("")
    lines.append("Best member portfolio:")
    holdings = format_member_portfolio(best.member)
    if holdings:
        for line in holdings:
            lines.append(f"  {line}")
    else:
        lines.append("  (no holdings)")

    result = best.backtest
    if result.cash_curve:
        final_cash = result.cash_curve[-1]
        min_cash = min(result.cash_curve)
        max_cash = max(result.cash_curve)
        lines.extend(
            [
                "",
                "Best member cash utilisation:",
                f"  Final cash: {final_cash:,.2f}",
                f"  Cash range: {min_cash:,.2f} – {max_cash:,.2f}",
            ]
        )

    return lines


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
        top_values = [m.top_final_cash for m in self.history]
        top10_values = [m.top10_mean_final_cash for m in self.history]
        top20_values = [m.top20_mean_final_cash for m in self.history]

        plt.figure(figsize=(8, 4.5))
        plt.plot(generations, top_values, label="Top Member")
        plt.plot(generations, top10_values, label="Top 10% Avg")
        plt.plot(generations, top20_values, label="Top 20% Avg")
        plt.xlabel("Generation")
        plt.ylabel("Final Cash")
        plt.title("Evolution Final Cash by Generation")
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
        print("=" * 60)
        for line in _build_generation_lines(report):
            print(line)
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
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._report_queue: queue.Queue[GenerationReport | None] = queue.Queue()
        self._latest_report: GenerationReport | None = None
        self._cash_points: List[tuple[int, float, float]] = []
        self._detail_window: tk.Toplevel | None = None
        self._detail_text: tk.Text | None = None
        self._chart_window: tk.Toplevel | None = None
        self._figure = None
        self._axis = None
        self._canvas = None
        self._line_top = None
        self._line_avg = None
        self._plot_enabled = False
        self._top_stock_window: tk.Toplevel | None = None
        self._top_stock_canvas = None
        self._top_stock_figure = None
        self._top_stock_axes = None
        self._top_stock_label_var: tk.StringVar | None = None
        self._top_stock_enabled = False

        self.root = tk.Tk()
        self.root.title("STCK Evolution Monitor")
        self.root.configure(padx=12, pady=12)
        self.status_text = tk.StringVar()
        self._top_stock_label_var = tk.StringVar(value="Top stock data unavailable.")
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
        controls = tk.Frame(self.root)
        controls.pack(fill="x", expand=False, pady=(8, 0))
        self.pause_button = tk.Button(
            controls,
            text="Pause",
            width=10,
            command=self.toggle_pause,
        )
        self.pause_button.pack(side="left")
        self.details_button = tk.Button(
            controls,
            text="Show details",
            command=self.show_top_performers,
        )
        self.details_button.pack(side="left", padx=(8, 0))
        self.chart_button = tk.Button(
            controls,
            text="Show chart",
            command=self._ensure_chart_window,
        )
        self.chart_button.pack(side="left", padx=(8, 0))
        self.top_stock_button = tk.Button(
            controls,
            text="Top stock charts",
            command=self._show_top_stock_charts,
        )
        self.top_stock_button.pack(side="left", padx=(8, 0))

        self._ensure_chart_window()
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

    def pause(self) -> None:
        if self._pause_event.is_set():
            self._pause_event.clear()
            self.pause_button.configure(text="Resume")
            self.status_text.set(self.status_text.get() + "\nSimulation paused.")

    def resume(self) -> None:
        if not self._pause_event.is_set():
            self._pause_event.set()
            self.pause_button.configure(text="Pause")
            self.status_text.set(self.status_text.get() + "\nResuming simulation...")

    def toggle_pause(self) -> None:
        if self._pause_event.is_set():
            self.pause()
        else:
            self.resume()

    def _ensure_chart_window(self) -> None:
        if tk is None:
            return
        if (
            self._plot_enabled
            and self._chart_window is not None
            and self._chart_window.winfo_exists()
        ):
            self._chart_window.deiconify()
            self._chart_window.lift()
            return
        self._plot_enabled = self._init_chart()
        if self._plot_enabled:
            self._update_plot()

    def _show_top_stock_charts(self) -> None:
        if self._latest_report is None:
            self.status_text.set(
                self.status_text.get() + "\nTop stock data is not available yet."
            )
            return
        self._ensure_top_stock_window()
        self._update_top_stock_charts()

    def _ensure_top_stock_window(self) -> None:
        if tk is None:
            return
        if (
            self._top_stock_enabled
            and self._top_stock_window is not None
            and self._top_stock_window.winfo_exists()
        ):
            self._top_stock_window.deiconify()
            self._top_stock_window.lift()
            return
        self._top_stock_enabled = self._init_top_stock_window()

    def _init_top_stock_window(self) -> bool:
        try:  # pragma: no cover - optional dependency
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except ImportError:  # pragma: no cover - optional dependency
            self._top_stock_window = None
            self._top_stock_canvas = None
            self._top_stock_figure = None
            self._top_stock_axes = None
            if self.status_text is not None:
                self.status_text.set(
                    self.status_text.get()
                    + "\nInstall matplotlib to view top stock charts."
                )
            return False

        if self._top_stock_window is not None and self._top_stock_window.winfo_exists():
            self._top_stock_window.destroy()

        self._top_stock_figure = Figure(figsize=(7, 5), dpi=100)
        ax_equity = self._top_stock_figure.add_subplot(211)
        ax_alloc = self._top_stock_figure.add_subplot(212)
        ax_equity.set_title("Total portfolio value (5-year window)")
        ax_equity.set_xlabel("Step")
        ax_equity.set_ylabel("Value")
        ax_alloc.set_title("Portfolio allocation percentages")
        ax_alloc.set_xlabel("Step")
        ax_alloc.set_ylabel("Allocation %")
        self._top_stock_axes = (ax_equity, ax_alloc)

        self._top_stock_window = tk.Toplevel(self.root)
        self._top_stock_window.title("Top stock overview")
        self._top_stock_window.geometry("720x520")
        self._top_stock_window.protocol("WM_DELETE_WINDOW", self._on_top_stock_close)

        label = tk.Label(
            self._top_stock_window,
            textvariable=self._top_stock_label_var,
            anchor="w",
            justify="left",
            font=("Courier", 10),
            padx=8,
            pady=4,
        )
        label.pack(fill="x", expand=False)

        self._top_stock_canvas = FigureCanvasTkAgg(
            self._top_stock_figure, master=self._top_stock_window
        )
        widget = self._top_stock_canvas.get_tk_widget()
        widget.pack(fill="both", expand=True, padx=8, pady=8)
        self._top_stock_canvas.draw()
        return True

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
            self._chart_window = None
            return False

        if self._chart_window is not None and self._chart_window.winfo_exists():
            self._chart_window.destroy()

        self._figure = Figure(figsize=(7, 3.5), dpi=100)
        self._axis = self._figure.add_subplot(111)
        self._axis.set_title("Top final cash by generation")
        self._axis.set_xlabel("Generation")
        self._axis.set_ylabel("Final cash")
        (self._line_top,) = self._axis.plot([], [], color="#1f77b4", label="Top cash")
        (self._line_avg,) = self._axis.plot([], [], color="#ff7f0e", label="Population avg cash")
        self._axis.legend(loc="upper left")

        self._chart_window = tk.Toplevel(self.root)
        self._chart_window.title("Performance chart")
        self._chart_window.geometry("720x360")
        self._chart_window.protocol("WM_DELETE_WINDOW", self._on_chart_close)
        self._canvas = FigureCanvasTkAgg(self._figure, master=self._chart_window)
        widget = self._canvas.get_tk_widget()
        widget.pack(fill="both", expand=True, padx=8, pady=8)
        self._canvas.draw()
        return True

    def _wait_if_paused(self) -> bool:
        while not self._pause_event.is_set():
            if self._stop_event.is_set():
                return False
            time.sleep(0.1)
        return not self._stop_event.is_set()

    def show_top_performers(self) -> None:
        if self._latest_report is None:
            self.status_text.set(self.status_text.get() + "\nTop performer data is not available yet.")
            return

        if self._detail_window is None or not self._detail_window.winfo_exists():
            self._detail_window = tk.Toplevel(self.root)
            self._detail_window.title("Top performer details")
            self._detail_window.geometry("760x420")
            self._detail_window.protocol("WM_DELETE_WINDOW", self._close_detail_window)

            container = tk.Frame(self._detail_window)
            container.pack(fill="both", expand=True, padx=8, pady=8)
            scrollbar = tk.Scrollbar(container)
            scrollbar.pack(side="right", fill="y")
            self._detail_text = tk.Text(
                container,
                wrap="none",
                font=("Courier", 10),
                state="disabled",
            )
            self._detail_text.pack(fill="both", expand=True)
            self._detail_text.configure(yscrollcommand=scrollbar.set)
            scrollbar.configure(command=self._detail_text.yview)
        else:
            self._detail_window.deiconify()
            self._detail_window.lift()

        self._refresh_detail_window()

    def _refresh_detail_window(self) -> None:
        if self._detail_window is None or not self._detail_window.winfo_exists():
            self._detail_window = None
            self._detail_text = None
            return
        if self._detail_text is None:
            return

        content = self._build_top_performers_text()
        self._detail_text.configure(state="normal")
        self._detail_text.delete("1.0", tk.END)
        self._detail_text.insert(tk.END, content)
        self._detail_text.configure(state="disabled")

    def _build_top_performers_text(self) -> str:
        report = self._latest_report
        if report is None or not report.performances:
            return "No performance data yet."

        lines: List[str] = []
        for idx, performance in enumerate(report.performances[:5], start=1):
            lines.append(
                (
                    f"#{idx}: Final cash {performance.final_cash:,.2f} | "
                    f"Cash gain {performance.cash_percent_gain:.2f}% | "
                    f"Final equity {performance.final_equity:,.2f}"
                )
            )
            result = performance.backtest
            lines.append(
                (
                    "    Score: "
                    f"{performance.score:,.2f} | Complexity: {performance.complexity} | "
                    f"Penalty: {performance.complexity_penalty:,.2f}"
                )
            )
            if result.cash_curve:
                final_cash = result.cash_curve[-1]
                min_cash = min(result.cash_curve)
                max_cash = max(result.cash_curve)
                lines.append(
                    (
                        "    Cash range: "
                        f"{min_cash:,.2f} – {max_cash:,.2f} | Final cash: {final_cash:,.2f}"
                    )
                )
            if result.equity_curve:
                final_equity = result.equity_curve[-1]
                lines.append(f"    Final equity: {final_equity:,.2f}")
            if idx == 1:
                lines.append("    Holdings:")
                holdings = format_member_portfolio(performance.member)
                if holdings:
                    for holding in holdings:
                        lines.append(f"      {holding}")
                else:
                    lines.append("      (no holdings)")
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def _close_detail_window(self) -> None:
        if self._detail_window is not None and self._detail_window.winfo_exists():
            self._detail_window.destroy()
        self._detail_window = None
        self._detail_text = None

    def _run_loop(self, max_generations: Optional[int]) -> None:
        limit = determine_generation_limit(self.engine.config.generations, max_generations)
        for gen in generation_sequence(limit):
            if self._stop_event.is_set():
                break
            if not self._wait_if_paused():
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
                self._refresh_detail_window()
                self._record_cash(report)
                self._update_plot()
                self._update_top_stock_charts()
        except queue.Empty:
            pass
        if not self._stop_event.is_set():
            self.root.after(200, self._process_reports)

    def _format_report(self, report: GenerationReport) -> str:
        lines = _build_generation_lines(report)
        if not lines:
            return "No performance data yet."
        # Replace the header to match previous colon-separated style for Tk labels.
        if lines and lines[0].startswith("Generation "):
            lines[0] = lines[0].replace("Generation ", "Generation: ")
        return "\n".join(lines)

    def _record_cash(self, report: GenerationReport) -> None:
        metrics = report.metrics
        self._cash_points.append(
            (
                report.generation + 1,
                metrics.top_final_cash,
                metrics.average_final_cash,
            )
        )

    def _update_plot(self) -> None:
        if not self._plot_enabled or self._canvas is None or self._axis is None:
            return
        xs = [point[0] for point in self._cash_points]
        top_values = [point[1] for point in self._cash_points]
        average_values = [point[2] for point in self._cash_points]
        self._line_top.set_data(xs, top_values)
        self._line_avg.set_data(xs, average_values)
        self._axis.relim()
        self._axis.autoscale_view()
        self._canvas.draw_idle()

    def _update_top_stock_charts(self) -> None:
        if (
            not self._top_stock_enabled
            or self._top_stock_canvas is None
            or self._top_stock_axes is None
            or self._top_stock_label_var is None
        ):
            return
        report = self._latest_report
        if report is None or report.best_member is None:
            self._top_stock_label_var.set("Top stock data unavailable.")
            return
        result = report.best_member.backtest
        if not result.equity_curve:
            self._top_stock_label_var.set("Top stock data unavailable.")
            return

        ax_equity, ax_alloc = self._top_stock_axes
        ax_equity.clear()
        ax_alloc.clear()
        xs = list(range(len(result.equity_curve)))
        ax_equity.set_title("Total portfolio value (5-year window)")
        ax_equity.set_xlabel("Step")
        ax_equity.set_ylabel("Value")
        ax_equity.plot(xs, result.equity_curve, label="Equity", color="#1f77b4")
        ax_equity.plot(xs, result.cash_curve, label="Cash", color="#ff7f0e")
        ax_equity.legend(loc="upper left")

        ax_alloc.set_title("Portfolio allocation percentages")
        ax_alloc.set_xlabel("Step")
        ax_alloc.set_ylabel("Allocation %")
        if result.value_percentages:
            series = [
                [step.get(ticker, 0.0) * 100.0 for step in result.value_percentages]
                for ticker in result.tickers
            ]
            if any(any(values) for values in series):
                ax_alloc.stackplot(xs, series, labels=result.tickers)
                ax_alloc.set_ylim(0, 100)
                ax_alloc.legend(loc="upper left", ncol=2, fontsize=8)
        ax_alloc.margins(x=0)

        top_stock = result.top_stock()
        if top_stock is not None:
            ticker, value = top_stock
            percent = result.value_percentages[-1].get(ticker, 0.0) * 100.0
            self._top_stock_label_var.set(
                f"Top performing stock: {ticker} | Value {value:,.2f} | Share {percent:.2f}%"
            )
        else:
            self._top_stock_label_var.set("Top performing stock: (no holdings)")

        self._top_stock_canvas.draw_idle()

    def _on_close(self) -> None:
        self._stop_event.set()
        self._pause_event.set()
        if self._chart_window is not None and self._chart_window.winfo_exists():
            self._chart_window.destroy()
        if self._detail_window is not None and self._detail_window.winfo_exists():
            self._detail_window.destroy()
        if self._top_stock_window is not None and self._top_stock_window.winfo_exists():
            self._top_stock_window.destroy()
        self.root.destroy()

    def _on_chart_close(self) -> None:
        if self._chart_window is not None and self._chart_window.winfo_exists():
            self._chart_window.destroy()
        self._chart_window = None
        self._canvas = None
        self._figure = None
        self._axis = None
        self._line_top = None
        self._line_avg = None
        self._plot_enabled = False

    def _on_top_stock_close(self) -> None:
        if self._top_stock_window is not None and self._top_stock_window.winfo_exists():
            self._top_stock_window.destroy()
        self._top_stock_window = None
        self._top_stock_canvas = None
        self._top_stock_figure = None
        self._top_stock_axes = None
        self._top_stock_enabled = False
