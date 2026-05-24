"""Gradio dashboard for exploring PSO interactively.

Gradio is an optional dependency. The module imports it lazily so the rest of
the project keeps working when Gradio is not installed.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pso_lab.core import PSOConfig
from pso_lab.experiments import run_single_experiment
from pso_lab.objectives import list_objectives


def _convergence_figure(history: list[dict]) -> "matplotlib.figure.Figure":
    fig, ax = plt.subplots(figsize=(7, 4))
    iterations = [row["iteration"] for row in history]
    best = [row["best_fitness"] for row in history]
    mean = [row["mean_fitness"] for row in history]
    ax.plot(iterations, best, label="Global best", color="#005f73", linewidth=2)
    ax.plot(iterations, mean, label="Swarm mean", color="#94d2bd", linewidth=1, alpha=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("PSO convergence")
    fig.tight_layout()
    return fig


def _run_pso_for_ui(
    objective: str,
    dimensions: int,
    swarm_size: int,
    iterations: int,
    strategy: str,
    topology: str,
    inertia: float,
    cognitive: float,
    social: float,
    seed: int,
):
    config = PSOConfig(
        objective=objective,
        dimensions=int(dimensions),
        swarm_size=int(swarm_size),
        iterations=int(iterations),
        strategy=strategy,
        topology=topology,
        inertia=float(inertia),
        cognitive=float(cognitive),
        social=float(social),
        seed=int(seed),
        log_every=999,
        enable_early_stopping=False,
    )
    info = run_single_experiment(
        config,
        output_dir=str(Path(tempfile.gettempdir()) / "pso_dashboard"),
        run_prefix="dashboard",
        log_level="ERROR",
        save_results=False,
    )
    result = info["result"]
    summary = info["summary"]
    metrics = summary["metrics"]

    fig = _convergence_figure(result.history)
    info_md = (
        f"**Best fitness:** {result.best_value:.6e}\n\n"
        f"**Stop reason:** {result.stop_reason}\n\n"
        f"**Total time:** {metrics['total_run_time']:.4f} s\n\n"
        f"**Eval time:** {metrics['total_eval_time']:.4f} s\n\n"
        f"**Overhead time:** {metrics['total_overhead_time']:.4f} s\n\n"
        f"**Best position:** `{np.array2string(result.best_position, precision=4, separator=', ')}`"
    )
    return fig, info_md


def build_interface():
    import gradio as gr

    objectives = list_objectives()
    strategies = ["sequential", "thread", "process", "async", "vectorized", "joblib"]
    topologies = ["global", "ring", "von_neumann"]

    with gr.Blocks(title="PSO Lab Dashboard") as interface:
        gr.Markdown("# PSO Lab Dashboard\nRun a PSO experiment and inspect convergence in real time.")
        with gr.Row():
            with gr.Column():
                objective = gr.Dropdown(objectives, value="sphere", label="Objective")
                dimensions = gr.Slider(2, 30, value=10, step=1, label="Dimensions")
                swarm_size = gr.Slider(8, 80, value=30, step=2, label="Swarm size")
                iterations = gr.Slider(20, 400, value=120, step=10, label="Iterations")
                strategy = gr.Dropdown(strategies, value="vectorized", label="Strategy")
                topology = gr.Dropdown(topologies, value="global", label="Topology")
                inertia = gr.Slider(0.1, 1.2, value=0.7298, step=0.01, label="Inertia (w)")
                cognitive = gr.Slider(0.1, 3.0, value=1.49618, step=0.01, label="Cognitive (c1)")
                social = gr.Slider(0.1, 3.0, value=1.49618, step=0.01, label="Social (c2)")
                seed = gr.Number(value=123, precision=0, label="Seed")
                run_button = gr.Button("Run PSO", variant="primary")
            with gr.Column():
                plot = gr.Plot(label="Convergence")
                info = gr.Markdown()

        run_button.click(
            _run_pso_for_ui,
            inputs=[objective, dimensions, swarm_size, iterations, strategy, topology, inertia, cognitive, social, seed],
            outputs=[plot, info],
        )

    return interface


def launch(server_port: int = 7860, share: bool = False) -> None:
    interface = build_interface()
    interface.launch(server_port=server_port, share=share)
