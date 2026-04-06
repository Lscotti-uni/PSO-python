"""Gradio dashboard for inspecting stored PSO runs and analysis outputs."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt

from pso_lab.experiments.analysis import load_summary_rows
from pso_lab.io import load_history_csv


def _discover_rows(results_root: str) -> list[dict]:
    rows = load_summary_rows(results_root)
    rows.sort(key=lambda row: row.get("timestamp_utc", ""), reverse=True)
    for row in rows:
        row["_run_dir"] = str(Path(results_root).joinpath())  # placeholder overwritten below if needed
    return rows


def _load_runs(results_root: str) -> tuple[list[dict], list[list[object]], list[str]]:
    rows = []
    table_rows = []
    run_ids = []
    for summary_path in sorted(Path(results_root).rglob("summary.json")):
        summary = __import__("json").loads(summary_path.read_text(encoding="utf-8"))
        summary["_run_dir"] = str(summary_path.parent)
        rows.append(summary)
    rows.sort(key=lambda item: item.get("timestamp_utc", ""), reverse=True)

    for row in rows:
        config = row["config"]
        table_rows.append(
            [
                row["run_id"],
                row["objective"],
                row["strategy"],
                row["update_mode"],
                config["dimensions"],
                row["best_value"],
                row["metrics"]["total_run_time"],
                row["stop_reason"],
            ]
        )
        run_ids.append(row["run_id"])
    return rows, table_rows, run_ids


def _find_run(rows: list[dict], run_id: str) -> dict | None:
    for row in rows:
        if row["run_id"] == run_id:
            return row
    return None


def _make_convergence_plot(run_dir: str):
    history = load_history_csv(Path(run_dir) / "history.csv")
    iterations = [row["iteration"] for row in history]
    best = [row["best_fitness"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(iterations, best, color="#005f73", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness")
    ax.set_yscale("log")
    ax.set_title("Convergence")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _collect_media(run_dir: str) -> list[str]:
    run_path = Path(run_dir)
    media = []
    for pattern in ["*.png", "*.gif", "frames/*.png"]:
        media.extend(str(path) for path in sorted(run_path.glob(pattern))[:24])
    return media


def _artifact_markdown(run_dir: str) -> str:
    run_path = Path(run_dir)
    lines = ["### Artifacts", ""]
    for path in sorted(run_path.glob("*")):
        if path.is_file():
            lines.append(f"- `{path.name}`")
    return "\n".join(lines)


def build_dashboard(default_results_root: str = "results") -> gr.Blocks:
    with gr.Blocks(title="PSO Lab Dashboard") as demo:
        rows_state = gr.State([])

        gr.Markdown(
            """
            # PSO Lab Dashboard

            Navega ejecuciones guardadas, revisa métricas, abre artefactos y observa la convergencia.
            """
        )

        with gr.Row():
            results_root = gr.Textbox(label="Results root", value=default_results_root)
            refresh_button = gr.Button("Refresh", variant="primary")

        runs_table = gr.Dataframe(
            headers=["run_id", "objective", "strategy", "update_mode", "dimensions", "best_value", "total_time", "stop_reason"],
            interactive=False,
            wrap=True,
            row_count=10,
            column_count=(8, "fixed"),
            label="Saved runs",
        )
        run_selector = gr.Dropdown(label="Select run", choices=[])

        with gr.Row():
            summary_json = gr.JSON(label="Run summary")
            convergence_plot = gr.Plot(label="Convergence curve")

        with gr.Row():
            artifacts_md = gr.Markdown()
            media_gallery = gr.Gallery(label="Visual artifacts", columns=3, height=420)

        analysis_gallery = gr.Gallery(label="Analysis plots", columns=3, height=320)

        def refresh(root: str):
            rows, table_rows, run_ids = _load_runs(root)
            analysis_images = [str(path) for path in sorted((Path(root) / "analysis").glob("*.png"))]
            selected = run_ids[0] if run_ids else None
            summary = _find_run(rows, selected) if selected else None
            plot = _make_convergence_plot(summary["_run_dir"]) if summary else None
            artifacts = _artifact_markdown(summary["_run_dir"]) if summary else "No run selected."
            media = _collect_media(summary["_run_dir"]) if summary else []
            return rows, table_rows, gr.update(choices=run_ids, value=selected), summary, plot, artifacts, media, analysis_images

        def on_select(run_id: str, rows: list[dict]):
            summary = _find_run(rows, run_id)
            if not summary:
                return None, None, "Run not found.", []
            plot = _make_convergence_plot(summary["_run_dir"])
            artifacts = _artifact_markdown(summary["_run_dir"])
            media = _collect_media(summary["_run_dir"])
            return summary, plot, artifacts, media

        refresh_button.click(
            refresh,
            inputs=[results_root],
            outputs=[rows_state, runs_table, run_selector, summary_json, convergence_plot, artifacts_md, media_gallery, analysis_gallery],
        )
        run_selector.change(
            on_select,
            inputs=[run_selector, rows_state],
            outputs=[summary_json, convergence_plot, artifacts_md, media_gallery],
        )
        demo.load(
            refresh,
            inputs=[results_root],
            outputs=[rows_state, runs_table, run_selector, summary_json, convergence_plot, artifacts_md, media_gallery, analysis_gallery],
        )
    return demo


def launch_dashboard(results_root: str = "results", share: bool = False) -> None:
    demo = build_dashboard(results_root)
    demo.launch(share=share)
