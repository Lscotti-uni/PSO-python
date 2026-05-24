"""Launch the Gradio dashboard for interactive PSO exploration."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the PSO Lab Gradio dashboard.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Publish a temporary public URL.")
    args = parser.parse_args()

    try:
        from pso_lab.dashboard import launch
    except ImportError as exc:
        raise SystemExit(
            "Gradio is required for the dashboard. Install it with `pip install gradio`."
        ) from exc

    launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
