"""CLI launcher for the local Gradio dashboard."""

import argparse

from pso_lab.dashboard import launch_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Gradio dashboard for PSO results.")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    launch_dashboard(results_root=args.results_root, share=args.share)


if __name__ == "__main__":
    main()
