"""CLI wrapper that builds the final protocol report from saved outputs."""

import argparse

from pso_lab.experiments.report import generate_protocol_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the detailed markdown report for a protocol run.")
    parser.add_argument("--protocol-root", type=str, default="results/protocol_full")
    parser.add_argument("--output", type=str, default="docs/final_report.md")
    args = parser.parse_args()

    path = generate_protocol_report(args.protocol_root, args.output)
    print(f"Report written to {path.resolve()}")


if __name__ == "__main__":
    main()
