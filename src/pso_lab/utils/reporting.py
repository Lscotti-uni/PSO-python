"""Console formatting helpers used by the CLI scripts."""

from __future__ import annotations

from collections.abc import Iterable

from prettytable import PrettyTable


def build_table(rows: Iterable[dict[str, object]], columns: list[str]) -> PrettyTable:
    table = PrettyTable()
    table.field_names = columns
    for row in rows:
        rendered_row = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                if abs(value) < 1e-3 and value != 0:
                    rendered_row.append(f"{value:.3e}")
                else:
                    rendered_row.append(f"{value:.4f}")
            else:
                rendered_row.append(value)
        table.add_row(rendered_row)
    return table
