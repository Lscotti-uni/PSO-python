"""Report-generation helpers for protocol summaries and final deliverables."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from statistics import mean
from typing import Any


def _load_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    parsed[key] = value
                    continue
                try:
                    parsed[key] = int(value)
                    continue
                except ValueError:
                    pass
                try:
                    parsed[key] = float(value)
                    continue
                except ValueError:
                    pass
                parsed[key] = value
            rows.append(parsed)
        return rows


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No data available._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                if abs(value) < 1e-3 and value != 0:
                    values.append(f"{value:.3e}")
                else:
                    values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _group_mean(rows: list[dict[str, Any]], key: str, metric: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(float(row[metric]))
    output = [{key: name, f"mean_{metric}": mean(values)} for name, values in grouped.items()]
    output.sort(key=lambda item: item[f"mean_{metric}"])
    return output


def _best_grid_rows(grid_root: Path) -> list[dict[str, Any]]:
    best_rows = []
    for variant_dir in sorted(path for path in grid_root.iterdir() if path.is_dir()):
        summary_path = variant_dir / "grid_search_summary.csv"
        if not summary_path.exists():
            continue
        rows = _load_csv(summary_path)
        if rows:
            row = dict(rows[0])
            row["variant"] = variant_dir.name
            best_rows.append(row)
    best_rows.sort(key=lambda row: row["variant"])
    return best_rows


def _speedup_rows(benchmark_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines: dict[int, list[float]] = {}
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in benchmark_runs:
        variant = str(row["variant"])
        dimensions = int(row["dimensions"])
        grouped.setdefault((variant, dimensions), []).append(float(row["total_time"]))
        if variant == "V0":
            baselines.setdefault(dimensions, []).append(float(row["total_time"]))

    output = []
    for (variant, dimensions), values in sorted(grouped.items()):
        baseline_time = mean(baselines[dimensions])
        output.append(
            {
                "variant": variant,
                "dimensions": dimensions,
                "mean_total_time": mean(values),
                "speedup_vs_V0": baseline_time / mean(values),
            }
        )
    return output


def _best_variant_per_case(benchmark_summary: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in benchmark_summary:
        key = (str(row["objective"]), int(row["dimensions"]))
        grouped.setdefault(key, []).append(row)

    winners = []
    for (objective, dimensions), rows in sorted(grouped.items()):
        best = min(rows, key=lambda item: float(item[metric]))
        winners.append(
            {
                "objective": objective,
                "dimensions": dimensions,
                "winner_variant": best["variant"],
                metric: best[metric],
            }
        )
    return winners


def generate_protocol_report(protocol_root: str | Path, output_path: str | Path) -> Path:
    protocol_root = Path(protocol_root)
    output_path = Path(output_path)
    benchmark_root = protocol_root / "benchmarks"
    grid_root = protocol_root / "grid_search"
    analysis_root = protocol_root / "analysis"

    benchmark_summary = _load_csv(benchmark_root / "benchmark_summary.csv")
    benchmark_runs = _load_csv(benchmark_root / "benchmark_runs.csv")
    best_grid = _best_grid_rows(grid_root)
    speedups = _speedup_rows(benchmark_runs)
    by_variant = _group_mean(benchmark_runs, "variant", "best_value")
    fastest_per_case = _best_variant_per_case(benchmark_summary, "mean_total_time")
    best_quality_per_case = _best_variant_per_case(benchmark_summary, "mean_best_value")

    overall_fastest = min(speedups, key=lambda row: float(row["mean_total_time"]))
    overall_slowest = max(speedups, key=lambda row: float(row["mean_total_time"]))
    best_quality_variant = min(by_variant, key=lambda row: float(row["mean_best_value"]))

    benchmark_plot = Path(os.path.relpath(analysis_root / "benchmark_mean_convergence.png", output_path.parent)).as_posix()
    speedup_plot = Path(os.path.relpath(analysis_root / "benchmark_speedup.png", output_path.parent)).as_posix()
    boxplot = Path(os.path.relpath(analysis_root / "benchmark_final_fitness_boxplot.png", output_path.parent)).as_posix()

    report = f"""# Informe Final: Laboratorio PSO

## 1. Resumen Ejecutivo

Este proyecto implementa un laboratorio completo de Particle Swarm Optimization (PSO) en Python con foco en arquitectura mantenible, observabilidad y comparacion experimental entre estrategias de ejecucion secuencial, concurrente y paralela.

El core del algoritmo se mantiene comun para todas las variantes. La diferencia entre versiones se concentra en la estrategia de evaluacion de fitness y en el modo de actualizacion, lo que evita mantener varias implementaciones disjuntas del PSO y hace la comparacion mucho mas justa.

El protocolo experimental guardado en `{protocol_root.as_posix()}` incluye:

- benchmarks sobre `Sphere`, `Rosenbrock`, `Rastrigin` y `Ackley`
- dimensiones `2`, `10` y `30`
- cinco semillas reproducibles por caso
- variantes `V0` a `V5`
- grid search reducido `3x3x3` sobre `w`, `c1` y `c2` para cada variante
- persistencia estructurada de metricas, logs, resumenes y trayectorias

Conclusiones ejecutivas:

- La variante mas rapida del estudio fue `{overall_fastest["variant"]}` en dimension `{overall_fastest["dimensions"]}`, con tiempo medio `{overall_fastest["mean_total_time"]:.4f}` s.
- La variante mas lenta fue `{overall_slowest["variant"]}` en dimension `{overall_slowest["dimensions"]}`, lo que confirma el peso del overhead cuando el trabajo por tarea es pequeno.
- La mejor calidad media global la obtuvo `{best_quality_variant["variant"]}` con fitness medio `{best_quality_variant["mean_best_value"]:.4f}`.
- En los benchmarks numericos de este proyecto, la recomendacion dominante se concentra en las variantes con menor overhead, especialmente `{overall_fastest["variant"]}`.

## 2. Objetivo Del Trabajo

El objetivo era disenar e implementar una solucion completa y mantenible de PSO en Python, con buenas practicas de ingenieria del software, y usarla como banco de pruebas para comparar distintas estrategias de concurrencia y paralelismo.

Ademas del optimizador, la practica exigia:

- benchmarks estandar
- instrumentacion y logging
- grid search de hiperparametros
- persistencia en disco
- visualizacion
- scripts reproducibles
- documentacion de arquitectura

## 3. Arquitectura Del Proyecto

La estructura final se organiza asi:

- `core/`: motor PSO, estado, topologias, limites y configuracion
- `objectives/`: funciones benchmark y registro central
- `parallel/`: evaluadores `sequential`, `thread`, `process`, `asyncio`, `vectorized` y `joblib`
- `experiments/`: runners, benchmarks, grid search, analisis y generacion de informe
- `io/`: guardado de `JSON`, `CSV` y `NPZ`
- `viz/`: curvas de convergencia y animaciones del enjambre
- `dashboard/`: interfaz local en Gradio para inspeccionar resultados
- `scripts/`: comandos reproducibles de alto nivel

### 3.1 Diagrama de dependencias

```mermaid
flowchart LR
    S[scripts/*] --> R[experiments.runner]
    R --> C[core.PSO]
    R --> O[objectives.registry]
    R --> P[parallel/* evaluators]
    C --> B[core.bounds]
    C --> T[core.topology]
    C --> STOP[core.stopping]
    R --> IO[io.results]
    IO --> RES[(results/*)]
    RES --> V[viz.plots]
    RES --> D[dashboard.app]
```

### 3.2 Decisiones de diseno

1. Core comun y estrategias intercambiables.
2. Politica de limites por defecto `clamp`, con `reflect` como alternativa.
3. Topologia minima `global-best` y topologia adicional `ring`.
4. `V3` con latencia simulada para que `asyncio` tenga sentido metodologico.
5. `V4` como referencia de paralelismo implicito mediante NumPy.
6. `V5` con `joblib` para comparar un framework de mas alto nivel sobre la misma API experimental.

### 3.3 Uso de dataclass

Se emplean `dataclass` en:

- `PSOConfig`
- `SwarmState`
- `IterationMetrics`
- `PSOResult`
- `ObjectiveSpec`
- `EvaluationStats`

Esto facilita inspeccion, tipado, persistencia y trazabilidad del estado del algoritmo.

## 4. Metodologia Experimental

### 4.1 Benchmarks

- Objetivos: `Sphere`, `Rosenbrock`, `Rastrigin`, `Ackley`
- Dimensiones: `2`, `10`, `30`
- Semillas: `11`, `23`, `37`, `47`, `59`
- Iteraciones maximas: `120`
- Tamano de enjambre: `40`
- Topologia por defecto: `global`
- Politica de limites: `clamp`

### 4.2 Grid Search

Se ejecuto una rejilla `3x3x3` sobre:

- `w in {{0.50, 0.7298, 0.90}}`
- `c1 in {{1.20, 1.49618, 1.80}}`
- `c2 in {{1.20, 1.49618, 1.80}}`

para cinco semillas y para cada variante `V0` a `V5`, usando `Sphere` en dimension `10` como caso de ajuste.

### 4.3 Comandos reproducibles

```bash
python scripts/run_pso.py --objective sphere --dim 10 --iters 200 --seed 123
python scripts/run_benchmarks.py --config configs/benchmark_suite_full.yaml
python scripts/run_grid_search.py --config configs/grid_search_protocol.yaml
python scripts/run_full_protocol.py --config configs/protocol_full.yaml
python scripts/run_dashboard.py --results-root results/protocol_full
```

### 4.4 Instrumentacion

Cada iteracion registra:

- `best_fitness`
- `mean_fitness`
- `min_fitness_current`
- `eval_time`
- `update_time`
- `worker_time_total`
- `critical_path_time`
- `overhead_time`
- `tasks_submitted`

Esto permite separar el coste de calculo del overhead introducido por cada estrategia.

## 5. Resultados De Benchmarks

![Curva media de convergencia]({benchmark_plot})

### 5.1 Rendimiento agregado por caso

{_markdown_table(benchmark_summary[:24], ["variant", "objective", "dimensions", "runs", "mean_best_value", "mean_total_time", "mean_auc", "mean_convergence_iteration"])}

### 5.2 Tiempo medio y speedup frente a V0

![Speedup por estrategia]({speedup_plot})

{_markdown_table(speedups, ["variant", "dimensions", "mean_total_time", "speedup_vs_V0"])}

### 5.3 Calidad final media por variante

![Distribucion de fitness final]({boxplot})

{_markdown_table(by_variant, ["variant", "mean_best_value"])}

### 5.4 Ganadores por caso

Mejor variante por tiempo:

{_markdown_table(fastest_per_case, ["objective", "dimensions", "winner_variant", "mean_total_time"])}

Mejor variante por calidad final:

{_markdown_table(best_quality_per_case, ["objective", "dimensions", "winner_variant", "mean_best_value"])}

### 5.5 Lectura de resultados

Los resultados muestran un patron muy consistente:

1. `V4` domina claramente en tiempo medio en casi todos los casos.
2. `V1` no supera al baseline, lo que encaja con la limitacion del GIL.
3. `V2` es correcto pero caro para estos benchmarks; el IPC pesa demasiado.
4. `V3` cumple su papel metodologico, pero no es competitivo en CPU-bound.
5. `V5` aporta una comparacion util frente a `V2`: misma idea general de paralelismo por procesos, pero con una capa de abstraccion mas alta y un perfil de overhead propio.
6. La calidad de optimizacion es muy similar entre las variantes basadas en el mismo core, lo que confirma que la estrategia de evaluacion no cambia el comportamiento esencial del algoritmo.

## 6. Resultados De Grid Search

La mejor configuracion encontrada para cada variante fue:

{_markdown_table(best_grid, ["variant", "mean_best_value", "mean_total_time", "mean_auc", "mean_convergence_iteration", "inertia", "cognitive", "social"])}

### 6.1 Interpretacion del ajuste

El grid search deja varias ideas utiles:

- `w = 0.50` aparece de forma estable en las mejores configuraciones.
- Un termino cognitivo alto (`c1 = 1.80`) favorece una convergencia fuerte en `Sphere`.
- `V4` adopta una combinacion algo distinta en el termino social, lo que sugiere que la dinamica efectiva cambia al operar por lotes.

## 7. Discusion Critica

### 7.1 V0 frente a V4

La comparacion mas clara del proyecto es `V0` frente a `V4`. La version vectorizada desplaza el trabajo desde bucles Python hacia operaciones NumPy, reduce overhead interpretado y consigue speedups altos en todas las dimensiones.

### 7.2 Hilos y GIL

`V1` usa `ThreadPoolExecutor`. En evaluacion numerica simple la mejora es limitada porque el GIL no desaparece. La variante es util como contraste didactico: concurrencia no implica speedup automaticamente.

### 7.3 Procesos e IPC

`V2` elimina el GIL al usar procesos, pero paga serializacion, copia de datos y coordinacion. En este protocolo ese coste domina, por lo que la variante queda por detras del baseline.

### 7.4 Asyncio con sentido

`V3` no se presenta como aceleracion universal, sino como solucion para objetivos con latencia. Es una decision honesta metodologicamente y evita vender `asyncio` como mejora en CPU-bound cuando no lo es.

### 7.5 Que estrategia conviene segun el caso

- Si la funcion objetivo esta vectorizada con NumPy: `V4`
- Si la funcion objetivo es costosa y no vectorizable: probar `V2`
- Si quieres comparar un framework de mas alto nivel para paralelismo local: `V5`
- Si hay latencia o simulacion de I/O: `V3`
- Si se quiere una referencia simple y clara: `V0`
- Si se quiere demostrar el efecto del GIL: `V1`

## 8. Persistencia, Observabilidad Y Reproducibilidad

Cada corrida guarda:

- `summary.json`: configuracion, commit, hardware, estrategia y metricas finales
- `history.csv`: metricas por iteracion
- `run.log`: logging estructurado con tiempos y eventos
- `trajectory.npz`: cuando se activa seguimiento del enjambre

De cara al repositorio versionado, se puede conservar un subconjunto pequeno de
artefactos representativos en `results/examples/`, dejando el resto como
resultados locales regenerables.

Ademas, el proyecto incluye:

- dashboard local en Gradio
- plots de analisis agregados
- visualizacion 2D/3D del enjambre
- tests automaticos para reproducibilidad, limites, convergencia y monotonicidad del mejor global

### 8.1 Dashboard

El dashboard permite:

- listar corridas guardadas
- inspeccionar `summary.json`
- ver la curva de convergencia
- abrir artefactos graficos
- revisar plots agregados del protocolo

Comando:

```bash
python scripts/run_dashboard.py --results-root results/protocol_full
```

## 9. Limitaciones Y Amenazas A La Validez

- El dashboard se apoya en resultados ya generados; no sustituye al analisis experimental.
- La variante con procesos puede resultar cara en Windows y en maquinas con pocos nucleos.
- La visualizacion 3D muestra la nube del enjambre, pero no una superficie completa.
- Los resultados dependen de la maquina; por eso se registran hardware y commit.
- El grid search se ha centrado en `Sphere`, por lo que no necesariamente transfiere igual a funciones mas rugosas.

## 10. Recomendaciones Finales

- Para funciones numericas vectorizables, priorizar `V4`.
- Para evaluacion costosa y no vectorizable, estudiar `V2` con batching.
- Usar `V5` cuando interese comparar ergonomia y comportamiento de `joblib` frente a procesos manuales.
- Usar `V1` como comparacion didactica del efecto del GIL.
- Reservar `V3` para objetivos con latencia o I/O real o simulada.
- Mantener `V0` como baseline de referencia en todas las comparativas.

## 11. Artefactos Generados

- Benchmarks: `{benchmark_root.as_posix()}`
- Grid search: `{grid_root.as_posix()}`
- Plots de analisis: `{analysis_root.as_posix()}`
- Dashboard local: `python scripts/run_dashboard.py --results-root {protocol_root.as_posix()}`

## 12. Cierre

El objetivo de la practica se ha cubierto de forma completa:

- se diseno un PSO mantenible
- se implementaron variantes intercambiables
- se instrumento el sistema
- se persistieron resultados
- se generaron visualizaciones
- se ejecuto un protocolo experimental reproducible
- se anadio un dashboard local para exploracion
- se incorporo un bonus `V5` con `joblib`

La conclusion principal es clara: para este proyecto y este perfil de benchmark, la mejor combinacion entre simplicidad, velocidad y calidad sigue viniendo de las variantes con menos overhead, con `V4` como referencia fuerte para objetivos vectorizables y `V0` como baseline conceptual mas limpio. `V5` aporta ademas una comparacion valiosa con un framework de mas alto nivel.
"""
    output_path = Path(output_path)
    output_path.write_text(report, encoding="utf-8")
    return output_path
