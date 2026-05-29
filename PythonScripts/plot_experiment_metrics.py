from __future__ import annotations

import argparse
import csv
import html
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


COLORS = [
    "#2F6FED",
    "#D1495B",
    "#00876C",
    "#7B3FB3",
    "#E17C05",
    "#4D908E",
    "#8F2D56",
    "#5C677D",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate crowd simulation performance graphs from Unity MetricsLogger CSV files."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="Assets/Data",
        help="CSV file or directory containing metrics CSV files. Defaults to Assets/Data.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated graphs and summary CSV files.",
    )
    parser.add_argument(
        "--agent-counts",
        nargs="*",
        type=int,
        default=None,
        help="Optional agent counts to include, for example: --agent-counts 1000 2000 5000 10000.",
    )
    parser.add_argument("--variants", nargs="*", default=None, help="Optional variant names to include.")
    parser.add_argument("--show", action="store_true", help="Show matplotlib plots interactively when available.")
    return parser.parse_args()


def find_csv_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")

    csv_files = sorted(input_path.rglob("*.csv"))
    return [
        path
        for path in csv_files
        if ("crowd_metrics" in path.name.lower() or "experiment" in path.name.lower())
        and "summary" not in path.name.lower()
    ]


def load_metrics(csv_files: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    required = {"time_seconds", "variant", "agent_count", "average_delta_time_ms", "average_fps"}

    for csv_file in csv_files:
        try:
            with csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or not required.issubset(reader.fieldnames):
                    print(f"Skipping non-metrics CSV: {csv_file}", file=sys.stderr)
                    continue

                for source_row, row in enumerate(reader):
                    parsed = parse_metric_row(row)
                    if parsed is None:
                        continue

                    parsed["source_file"] = str(csv_file)
                    parsed["source_row"] = source_row
                    rows.append(parsed)
        except Exception as exc:
            print(f"Skipping unreadable CSV: {csv_file} ({exc})", file=sys.stderr)

    if not rows:
        raise SystemExit("No compatible metrics CSV files found.")

    return rows


def parse_metric_row(row: dict[str, str]) -> dict[str, object] | None:
    try:
        variant = row["variant"] or "Unknown"
        if variant == "Baseline":
            variant = "BaselineNavMesh"

        return {
            "time_seconds": float(row["time_seconds"]),
            "variant": variant,
            "agent_count": int(float(row["agent_count"])),
            "completed_tasks": int(float(row.get("completed_tasks") or 0)),
            "stuck_agents": int(float(row.get("stuck_agents") or 0)),
            "average_delta_time_ms": float(row["average_delta_time_ms"]),
            "average_fps": float(row["average_fps"]),
        }
    except (TypeError, ValueError, KeyError):
        return None


def add_run_ids(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = sorted(rows, key=lambda row: (str(row["source_file"]), int(row["source_row"])))
    run_id = 0
    previous: dict[str, object] | None = None

    for row in rows:
        if (
            previous is None
            or row["source_file"] != previous["source_file"]
            or row["variant"] != previous["variant"]
            or row["agent_count"] != previous["agent_count"]
            or float(row["time_seconds"]) < float(previous["time_seconds"])
        ):
            run_id += 1

        row["run_id"] = run_id
        previous = row

    return rows


def summarize_runs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object, object, object], list[dict[str, object]]] = defaultdict(list)

    for row in rows:
        key = (row["source_file"], row["run_id"], row["variant"], row["agent_count"])
        grouped[key].append(row)

    summaries: list[dict[str, object]] = []
    for (source_file, run_id, variant, agent_count), samples in grouped.items():
        fps_values = [float(sample["average_fps"]) for sample in samples]
        delta_values = [float(sample["average_delta_time_ms"]) for sample in samples]
        duration_seconds = max(float(sample["time_seconds"]) for sample in samples)
        final_completed_tasks = max(int(sample["completed_tasks"]) for sample in samples)
        stuck_values = [int(sample["stuck_agents"]) for sample in samples]
        completed_tasks_per_minute = final_completed_tasks / max(duration_seconds, 0.001) * 60.0
        stuck_percent = statistics.fmean(stuck_values) / max(int(agent_count), 1) * 100.0

        summaries.append(
            {
                "source_file": source_file,
                "run_id": int(run_id),
                "variant": str(variant),
                "agent_count": int(agent_count),
                "sample_count": len(samples),
                "duration_seconds": duration_seconds,
                "average_fps": statistics.fmean(fps_values),
                "p05_fps": percentile(fps_values, 0.05),
                "average_delta_time_ms": statistics.fmean(delta_values),
                "final_completed_tasks": final_completed_tasks,
                "completed_tasks_per_minute": completed_tasks_per_minute,
                "average_stuck_agents": statistics.fmean(stuck_values),
                "max_stuck_agents": max(stuck_values),
                "stuck_percent": stuck_percent,
            }
        )

    return sorted(summaries, key=lambda row: (str(row["variant"]), int(row["agent_count"]), int(row["run_id"])))


def summarize_variants(run_summary: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in run_summary:
        grouped[(str(row["variant"]), int(row["agent_count"]))].append(row)

    metric_names = [
        "average_fps",
        "p05_fps",
        "average_delta_time_ms",
        "completed_tasks_per_minute",
        "stuck_percent",
        "final_completed_tasks",
    ]
    summaries: list[dict[str, object]] = []

    for (variant, agent_count), runs in grouped.items():
        summary: dict[str, object] = {
            "variant": variant,
            "agent_count": agent_count,
            "run_count": len(runs),
        }

        for metric in metric_names:
            values = [float(run[metric]) for run in runs]
            summary[metric] = statistics.fmean(values)
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0

        summaries.append(summary)

    return sorted(summaries, key=lambda row: (str(row["variant"]), int(row["agent_count"])))


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0

    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)

    if lower == upper:
        return values[int(index)]

    lower_weight = upper - index
    upper_weight = index - lower
    return values[lower] * lower_weight + values[upper] * upper_weight


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_all(summary: list[dict[str, object]], output_dir: Path, show: bool) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return plot_all_svg(summary, output_dir)

    outputs = [
        plot_matplotlib(
            summary,
            plt,
            output_dir,
            "average_fps",
            "Average FPS",
            "Average FPS by Agent Count",
            "average_fps_by_agent_count.png",
        ),
        plot_matplotlib(
            summary,
            plt,
            output_dir,
            "p05_fps",
            "5th Percentile FPS",
            "Low FPS by Agent Count",
            "p05_fps_by_agent_count.png",
        ),
        plot_matplotlib(
            summary,
            plt,
            output_dir,
            "average_delta_time_ms",
            "Average Frame Time (ms)",
            "Average Frame Time by Agent Count",
            "frame_time_by_agent_count.png",
        ),
        plot_matplotlib(
            summary,
            plt,
            output_dir,
            "completed_tasks_per_minute",
            "Completed Tasks per Minute",
            "Task Throughput by Agent Count",
            "task_throughput_by_agent_count.png",
        ),
    ]

    if sum(abs(float(row["stuck_percent"])) for row in summary) > 0:
        outputs.append(
            plot_matplotlib(
                summary,
                plt,
                output_dir,
                "stuck_percent",
                "Average Stuck Agents (%)",
                "Stuck Agents by Agent Count",
                "stuck_agents_by_agent_count.png",
            )
        )

    if show:
        plt.show()
    else:
        plt.close("all")

    return outputs


def plot_matplotlib(summary, plt, output_dir: Path, metric: str, ylabel: str, title: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for variant in sorted({str(row["variant"]) for row in summary}):
        frame = sorted((row for row in summary if row["variant"] == variant), key=lambda row: int(row["agent_count"]))
        x_values = [int(row["agent_count"]) for row in frame]
        y_values = [float(row[metric]) for row in frame]
        y_errors = [float(row.get(f"{metric}_std", 0.0)) for row in frame]
        use_errors = any(error > 0 for error in y_errors)
        ax.errorbar(
            x_values,
            y_values,
            yerr=y_errors if use_errors else None,
            marker="o",
            linewidth=2,
            capsize=3,
            label=variant,
        )

    ax.set_title(title)
    ax.set_xlabel("Agent Count")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=180)
    return output_path


def plot_all_svg(summary: list[dict[str, object]], output_dir: Path) -> list[Path]:
    specs = [
        ("average_fps", "Average FPS", "Average FPS by Agent Count", "average_fps_by_agent_count.svg"),
        ("p05_fps", "5th Percentile FPS", "Low FPS by Agent Count", "p05_fps_by_agent_count.svg"),
        (
            "average_delta_time_ms",
            "Average Frame Time (ms)",
            "Average Frame Time by Agent Count",
            "frame_time_by_agent_count.svg",
        ),
        (
            "completed_tasks_per_minute",
            "Completed Tasks per Minute",
            "Task Throughput by Agent Count",
            "task_throughput_by_agent_count.svg",
        ),
    ]

    if sum(abs(float(row["stuck_percent"])) for row in summary) > 0:
        specs.append(("stuck_percent", "Average Stuck Agents (%)", "Stuck Agents by Agent Count", "stuck_agents_by_agent_count.svg"))

    return [plot_svg(summary, output_dir, *spec) for spec in specs]


def plot_svg(summary: list[dict[str, object]], output_dir: Path, metric: str, ylabel: str, title: str, filename: str) -> Path:
    width = 1000
    height = 620
    left = 95
    right = 260
    top = 70
    bottom = 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    variants = sorted({str(row["variant"]) for row in summary})
    x_values = sorted({int(row["agent_count"]) for row in summary})
    y_values = [float(row[metric]) for row in summary]
    y_min = min(0.0, min(y_values))
    y_max = max(y_values)

    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0

    y_padding = (y_max - y_min) * 0.08
    y_max += y_padding

    def x_scale(value: int) -> float:
        if len(x_values) == 1:
            return left + plot_width * 0.5
        return left + (value - min(x_values)) / (max(x_values) - min(x_values)) * plot_width

    def y_scale(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222" stroke-width="1.5"/>',
    ]

    for i in range(6):
        value = y_min + (y_max - y_min) * i / 5
        y = y_scale(value)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#ddd" stroke-width="1"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.1f}</text>')

    for value in x_values:
        x = x_scale(value)
        lines.append(f'<line x1="{x:.1f}" y1="{top + plot_height}" x2="{x:.1f}" y2="{top + plot_height + 6}" stroke="#222" stroke-width="1"/>')
        lines.append(f'<text x="{x:.1f}" y="{top + plot_height + 25}" text-anchor="middle" font-family="Arial" font-size="12">{value}</text>')

    for variant_index, variant in enumerate(variants):
        color = COLORS[variant_index % len(COLORS)]
        points = [
            (x_scale(int(row["agent_count"])), y_scale(float(row[metric])))
            for row in sorted((row for row in summary if row["variant"] == variant), key=lambda row: int(row["agent_count"]))
        ]
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        lines.append(f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="3"/>')

        for x, y in points:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" stroke="#fff" stroke-width="1.5"/>')

        legend_y = top + variant_index * 24
        legend_x = left + plot_width + 35
        lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<circle cx="{legend_x + 12}" cy="{legend_y}" r="4" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 34}" y="{legend_y + 4}" font-family="Arial" font-size="13">{html.escape(variant)}</text>')

    lines.append(f'<text x="{left + plot_width / 2}" y="{height - 25}" text-anchor="middle" font-family="Arial" font-size="15">Agent Count</text>')
    lines.append(
        f'<text x="24" y="{top + plot_height / 2}" text-anchor="middle" font-family="Arial" font-size="15" transform="rotate(-90 24 {top + plot_height / 2})">{html.escape(ylabel)}</text>'
    )
    lines.append("</svg>")

    output_path = output_dir / filename
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def default_output_dir(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.parent / "graphs"

    return input_path / "experiment_graphs"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    csv_files = find_csv_files(input_path)
    rows = load_metrics(csv_files)

    if args.agent_counts:
        rows = [row for row in rows if int(row["agent_count"]) in args.agent_counts]

    if args.variants:
        allowed_variants = set(args.variants)
        rows = [row for row in rows if str(row["variant"]) in allowed_variants]

    if not rows:
        raise SystemExit("No metrics remained after filtering.")

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = add_run_ids(rows)
    run_summary = summarize_runs(rows)
    summary = summarize_variants(run_summary)
    write_csv(output_dir / "experiment_run_summary.csv", run_summary)
    write_csv(output_dir / "experiment_summary.csv", summary)
    graph_paths = plot_all(summary, output_dir, args.show)

    print(f"Read {len(csv_files)} CSV file(s).")
    print(f"Wrote {len(graph_paths)} graph(s), experiment_summary.csv, and experiment_run_summary.csv to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
