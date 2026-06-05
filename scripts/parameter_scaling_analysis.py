#!/usr/bin/env python3
"""Parameter-count scaling diagnostics for the three-judge MARS-Bench snapshot.

This script joins public model metadata (`models.size_b`) to the June 5
three-judge analysis output. It reports correlations between model size and
performance overall and by domain, plus a sparse-MoE sensitivity check that
uses active parameters where those counts are available in the provenance JSON.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


DOMAIN_ORDER = [1, 2, 3, 4, 5, 6, 7]

SHORT_LABELS = {
    "command-a-03-2025": "Command A",
    "deepseek-v4-flash": "DeepSeek V4F",
    "gemma-4-12b-it-q8_0-llamacpp-a6000": "Gemma12",
    "gemma-4-31b-it": "Gemma31",
    "glm-5.1": "GLM5.1",
    "hermes-3-llama-3.1-405b": "Hermes405",
    "llada2.1-flash": "LLaDA100",
    "llama-3.3-70b-instruct": "Llama70",
    "llama4-maverick-17b-128e-moe": "Llama4",
    "mistral-medium-3.5": "Mistral128",
    "qwen3.5-122b-a10b": "Qwen122",
    "qwen3.5-397b-a17b-moe": "Qwen397",
    "qwen3.5-9b": "Qwen9",
    "qwen3.6-27b-fp8": "Qwen27",
    "qwen3.6-35b-a3b-q8_0": "Qwen35",
}

LABEL_OFFSETS = {
    "gemma-4-12b-it-q8_0-llamacpp-a6000": (6, -8),
    "qwen3.5-9b": (6, 12),
    "qwen3.6-27b-fp8": (6, -10),
    "qwen3.6-35b-a3b-q8_0": (6, 10),
    "gemma-4-31b-it": (6, -10),
    "llama-3.3-70b-instruct": (6, -8),
    "llada2.1-flash": (6, 14),
    "command-a-03-2025": (6, -8),
    "qwen3.5-122b-a10b": (6, 12),
    "mistral-medium-3.5": (6, 14),
    "deepseek-v4-flash": (6, 12),
    "llama4-maverick-17b-128e-moe": (6, -8),
    "qwen3.5-397b-a17b-moe": (6, 12),
    "hermes-3-llama-3.1-405b": (6, -8),
    "glm-5.1": (-72, 16),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/mars_bench_stats_public.sqlite")
    parser.add_argument(
        "--three-judge-json",
        default="results/three_judge/final_2026_06_05_three_judge_results.json",
    )
    parser.add_argument(
        "--parameter-provenance-json",
        default="results/diagnostics/model_parameter_counts_2026_06_05.json",
    )
    parser.add_argument("--output-dir", default="results/diagnostics")
    parser.add_argument("--label", default="parameter_scaling_2026_06_05")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fnum(value: Any) -> float:
    return float(value)


def corr(x: list[float], y: list[float]) -> dict[str, float]:
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {
        "n": len(x),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def load_model_sizes(db_path: Path) -> tuple[dict[str, float], dict[int, str]]:
    conn = sqlite3.connect(db_path)
    try:
        sizes = {
            str(name): float(size_b)
            for name, size_b in conn.execute("SELECT name, size_b FROM models")
            if size_b is not None
        }
        domains = {
            int(domain_id): str(name)
            for domain_id, name in conn.execute("SELECT id, name FROM domains")
        }
    finally:
        conn.close()
    return sizes, domains


def load_active_sizes(provenance_path: Path) -> dict[str, float | None]:
    provenance = read_json(provenance_path)
    active: dict[str, float | None] = {}
    for row in provenance["models"]:
        size = row["size_b"]
        active[row["mars_model"]] = row["active_b"] if row["active_b"] is not None else size
    return active


def build_rows(
    model_sizes: dict[str, float],
    active_sizes: dict[str, float | None],
    three_judge: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    domain_means = three_judge["domain_model_means_3judge"]
    composite = three_judge["composite_3judge_with_deepseek"]
    for model in sorted(domain_means):
        if model not in model_sizes:
            continue
        domains = {int(k): fnum(v) for k, v in domain_means[model].items()}
        size_b = model_sizes[model]
        active_b = active_sizes.get(model) or size_b
        row = {
            "model": model,
            "label": SHORT_LABELS.get(model, model),
            "size_b": size_b,
            "active_or_dense_b": float(active_b),
            "log10_size_b": math.log10(size_b),
            "log10_active_or_dense_b": math.log10(float(active_b)),
            "role_weighted_composite_3judge": fnum(composite[model]),
            "all_domain_mean_3judge": float(np.mean([domains[d] for d in DOMAIN_ORDER])),
        }
        for domain_id in DOMAIN_ORDER:
            row[f"d{domain_id}_mean_3judge"] = domains[domain_id]
        rows.append(row)
    return rows


def target_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [fnum(row[key]) for row in rows]


def compute_correlations(rows: list[dict[str, Any]], domains: dict[int, str]) -> dict[str, Any]:
    total_raw = target_values(rows, "size_b")
    total_log = target_values(rows, "log10_size_b")
    active_raw = target_values(rows, "active_or_dense_b")
    active_log = target_values(rows, "log10_active_or_dense_b")

    targets = {
        "all_domain_mean_3judge": "Unweighted mean of D1-D7 three-judge domain means",
        "role_weighted_composite_3judge": "Role-weighted three-judge composite",
    }
    for domain_id in DOMAIN_ORDER:
        targets[f"d{domain_id}_mean_3judge"] = f"D{domain_id} {domains[domain_id]}"

    by_target: dict[str, Any] = {}
    for key, description in targets.items():
        y = target_values(rows, key)
        by_target[key] = {
            "description": description,
            "total_size_raw": corr(total_raw, y),
            "total_size_log10": corr(total_log, y),
            "active_or_dense_raw": corr(active_raw, y),
            "active_or_dense_log10": corr(active_log, y),
        }
    return by_target


def regression(rows: list[dict[str, Any]], y_key: str) -> dict[str, Any]:
    x = np.array(target_values(rows, "log10_size_b"))
    y = np.array(target_values(rows, y_key))
    fit = stats.linregress(x, y)
    residuals = []
    for row, x_value, y_value in zip(rows, x, y):
        predicted = float(fit.intercept + fit.slope * x_value)
        residuals.append(
            {
                "model": row["model"],
                "label": row["label"],
                "size_b": row["size_b"],
                "observed": float(y_value),
                "predicted": predicted,
                "residual": float(y_value - predicted),
            }
        )
    residuals.sort(key=lambda item: item["residual"], reverse=True)
    return {
        "target": y_key,
        "x": "log10_size_b",
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "r": float(fit.rvalue),
        "r_squared": float(fit.rvalue**2),
        "p": float(fit.pvalue),
        "stderr": float(fit.stderr),
        "positive_residuals": residuals[:5],
        "negative_residuals": list(reversed(residuals[-5:])),
    }


def write_model_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "model",
        "label",
        "size_b",
        "active_or_dense_b",
        "all_domain_mean_3judge",
        "role_weighted_composite_3judge",
        *[f"d{d}_mean_3judge" for d in DOMAIN_ORDER],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["all_domain_mean_3judge"], reverse=True):
            writer.writerow({field: row[field] for field in fields})


def write_correlation_csv(correlations: dict[str, Any], path: Path) -> None:
    fields = [
        "target",
        "description",
        "metric",
        "n",
        "pearson_r",
        "pearson_p",
        "spearman_rho",
        "spearman_p",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for target, payload in correlations.items():
            for metric in [
                "total_size_raw",
                "total_size_log10",
                "active_or_dense_raw",
                "active_or_dense_log10",
            ]:
                writer.writerow(
                    {
                        "target": target,
                        "description": payload["description"],
                        "metric": metric,
                        **payload[metric],
                    }
                )


def sx(value: float, xmin: float, xmax: float, left: float, width: float) -> float:
    return left + (value - xmin) / (xmax - xmin) * width


def sy(value: float, ymin: float, ymax: float, top: float, height: float) -> float:
    return top + (ymax - value) / (ymax - ymin) * height


def svg_text(x: float, y: float, text: str, size: int = 11, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Arial, sans-serif" text-anchor="{anchor}" fill="#263238">'
        f"{html.escape(text)}</text>"
    )


def write_scatter_svg(rows: list[dict[str, Any]], reg: dict[str, Any], path: Path) -> None:
    width, height = 980, 620
    left, top, plot_w, plot_h = 78, 50, 790, 480
    x_values = target_values(rows, "log10_size_b")
    y_values = target_values(rows, "all_domain_mean_3judge")
    xmin, xmax = math.log10(8), math.log10(850)
    ymin, ymax = min(y_values) - 0.08, max(y_values) + 0.10
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>.axis{stroke:#455a64;stroke-width:1}.grid{stroke:#d6dde1;stroke-width:1}.trend{stroke:#0b7285;stroke-width:2.5}.point{stroke:#263238;stroke-width:1}</style>',
        svg_text(32, 28, "MARS-Bench Parameter Scaling: all-domain mean vs total parameters", 18),
        svg_text(32, 48, "Three-judge mean scores; x-axis is log10(total parameters in billions)", 12),
    ]
    for tick in [10, 30, 100, 300, 800]:
        x = sx(math.log10(tick), xmin, xmax, left, plot_w)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}"/>')
        parts.append(f'<line class="axis" x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 5}"/>')
        parts.append(svg_text(x, top + plot_h + 22, f"{tick}B", 11, "middle"))
    y_step = 0.2
    y_tick = math.floor(ymin / y_step) * y_step
    while y_tick <= ymax:
        y = sy(y_tick, ymin, ymax, top, plot_h)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        parts.append(f'<line class="axis" x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}"/>')
        parts.append(svg_text(left - 10, y + 4, f"{y_tick:.1f}", 11, "end"))
        y_tick += y_step
    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
            f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
            svg_text(left + plot_w / 2, height - 34, "Total parameters (log scale)", 13, "middle"),
            f'<text x="22" y="{top + plot_h / 2:.1f}" font-size="13" font-family="Arial, sans-serif" text-anchor="middle" fill="#263238" transform="rotate(-90 22 {top + plot_h / 2:.1f})">All-domain mean score</text>',
        ]
    )
    x1, x2 = xmin, xmax
    y1 = reg["intercept"] + reg["slope"] * x1
    y2 = reg["intercept"] + reg["slope"] * x2
    parts.append(
        f'<line class="trend" x1="{sx(x1, xmin, xmax, left, plot_w):.1f}" y1="{sy(y1, ymin, ymax, top, plot_h):.1f}" x2="{sx(x2, xmin, xmax, left, plot_w):.1f}" y2="{sy(y2, ymin, ymax, top, plot_h):.1f}"/>'
    )
    for row in rows:
        x = sx(row["log10_size_b"], xmin, xmax, left, plot_w)
        y = sy(row["all_domain_mean_3judge"], ymin, ymax, top, plot_h)
        color = "#2b8a3e" if row["all_domain_mean_3judge"] >= 2.55 else "#f08c00" if row["all_domain_mean_3judge"] >= 2.35 else "#c92a2a"
        parts.append(f'<circle class="point" cx="{x:.1f}" cy="{y:.1f}" r="5.2" fill="{color}"/>')
        dx, dy = LABEL_OFFSETS.get(row["model"], (6, -8))
        parts.append(svg_text(x + dx, y + dy, row["label"], 10))
    parts.append(
        svg_text(
            left + plot_w - 14,
            top + 18,
            f"Pearson r(log total,size)={reg['r']:.3f}; R^2={reg['r_squared']:.3f}; p={reg['p']:.4f}",
            12,
            "end",
        )
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_domain_bar_svg(correlations: dict[str, Any], domains: dict[int, str], path: Path) -> None:
    width, height = 980, 500
    left, top, plot_w, plot_h = 84, 56, 800, 330
    ymin, ymax = -0.1, 0.9
    bar_w = plot_w / (len(DOMAIN_ORDER) * 2.7)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>.axis{stroke:#455a64;stroke-width:1}.grid{stroke:#d6dde1;stroke-width:1}</style>',
        svg_text(32, 28, "Parameter/performance correlation by domain", 18),
        svg_text(32, 48, "Bars show correlation between total parameter count and three-judge domain mean", 12),
    ]
    for tick in [-0.1, 0.0, 0.3, 0.6, 0.9]:
        y = sy(tick, ymin, ymax, top, plot_h)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        parts.append(svg_text(left - 10, y + 4, f"{tick:.1f}", 11, "end"))
    zero = sy(0.0, ymin, ymax, top, plot_h)
    parts.append(f'<line class="axis" x1="{left}" y1="{zero:.1f}" x2="{left + plot_w}" y2="{zero:.1f}"/>')
    group_w = plot_w / len(DOMAIN_ORDER)
    for idx, domain_id in enumerate(DOMAIN_ORDER):
        target = f"d{domain_id}_mean_3judge"
        pearson = correlations[target]["total_size_log10"]["pearson_r"]
        spearman = correlations[target]["total_size_raw"]["spearman_rho"]
        center = left + group_w * (idx + 0.5)
        for offset, value, color in [(-bar_w * 0.55, pearson, "#0b7285"), (bar_w * 0.55, spearman, "#862e9c")]:
            x = center + offset - bar_w / 2
            y = sy(max(value, 0), ymin, ymax, top, plot_h)
            y0 = sy(0, ymin, ymax, top, plot_h)
            h = abs(y0 - sy(value, ymin, ymax, top, plot_h))
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(svg_text(center, top + plot_h + 22, f"D{domain_id}", 12, "middle"))
        parts.append(svg_text(center, top + plot_h + 38, domains[domain_id].replace("_", " "), 9, "middle"))
    parts.extend(
        [
            svg_text(left + plot_w / 2, height - 34, "Domain", 13, "middle"),
            f'<text x="22" y="{top + plot_h / 2:.1f}" font-size="13" font-family="Arial, sans-serif" text-anchor="middle" fill="#263238" transform="rotate(-90 22 {top + plot_h / 2:.1f})">Correlation</text>',
            f'<rect x="{left + plot_w - 190}" y="{top + 8}" width="12" height="12" fill="#0b7285"/>',
            svg_text(left + plot_w - 172, top + 19, "Pearson r on log10(total)", 11),
            f'<rect x="{left + plot_w - 190}" y="{top + 28}" width="12" height="12" fill="#862e9c"/>',
            svg_text(left + plot_w - 172, top + 39, "Spearman rho on total", 11),
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    db_path = Path(args.db)
    three_judge_path = Path(args.three_judge_json)
    provenance_path = Path(args.parameter_provenance_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_sizes, domains = load_model_sizes(db_path)
    active_sizes = load_active_sizes(provenance_path)
    three_judge = read_json(three_judge_path)
    rows = build_rows(model_sizes, active_sizes, three_judge)
    correlations = compute_correlations(rows, domains)
    reg = regression(rows, "all_domain_mean_3judge")

    payload = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "db_path": str(db_path),
            "three_judge_json": str(three_judge_path),
            "parameter_provenance_json": str(provenance_path),
            "n_models": len(rows),
            "size_b_definition": "Total parameters in billions from models.size_b; active_or_dense_b substitutes official active parameters for sparse MoE models when available.",
        },
        "correlations": correlations,
        "all_domain_log_total_regression": reg,
        "per_model": sorted(rows, key=lambda item: item["all_domain_mean_3judge"], reverse=True),
    }

    json_path = output_dir / f"{args.label}.json"
    model_csv_path = output_dir / f"{args.label}_model_rows.csv"
    corr_csv_path = output_dir / f"{args.label}_correlations.csv"
    scatter_path = output_dir / f"{args.label}_all_domain_scatter.svg"
    domain_svg_path = output_dir / f"{args.label}_domain_correlations.svg"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_model_csv(rows, model_csv_path)
    write_correlation_csv(correlations, corr_csv_path)
    write_scatter_svg(rows, reg, scatter_path)
    write_domain_bar_svg(correlations, domains, domain_svg_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {model_csv_path}")
    print(f"Wrote {corr_csv_path}")
    print(f"Wrote {scatter_path}")
    print(f"Wrote {domain_svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
