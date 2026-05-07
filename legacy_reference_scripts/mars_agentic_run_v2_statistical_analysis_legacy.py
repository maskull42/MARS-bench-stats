#!/usr/bin/env python3
"""
Release-aware statistical analysis for MARS-Bench v2.0.

Design principles:
- release-aware: only one named benchmark release
- judge-aware: only official v2.* rows for the requested judge run number
- completed-cohort only by default
- clustered uncertainty: bootstrap at the question level
- paired comparisons: question-level means, not raw evaluation rows

This is the first dissertation-safe replacement for the old mixed-cohort
`base-llm-benchmark/scripts/statistical_analysis.py` path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = Path(os.environ.get("MARS_BENCH_DB", "data/mars_bench_stats_public.sqlite"))
DEFAULT_RELEASE_LABEL = "mars_bench_v2_0"
DEFAULT_JUDGE_RUN_NUMBER = 1
DEFAULT_BOOTSTRAP_SAMPLES = 2000
EXPECTED_RESPONSES_PER_MODEL = 627


DOMAIN_CODE_BY_ID = {
    1: "D1",
    2: "D2",
    3: "D3",
    4: "D4",
    5: "D5",
    6: "D6",
    7: "D7",
}


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def round_float(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), digits)


def get_release_row(conn: sqlite3.Connection, release_label: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM benchmark_releases WHERE release_label = ?",
        (release_label,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"release not found: {release_label}")
    return row


def get_completed_models(
    conn: sqlite3.Connection,
    release_id: int,
    judge_run_number: int,
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        WITH response_counts AS (
          SELECT r.model_id, COUNT(*) AS response_count
          FROM responses r
          JOIN benchmark_runs br ON br.id = r.benchmark_run_id
          WHERE br.benchmark_release_id = ?
          GROUP BY r.model_id
        ),
        evaluation_counts AS (
          SELECT r.model_id, COUNT(*) AS evaluation_count
          FROM evaluations e
          JOIN responses r ON r.id = e.response_id
          WHERE e.benchmark_release_id = ?
            AND e.judge_prompt_version LIKE 'v2.%'
            AND e.judge_run_number = ?
          GROUP BY r.model_id
        )
        SELECT m.id AS model_id, m.name AS model_name
        FROM models m
        JOIN response_counts rc ON rc.model_id = m.id
        JOIN evaluation_counts ec ON ec.model_id = m.id
        WHERE rc.response_count = ? AND ec.evaluation_count = ?
        ORDER BY m.name
        """,
        (
            release_id,
            release_id,
            judge_run_number,
            EXPECTED_RESPONSES_PER_MODEL,
            EXPECTED_RESPONSES_PER_MODEL,
        ),
    ).fetchall()


def fetch_rows(
    conn: sqlite3.Connection,
    release_id: int,
    judge_run_number: int,
    model_names: list[str],
) -> list[dict]:
    placeholders = ",".join("?" for _ in model_names)
    if not placeholders:
        raise SystemExit("no models selected")

    rows = conn.execute(
        f"""
        SELECT
          m.name AS model,
          q.domain_id,
          q.id AS question_id,
          q.level,
          q.is_trap,
          r.run_number,
          r.response_text,
          e.overall_score,
          e.evaluation_regime
        FROM evaluations e
        JOIN responses r ON r.id = e.response_id
        JOIN models m ON m.id = r.model_id
        JOIN questions q ON q.id = r.question_id
        WHERE e.benchmark_release_id = ?
          AND e.judge_prompt_version LIKE 'v2.%'
          AND e.judge_run_number = ?
          AND m.name IN ({placeholders})
        ORDER BY m.name, q.id, r.run_number
        """,
        (release_id, judge_run_number, *model_names),
    ).fetchall()
    return [dict(row) for row in rows]


def aggregate_question_means(rows: Iterable[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], dict] = {}
    for row in rows:
        key = (row["model"], row["question_id"])
        if key not in grouped:
            grouped[key] = {
                "model": row["model"],
                "question_id": row["question_id"],
                "domain_id": row["domain_id"],
                "level": row["level"],
                "is_trap": row["is_trap"],
                "evaluation_regime": row["evaluation_regime"],
                "scores": [],
            }
        grouped[key]["scores"].append(float(row["overall_score"]))

    aggregated = []
    for entry in grouped.values():
        scores = entry.pop("scores")
        entry["mean_score"] = float(np.mean(scores))
        entry["score_count"] = len(scores)
        aggregated.append(entry)
    return aggregated


def metadata_block(
    raw_rows: list[dict],
    question_rows: list[dict],
    release_label: str,
    release_id: int,
    judge_prompt_version: str,
    judge_run_number: int,
) -> dict:
    return {
        "release_label": release_label,
        "benchmark_release_id": release_id,
        "judge_prompt_version": judge_prompt_version,
        "judge_run_number": judge_run_number,
        "total_evaluations": len(raw_rows),
        "total_models": len({row["model"] for row in raw_rows}),
        "total_questions": len({row["question_id"] for row in raw_rows}),
        "total_question_aggregates": len(question_rows),
        "random_seed": 42,
        "analysis_unit_for_inference": "question_mean",
        "raw_row_unit_used_for": ["verbosity_bias", "run_to_run_variance"],
    }


def verbosity_bias_analysis(raw_rows: list[dict]) -> dict:
    results = {}
    models = sorted({row["model"] for row in raw_rows})
    for model in models:
        model_rows = [row for row in raw_rows if row["model"] == model]
        word_counts = [len((row["response_text"] or "").split()) for row in model_rows]
        scores = [float(row["overall_score"]) for row in model_rows]
        if len(scores) < 3 or len(set(word_counts)) < 2:
            results[model] = {
                "pearson_r": None,
                "p_value": None,
                "avg_word_count": round_float(np.mean(word_counts), 1),
                "verbosity_inflation_index": None,
                "flagged": False,
            }
            continue
        pearson_r, p_value = stats.pearsonr(word_counts, scores)
        median_wc = float(np.median(word_counts))
        above = [score for score, wc in zip(scores, word_counts) if wc > median_wc]
        below = [score for score, wc in zip(scores, word_counts) if wc <= median_wc]
        inflation = 0.0
        if below and np.mean(below) > 0:
            inflation = (float(np.mean(above)) - float(np.mean(below))) / float(np.mean(below))
        results[model] = {
            "pearson_r": round_float(pearson_r, 4),
            "p_value": round_float(p_value, 6),
            "avg_word_count": round_float(np.mean(word_counts), 1),
            "verbosity_inflation_index": round_float(inflation, 4),
            "flagged": abs(float(pearson_r)) > 0.3,
        }
    return results


def run_to_run_variance(raw_rows: list[dict]) -> dict:
    results = {}
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in raw_rows:
        grouped[row["model"]][row["question_id"]].append(float(row["overall_score"]))
    for model, question_map in grouped.items():
        variances = []
        cvs = []
        for scores in question_map.values():
            if len(scores) < 2:
                continue
            variance = float(np.var(scores, ddof=1))
            variances.append(variance)
            mean_score = float(np.mean(scores))
            if mean_score > 0:
                cvs.append(float(np.std(scores, ddof=1)) / mean_score)
        results[model] = {
            "mean_variance": round_float(np.mean(variances) if variances else 0.0, 5),
            "mean_cv": round_float(np.mean(cvs) if cvs else 0.0, 4),
            "high_cv_questions": int(sum(1 for cv in cvs if cv > 0.15)),
            "total_questions": int(len(cvs)),
        }
    return results


def score_distribution_analysis(question_rows: list[dict]) -> dict:
    results = {}
    models = sorted({row["model"] for row in question_rows})
    for model in models:
        scores = [float(row["mean_score"]) for row in question_rows if row["model"] == model]
        results[model] = {
            "mean": round_float(np.mean(scores), 3),
            "median": round_float(np.median(scores), 3),
            "std": round_float(np.std(scores, ddof=1), 3),
            "skewness": round_float(stats.skew(scores), 3),
            "kurtosis": round_float(stats.kurtosis(scores), 3),
            "min": round_float(min(scores), 2),
            "max": round_float(max(scores), 2),
            "pct_below_2": round_float(sum(1 for s in scores if s < 2) / len(scores) * 100, 1),
            "pct_3_plus": round_float(sum(1 for s in scores if s >= 3) / len(scores) * 100, 1),
            "pct_4_plus": round_float(sum(1 for s in scores if s >= 4) / len(scores) * 100, 1),
        }
    return results


def difficulty_level_analysis(question_rows: list[dict]) -> dict:
    results = {}
    models = sorted({row["model"] for row in question_rows})
    for model in models:
        model_rows = [row for row in question_rows if row["model"] == model]
        levels = {}
        for level in [1, 2, 3]:
            level_scores = [float(row["mean_score"]) for row in model_rows if row["level"] == level]
            if level_scores:
                levels[f"L{level}"] = {
                    "mean": round_float(np.mean(level_scores), 3),
                    "n": len(level_scores),
                }
        results[model] = levels
    return results


def trap_question_analysis(question_rows: list[dict]) -> dict:
    results = {}
    models = sorted({row["model"] for row in question_rows})
    for model in models:
        model_rows = [row for row in question_rows if row["model"] == model]
        trap_scores = [float(row["mean_score"]) for row in model_rows if row["is_trap"]]
        non_trap_scores = [float(row["mean_score"]) for row in model_rows if not row["is_trap"]]
        if trap_scores and non_trap_scores:
            trap_mean = float(np.mean(trap_scores))
            non_trap_mean = float(np.mean(non_trap_scores))
            results[model] = {
                "trap_mean": round_float(trap_mean, 3),
                "non_trap_mean": round_float(non_trap_mean, 3),
                "difference": round_float(trap_mean - non_trap_mean, 3),
                "n_trap": len(trap_scores),
                "n_non_trap": len(non_trap_scores),
            }
    return results


def question_cluster_bootstrap(
    model_question_scores: dict[int, float],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    question_ids = sorted(model_question_scores.keys())
    observed = float(np.mean([model_question_scores[qid] for qid in question_ids]))
    boot_means = []
    for _ in range(n_bootstrap):
        sample_ids = rng.choice(question_ids, size=len(question_ids), replace=True)
        sample_scores = [model_question_scores[int(qid)] for qid in sample_ids]
        boot_means.append(float(np.mean(sample_scores)))
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return observed, float(ci_low), float(ci_high)


def bootstrap_confidence_intervals(question_rows: list[dict], n_bootstrap: int) -> dict:
    rng = np.random.default_rng(42)
    results = {}
    models = sorted({row["model"] for row in question_rows})
    for model in models:
        model_rows = [row for row in question_rows if row["model"] == model]
        overall_map = {int(row["question_id"]): float(row["mean_score"]) for row in model_rows}
        overall_mean, ci_low, ci_high = question_cluster_bootstrap(overall_map, n_bootstrap, rng)

        per_domain = {}
        for domain_id in sorted(DOMAIN_CODE_BY_ID):
            domain_rows = [row for row in model_rows if row["domain_id"] == domain_id]
            if not domain_rows:
                continue
            domain_map = {int(row["question_id"]): float(row["mean_score"]) for row in domain_rows}
            d_mean, d_low, d_high = question_cluster_bootstrap(domain_map, n_bootstrap, rng)
            per_domain[DOMAIN_CODE_BY_ID[domain_id]] = {
                "mean": round_float(d_mean, 3),
                "ci_low": round_float(d_low, 3),
                "ci_high": round_float(d_high, 3),
            }

        results[model] = {
            "overall_mean": round_float(overall_mean, 3),
            "ci_95_low": round_float(ci_low, 3),
            "ci_95_high": round_float(ci_high, 3),
            "per_domain": per_domain,
        }
    return results


def pairwise_model_comparisons(question_rows: list[dict]) -> list[dict]:
    models = sorted({row["model"] for row in question_rows if "ceiling" not in row["model"]})
    by_model = {
        model: {int(row["question_id"]): float(row["mean_score"]) for row in question_rows if row["model"] == model}
        for model in models
    }
    results = []
    p_values = []
    for model_1, model_2 in combinations(models, 2):
        common_questions = sorted(set(by_model[model_1]) & set(by_model[model_2]))
        scores_1 = [by_model[model_1][qid] for qid in common_questions]
        scores_2 = [by_model[model_2][qid] for qid in common_questions]
        diffs = [a - b for a, b in zip(scores_1, scores_2)]
        if all(diff == 0 for diff in diffs):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = stats.wilcoxon(diffs, alternative="two-sided")
        pooled_std = math.sqrt((np.var(scores_1, ddof=1) + np.var(scores_2, ddof=1)) / 2)
        cohens_d = 0.0 if pooled_std == 0 else (float(np.mean(scores_1)) - float(np.mean(scores_2))) / pooled_std
        results.append(
            {
                "model_1": model_1,
                "model_2": model_2,
                "mean_diff": round_float(np.mean(diffs), 3),
                "wilcoxon_stat": round_float(statistic, 2),
                "p_value": round_float(p_value, 6),
                "cohens_d": round_float(cohens_d, 3),
                "n_questions": len(common_questions),
            }
        )
        p_values.append(float(p_value))

    if p_values:
        n_tests = len(p_values)
        sorted_indices = np.argsort(p_values)
        for rank, idx in enumerate(sorted_indices, start=1):
            threshold = rank / n_tests * 0.05
            results[int(idx)]["bh_significant"] = p_values[int(idx)] <= threshold
    return results


def self_evaluation_bias(question_rows: list[dict]) -> dict:
    ceiling_models = sorted({row["model"] for row in question_rows if "ceiling" in row["model"]})
    if not ceiling_models:
        return {
            "ceiling_mean": 0.0,
            "best_open_weight_mean": 0.0,
            "all_open_weight_mean": 0.0,
            "ceiling_gap": 0.0,
            "ceiling_vs_best_gap": 0.0,
            "available": False,
        }

    by_model = defaultdict(list)
    for row in question_rows:
        by_model[row["model"]].append(float(row["mean_score"]))

    ceiling_scores = [score for model in ceiling_models for score in by_model[model]]
    open_weight_models = [model for model in by_model if "ceiling" not in model]
    open_weight_means = {model: float(np.mean(by_model[model])) for model in open_weight_models}
    ceiling_mean = float(np.mean(ceiling_scores))
    best_open = max(open_weight_means.values())
    all_open = float(np.mean([score for model in open_weight_models for score in by_model[model]]))

    return {
        "ceiling_mean": round_float(ceiling_mean, 3),
        "best_open_weight_mean": round_float(best_open, 3),
        "all_open_weight_mean": round_float(all_open, 3),
        "ceiling_gap": round_float(ceiling_mean - all_open, 3),
        "ceiling_vs_best_gap": round_float(ceiling_mean - best_open, 3),
        "available": True,
    }


def write_markdown(result: dict, output_path: Path) -> None:
    lines = [
        "# MARS-Bench v2.0 Statistical Analysis",
        "",
        f"- generated_at: `{result['metadata']['generated_at']}`",
        f"- release_label: `{result['metadata']['release_label']}`",
        f"- benchmark_release_id: `{result['metadata']['benchmark_release_id']}`",
        f"- judge_prompt_version: `{result['metadata']['judge_prompt_version']}`",
        f"- judge_run_number: `{result['metadata']['judge_run_number']}`",
        f"- total_models: `{result['metadata']['total_models']}`",
        f"- total_questions: `{result['metadata']['total_questions']}`",
        "",
        "## Leaderboard by Overall Mean",
        "",
        "| Rank | Model | Mean | 95% CI |",
        "|---|---|---:|---|",
    ]
    leaderboard = sorted(
        result["bootstrap_cis"].items(),
        key=lambda item: item[1]["overall_mean"],
        reverse=True,
    )
    for rank, (model, ci) in enumerate(leaderboard, start=1):
        lines.append(
            f"| {rank} | `{model}` | `{ci['overall_mean']}` | "
            f"`[{ci['ci_95_low']}, {ci['ci_95_high']}]` |"
        )

    lines.extend(
        [
            "",
            "## Pairwise Significant Differences (BH corrected)",
            "",
            "| Model 1 | Model 2 | Mean Diff | p | d | Significant |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in result["pairwise_comparisons"]:
        lines.append(
            f"| `{row['model_1']}` | `{row['model_2']}` | `{row['mean_diff']}` | "
            f"`{row['p_value']}` | `{row['cohens_d']}` | `{row.get('bh_significant', False)}` |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run release-aware MARS-Bench v2.0 statistical analysis.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--release-label", default=DEFAULT_RELEASE_LABEL)
    parser.add_argument("--judge-run-number", type=int, default=DEFAULT_JUDGE_RUN_NUMBER)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    args = parser.parse_args()

    db_path = Path(args.db_path).resolve()
    conn = connect(db_path)
    try:
        release = get_release_row(conn, args.release_label)
        completed_models = get_completed_models(conn, int(release["id"]), args.judge_run_number)
        model_names = [row["model_name"] for row in completed_models]
        raw_rows = fetch_rows(conn, int(release["id"]), args.judge_run_number, model_names)
    finally:
        conn.close()

    question_rows = aggregate_question_means(raw_rows)
    result = {
        "metadata": {
            **metadata_block(
                raw_rows=raw_rows,
                question_rows=question_rows,
                release_label=args.release_label,
                release_id=int(release["id"]),
                judge_prompt_version=str(release["judge_prompt_version"]),
                judge_run_number=args.judge_run_number,
            ),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "bootstrap_samples": args.bootstrap_samples,
            "completed_models": model_names,
        },
        "verbosity_bias": verbosity_bias_analysis(raw_rows),
        "run_to_run_variance": run_to_run_variance(raw_rows),
        "score_distributions": score_distribution_analysis(question_rows),
        "difficulty_levels": difficulty_level_analysis(question_rows),
        "trap_questions": trap_question_analysis(question_rows),
        "self_evaluation_bias": self_evaluation_bias(question_rows),
        "bootstrap_cis": bootstrap_confidence_intervals(question_rows, args.bootstrap_samples),
        "pairwise_comparisons": pairwise_model_comparisons(question_rows),
    }

    stamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    json_path = ROOT / "documentation" / "analysis" / f"{args.release_label}_statistical_analysis_{stamp}.json"
    md_path = ROOT / "documentation" / "analysis" / f"{args.release_label}_statistical_analysis_{stamp}.md"
    current_json = ROOT / "data" / "results" / f"{args.release_label}_statistical_analysis_current.json"
    current_md = ROOT / "data" / "results" / f"{args.release_label}_statistical_analysis_current.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    current_json.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(result, indent=2, sort_keys=True)
    json_path.write_text(payload, encoding="utf-8")
    current_json.write_text(payload, encoding="utf-8")
    write_markdown(result, md_path)
    write_markdown(result, current_md)

    print(json_path)
    print(md_path)
    print(current_json)
    print(current_md)


if __name__ == "__main__":
    main()
