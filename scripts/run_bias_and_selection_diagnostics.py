#!/usr/bin/env python3
"""Run final MARS-Bench model-selection diagnostics on the public export DB."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import numpy as np
from scipy import stats


DEFAULT_DB = Path("data/mars_bench_stats_public.sqlite")
DEFAULT_CLUSTER_JSON = Path("results/cluster_bootstrap/final_2026_05_07_cluster_bootstrap_results.json")
DEFAULT_RESULTS_DIR = Path("results/diagnostics")
DEFAULT_REPORT = Path("reports/final_statistical_methods_and_findings.md")

ROLE_AUDITION_WEIGHTS = {
    "D1": 0.08,
    "D2": 0.17,
    "D3": 0.175,
    "D4": 0.07,
    "D5": 0.14,
    "D6": 0.145,
    "D7": 0.22,
}


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def round_float(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), digits)


def format_p_value(value: float | None) -> str:
    if value is None:
        return "NA"
    if value == 0.0:
        return "< 1e-300"
    if value < 0.001:
        return f"{value:.2e}"
    return str(round_float(value, 4))


def load_paired_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
          p.*,
          d.name AS domain_name,
          d.display_name AS domain_display_name
        FROM paired_codex_claude_scores p
        JOIN domains d ON d.id = p.domain_id
        ORDER BY p.model, p.question_id, p.run_number
        """
    ).fetchall()
    return [dict(row) for row in rows]


def aggregate_question_means(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["model"]), int(row["question_id"]))
        if key not in grouped:
            grouped[key] = {
                "model": row["model"],
                "question_id": int(row["question_id"]),
                "domain_id": int(row["domain_id"]),
                "domain": f"D{int(row['domain_id'])}",
                "level": int(row["level"]),
                "is_trap": int(row["is_trap"]),
                "authoring_input_type": row["authoring_input_type"],
                "scores": [],
                "word_counts": [],
                "run_count": 0,
            }
        grouped[key]["scores"].append(float(row["paired_average_score"]))
        grouped[key]["word_counts"].append(float(row["response_word_count"] or 0))
        grouped[key]["run_count"] += 1
    out = []
    for item in grouped.values():
        scores = item.pop("scores")
        word_counts = item.pop("word_counts")
        item["mean_score"] = float(np.mean(scores))
        item["mean_word_count"] = float(np.mean(word_counts))
        item["score_count"] = len(scores)
        out.append(item)
    return out


def rank_map(values: dict[str, float]) -> dict[str, int]:
    return {model: idx + 1 for idx, (model, _) in enumerate(sorted(values.items(), key=lambda x: (-x[1], x[0])))}


def safe_pearson(xs: list[float], ys: list[float]) -> tuple[float | None, float | None]:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None, None
    r, p = stats.pearsonr(xs, ys)
    return float(r), float(p)


def verbosity_bias(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model"])].append(row)

    model_results = {}
    for model, model_rows in sorted(by_model.items()):
        word_counts = [float(row["response_word_count"] or 0.0) for row in model_rows]
        scores = [float(row["paired_average_score"]) for row in model_rows]
        r, p = safe_pearson(word_counts, scores)
        wc_median = median(word_counts)
        above = [score for wc, score in zip(word_counts, scores) if wc > wc_median]
        below = [score for wc, score in zip(word_counts, scores) if wc <= wc_median]
        inflation = None
        if below and mean(below) > 0:
            inflation = (mean(above) - mean(below)) / mean(below)
        model_results[model] = {
            "n_rows": len(model_rows),
            "avg_word_count": round_float(float(np.mean(word_counts)), 1),
            "median_word_count": round_float(float(wc_median), 1),
            "pearson_r_word_count_score": round_float(r),
            "pearson_p_value": round_float(p, 8),
            "verbosity_inflation_index": round_float(inflation),
            "flag_abs_r_gt_0_30": bool(r is not None and abs(r) > 0.30),
        }
    return model_results


def length_residualized_rank(rows: list[dict[str, Any]]) -> dict[str, Any]:
    score = np.array([float(row["paired_average_score"]) for row in rows])
    log_wc = np.log1p(np.array([float(row["response_word_count"] or 0.0) for row in rows]))
    domains = sorted({int(row["domain_id"]) for row in rows})
    levels = sorted({int(row["level"]) for row in rows})
    regimes = sorted({str(row["authoring_input_type"]) for row in rows})

    cols = [np.ones(len(rows)), log_wc]
    names = ["intercept", "log1p_word_count"]
    for did in domains[1:]:
        cols.append(np.array([1.0 if int(row["domain_id"]) == did else 0.0 for row in rows]))
        names.append(f"domain_D{did}")
    for level in levels[1:]:
        cols.append(np.array([1.0 if int(row["level"]) == level else 0.0 for row in rows]))
        names.append(f"level_L{level}")
    for regime in regimes[1:]:
        cols.append(np.array([1.0 if str(row["authoring_input_type"]) == regime else 0.0 for row in rows]))
        names.append(f"regime_{regime}")
    x = np.column_stack(cols)
    beta, _residuals, rank, _singular = np.linalg.lstsq(x, score, rcond=None)
    fitted = x @ beta
    residuals = score - fitted
    dof = max(len(score) - rank, 1)
    sigma2 = float(np.sum(residuals**2) / dof)
    cov = sigma2 * np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.diag(cov))
    t_value = float(beta[1] / se[1]) if se[1] else None
    p_value = float(2 * stats.t.sf(abs(t_value), dof)) if t_value is not None else None
    log10_p_value = None
    if t_value is not None:
        log10_p_value = float((math.log(2.0) + stats.t.logsf(abs(t_value), dof)) / math.log(10.0))

    mean_log_wc = float(np.mean(log_wc))
    adjusted_scores = score - float(beta[1]) * (log_wc - mean_log_wc)

    raw_by_model: dict[str, list[float]] = defaultdict(list)
    adjusted_by_model: dict[str, list[float]] = defaultdict(list)
    for row, raw, adj in zip(rows, score, adjusted_scores):
        raw_by_model[str(row["model"])].append(float(raw))
        adjusted_by_model[str(row["model"])].append(float(adj))

    raw_means = {model: float(np.mean(vals)) for model, vals in raw_by_model.items()}
    adj_means = {model: float(np.mean(vals)) for model, vals in adjusted_by_model.items()}
    raw_ranks = rank_map(raw_means)
    adj_ranks = rank_map(adj_means)

    model_results = []
    for model in sorted(raw_means, key=lambda m: adj_ranks[m]):
        model_results.append(
            {
                "model": model,
                "raw_mean": round_float(raw_means[model]),
                "raw_rank": raw_ranks[model],
                "length_adjusted_mean": round_float(adj_means[model]),
                "length_adjusted_rank": adj_ranks[model],
                "rank_shift_positive_is_improved": raw_ranks[model] - adj_ranks[model],
            }
        )

    return {
        "model": "OLS: paired_average_score ~ log1p(word_count) + domain FE + level FE + regime FE",
        "coefficient_names": names,
        "log1p_word_count_slope": round_float(float(beta[1]), 6),
        "log1p_word_count_t": round_float(t_value),
        "log1p_word_count_p": p_value,
        "log1p_word_count_log10_p": round_float(log10_p_value, 4),
        "dof": int(dof),
        "model_results": model_results,
    }


def within_level_verbosity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    models = sorted({str(row["model"]) for row in rows})
    for model in models:
        for level in [1, 2, 3]:
            subset = [row for row in rows if str(row["model"]) == model and int(row["level"]) == level]
            r, p = safe_pearson(
                [float(row["response_word_count"] or 0.0) for row in subset],
                [float(row["paired_average_score"]) for row in subset],
            )
            out.append({"model": model, "level": f"L{level}", "n_rows": len(subset), "pearson_r": round_float(r), "p_value": round_float(p, 8)})
    return out


def run_to_run_variance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    meta: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["model"]), int(row["question_id"]))
        grouped[key].append(float(row["paired_average_score"]))
        meta[key] = {"domain_id": int(row["domain_id"]), "level": int(row["level"])}

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    high_cv_questions = []
    for key, scores in grouped.items():
        model, qid = key
        if len(scores) < 2:
            continue
        sd = stdev(scores)
        m = mean(scores)
        variance = sd * sd
        cv = sd / m if m > 0 else None
        item = {
            "model": model,
            "question_id": qid,
            "domain": f"D{meta[key]['domain_id']}",
            "level": f"L{meta[key]['level']}",
            "run_count": len(scores),
            "mean_score": m,
            "sd": sd,
            "variance": variance,
            "cv": cv,
            "scores": scores,
        }
        by_model[model].append(item)
        if cv is not None and cv > 0.15:
            high_cv_questions.append(item)

    model_summary = {}
    for model, items in sorted(by_model.items()):
        cvs = [item["cv"] for item in items if item["cv"] is not None]
        variances = [item["variance"] for item in items]
        model_summary[model] = {
            "question_groups": len(items),
            "mean_variance": round_float(float(np.mean(variances)) if variances else None, 5),
            "mean_cv": round_float(float(np.mean(cvs)) if cvs else None),
            "median_cv": round_float(float(np.median(cvs)) if cvs else None),
            "high_cv_questions_cv_gt_0_15": sum(1 for cv in cvs if cv > 0.15),
        }
    high_cv_questions = sorted(high_cv_questions, key=lambda x: (x["cv"] or 0), reverse=True)
    return {"model_summary": model_summary, "top_high_cv_questions": high_cv_questions[:100]}


def trap_question_analysis(question_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in question_rows:
        by_model[str(row["model"])].append(row)
    out = {}
    for model, items in sorted(by_model.items()):
        trap_scores = [float(row["mean_score"]) for row in items if int(row["is_trap"]) == 1]
        non_trap_scores = [float(row["mean_score"]) for row in items if int(row["is_trap"]) == 0]
        out[model] = {
            "trap_mean": round_float(float(np.mean(trap_scores)) if trap_scores else None),
            "non_trap_mean": round_float(float(np.mean(non_trap_scores)) if non_trap_scores else None),
            "trap_minus_non_trap": round_float(float(np.mean(trap_scores) - np.mean(non_trap_scores)) if trap_scores and non_trap_scores else None),
            "n_trap_questions": len(trap_scores),
            "n_non_trap_questions": len(non_trap_scores),
        }
    return out


def difficulty_gradient(question_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for model in sorted({str(row["model"]) for row in question_rows}):
        items = [row for row in question_rows if str(row["model"]) == model]
        levels = {}
        for level in [1, 2, 3]:
            scores = [float(row["mean_score"]) for row in items if int(row["level"]) == level]
            levels[f"L{level}"] = {"mean": round_float(float(np.mean(scores)) if scores else None), "n_questions": len(scores)}
        l1 = levels["L1"]["mean"]
        l3 = levels["L3"]["mean"]
        levels["L1_minus_L3"] = round_float(l1 - l3 if l1 is not None and l3 is not None else None)
        out[model] = levels
    return out


def score_distribution(question_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for model in sorted({str(row["model"]) for row in question_rows}):
        scores = np.array([float(row["mean_score"]) for row in question_rows if str(row["model"]) == model])
        out[model] = {
            "n_question_means": int(len(scores)),
            "mean": round_float(float(np.mean(scores))),
            "median": round_float(float(np.median(scores))),
            "sd": round_float(float(np.std(scores, ddof=1))),
            "skewness": round_float(float(stats.skew(scores))),
            "kurtosis_fisher": round_float(float(stats.kurtosis(scores))),
            "min": round_float(float(np.min(scores))),
            "max": round_float(float(np.max(scores))),
            "pct_below_2": round_float(float(np.mean(scores < 2.0) * 100), 2),
            "pct_3_plus": round_float(float(np.mean(scores >= 3.0) * 100), 2),
            "pct_4_plus": round_float(float(np.mean(scores >= 4.0) * 100), 2),
        }
    return out


def domain_and_equal_weight_sensitivity(question_rows: list[dict[str, Any]]) -> dict[str, Any]:
    domain_means: dict[str, dict[str, float]] = defaultdict(dict)
    for model in sorted({str(row["model"]) for row in question_rows}):
        for domain_id in range(1, 8):
            scores = [float(row["mean_score"]) for row in question_rows if str(row["model"]) == model and int(row["domain_id"]) == domain_id]
            if scores:
                domain_means[f"D{domain_id}"][model] = float(np.mean(scores))

    all_models = sorted({str(row["model"]) for row in question_rows})
    equal_domain = {}
    role_weighted_observed = {}
    unweighted_observed = {}
    for model in all_models:
        vals = [domain_means[f"D{did}"].get(model) for did in range(1, 8)]
        if all(v is not None for v in vals):
            equal_domain[model] = float(np.mean(vals))
            role_weighted_observed[model] = sum(ROLE_AUDITION_WEIGHTS[f"D{did}"] * domain_means[f"D{did}"][model] for did in range(1, 8))
        all_scores = [float(row["mean_score"]) for row in question_rows if str(row["model"]) == model]
        unweighted_observed[model] = float(np.mean(all_scores))

    return {
        "per_domain_model_means": {
            domain: {model: round_float(score) for model, score in sorted(values.items(), key=lambda x: (-x[1], x[0]))}
            for domain, values in domain_means.items()
        },
        "equal_domain_composite": [
            {"model": model, "score": round_float(score), "rank": idx + 1}
            for idx, (model, score) in enumerate(sorted(equal_domain.items(), key=lambda x: (-x[1], x[0])))
        ],
        "role_weighted_observed_from_question_means": [
            {"model": model, "score": round_float(score), "rank": idx + 1}
            for idx, (model, score) in enumerate(sorted(role_weighted_observed.items(), key=lambda x: (-x[1], x[0])))
        ],
        "unweighted_question_mean": [
            {"model": model, "score": round_float(score), "rank": idx + 1}
            for idx, (model, score) in enumerate(sorted(unweighted_observed.items(), key=lambda x: (-x[1], x[0])))
        ],
    }


def bh_adjust(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = [1.0] * n
    running_min = 1.0
    for rank_from_end, idx in enumerate(reversed(order), start=1):
        rank = n - rank_from_end + 1
        value = p_values[int(idx)] * n / rank
        running_min = min(running_min, value)
        adjusted[int(idx)] = min(running_min, 1.0)
    return adjusted


def pairwise_wilcoxon(question_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    models = sorted({str(row["model"]) for row in question_rows})
    by_model = {
        model: {int(row["question_id"]): float(row["mean_score"]) for row in question_rows if str(row["model"]) == model}
        for model in models
    }
    rows = []
    p_values = []
    for model_a, model_b in combinations(models, 2):
        common = sorted(set(by_model[model_a]) & set(by_model[model_b]))
        a = np.array([by_model[model_a][qid] for qid in common])
        b = np.array([by_model[model_b][qid] for qid in common])
        diffs = a - b
        if len(common) < 2 or np.allclose(diffs, 0):
            stat, p_value = 0.0, 1.0
        else:
            stat, p_value = stats.wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
        pooled_sd = math.sqrt((float(np.var(a, ddof=1)) + float(np.var(b, ddof=1))) / 2.0) if len(common) > 1 else 0.0
        standardized_mean_difference = float((np.mean(a) - np.mean(b)) / pooled_sd) if pooled_sd > 0 else 0.0
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "mean_difference_a_minus_b": round_float(float(np.mean(diffs))),
                "wilcoxon_statistic": round_float(float(stat)),
                "p_value": float(p_value),
                "standardized_mean_difference_pooled_sd": round_float(standardized_mean_difference),
                "n_common_questions": len(common),
            }
        )
        p_values.append(float(p_value))
    adjusted = bh_adjust(p_values)
    for row, q_value in zip(rows, adjusted):
        row["p_value"] = round_float(row["p_value"], 10)
        row["bh_adjusted_p"] = round_float(q_value, 10)
        row["bh_significant_0_05"] = bool(q_value <= 0.05)
    return rows


def response_integrity_profile(conn: sqlite3.Connection) -> dict[str, Any]:
    model_totals = {
        row["model"]: int(row["n"])
        for row in conn.execute(
            """
            SELECT m.name AS model, COUNT(*) AS n
            FROM responses r
            JOIN models m ON m.id = r.model_id
            GROUP BY m.name
            """
        )
    }
    paired_totals = {
        row["model"]: int(row["n"])
        for row in conn.execute("SELECT model, COUNT(*) AS n FROM paired_codex_claude_scores GROUP BY model")
    }
    ri_rows = conn.execute(
        """
        SELECT
          m.name AS model,
          q.domain_id AS domain_id,
          COUNT(DISTINCT r.id) AS n_responses
        FROM evaluations e
        JOIN responses r ON r.id = e.response_id
        JOIN models m ON m.id = r.model_id
        JOIN questions q ON q.id = r.question_id
        WHERE e.error_flags LIKE '%response_integrity_failure%'
           OR e.trap_caps_applied LIKE '%response_integrity_failure%'
        GROUP BY m.name, q.domain_id
        ORDER BY m.name, q.domain_id
        """
    ).fetchall()
    by_model = {model: {"all_release_responses": total, "paired_responses": paired_totals.get(model, 0), "ri_failures": 0, "by_domain": {}} for model, total in model_totals.items()}
    for row in ri_rows:
        model = row["model"]
        count = int(row["n_responses"])
        by_model[model]["ri_failures"] += count
        by_model[model]["by_domain"][f"D{int(row['domain_id'])}"] = count
    for model, item in by_model.items():
        n = item["all_release_responses"]
        failures = item["ri_failures"]
        item["ri_failure_rate"] = round_float(failures / n if n else None, 6)
        item["rule_of_three_upper_95_if_zero"] = round_float(3 / n if n and failures == 0 else None, 6)
    return by_model


def response_set_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    totals = dict(conn.execute(
        """
        WITH paired AS (SELECT DISTINCT response_id FROM paired_codex_claude_scores)
        SELECT
          COUNT(*) AS total_release_responses,
          SUM(CASE WHEN p.response_id IS NOT NULL THEN 1 ELSE 0 END) AS paired_analysis_responses,
          SUM(CASE WHEN p.response_id IS NULL THEN 1 ELSE 0 END) AS nonpaired_responses
        FROM responses r
        LEFT JOIN paired p ON p.response_id = r.id
        """
    ).fetchone())
    totals["release_questions"] = int(conn.execute(
        "SELECT COUNT(*) FROM release_questions WHERE is_included = 1"
    ).fetchone()[0])
    totals["paired_analysis_questions"] = int(conn.execute(
        "SELECT COUNT(DISTINCT question_id) FROM paired_codex_claude_scores"
    ).fetchone()[0])
    totals["models"] = int(conn.execute(
        "SELECT COUNT(DISTINCT model) FROM paired_codex_claude_scores"
    ).fetchone()[0])

    by_domain = [
        dict(row) for row in conn.execute(
            """
            WITH paired AS (SELECT DISTINCT response_id FROM paired_codex_claude_scores)
            SELECT d.id AS domain_id, d.display_name,
                   COUNT(*) AS total_responses,
                   SUM(CASE WHEN p.response_id IS NOT NULL THEN 1 ELSE 0 END) AS paired_responses,
                   SUM(CASE WHEN p.response_id IS NULL THEN 1 ELSE 0 END) AS nonpaired_responses,
                   COUNT(DISTINCT r.question_id) AS total_questions,
                   COUNT(DISTINCT CASE WHEN p.response_id IS NOT NULL THEN r.question_id END) AS paired_questions,
                   COUNT(DISTINCT CASE WHEN p.response_id IS NULL THEN r.question_id END) AS nonpaired_questions
            FROM responses r
            JOIN questions q ON q.id = r.question_id
            JOIN domains d ON d.id = q.domain_id
            LEFT JOIN paired p ON p.response_id = r.id
            GROUP BY d.id, d.display_name
            ORDER BY d.id
            """
        )
    ]

    nonpaired_group = conn.execute(
        """
        WITH paired_questions AS (SELECT DISTINCT question_id FROM paired_codex_claude_scores),
        nonpaired_questions AS (
          SELECT q.id, q.text, rq.track, rq.tier, rq.authoring_input_type
          FROM release_questions rq
          JOIN questions q ON q.id = rq.question_id
          LEFT JOIN paired_questions pq ON pq.question_id = q.id
          WHERE rq.is_included = 1 AND pq.question_id IS NULL
        )
        SELECT track, tier, authoring_input_type, COUNT(*) AS questions,
               SUM(CASE WHEN text LIKE '%Greek text%' THEN 1 ELSE 0 END) AS greek_nt_questions,
               SUM(CASE WHEN text LIKE '%Septuagint%' THEN 1 ELSE 0 END) AS septuagint_questions,
               SUM(CASE WHEN text LIKE '%Biblical Hebrew%' THEN 1 ELSE 0 END) AS hebrew_questions,
               SUM(CASE WHEN text LIKE '%Biblical Aramaic%' THEN 1 ELSE 0 END) AS aramaic_questions
        FROM nonpaired_questions
        GROUP BY track, tier, authoring_input_type
        """
    ).fetchone()

    nonpaired_eval_coverage = [
        dict(row) for row in conn.execute(
            """
            WITH paired AS (SELECT DISTINCT response_id FROM paired_codex_claude_scores),
            nonpaired AS (
              SELECT r.id AS response_id
              FROM responses r LEFT JOIN paired p ON p.response_id = r.id
              WHERE p.response_id IS NULL
            )
            SELECT e.judge_model, e.judge_prompt_version,
                   COUNT(*) AS evaluation_rows,
                   COUNT(DISTINCT e.response_id) AS distinct_responses
            FROM nonpaired u
            JOIN evaluations e ON e.response_id = u.response_id
            GROUP BY e.judge_model, e.judge_prompt_version
            ORDER BY evaluation_rows DESC
            """
        )
    ]
    return {
        **totals,
        "by_domain": by_domain,
        "nonpaired_question_group": dict(nonpaired_group) if nonpaired_group else None,
        "nonpaired_evaluation_coverage": nonpaired_eval_coverage,
        "interpretation": (
            "The 924 nonpaired responses are current-release D1 structured morphology "
            "adjunct responses: 22 questions x 14 models x 3 runs. They were judged "
            "by d1-structured-morphology-scorer-2026-04-25 and have no final paired "
            "Codex-Claude evaluations, so they are excluded from paired model-selection statistics."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def flatten_model_dict(data: dict[str, dict[str, Any]], model_key: str = "model") -> list[dict[str, Any]]:
    rows = []
    for model, values in data.items():
        row = {model_key: model}
        row.update(values)
        rows.append(row)
    return rows


def top_models_from_cluster(cluster: dict[str, Any], n: int = 10) -> list[dict[str, Any]]:
    return cluster["analysis"]["weighted_composite_leaderboard"][:n]


def write_report(
    path: Path,
    db_path: Path,
    cluster_path: Path,
    diagnostics: dict[str, Any],
    cluster: dict[str, Any],
) -> None:
    metadata = cluster["metadata"]
    coverage = cluster["coverage"]
    response_sets = diagnostics["response_set_summary"]
    reliability = cluster["analysis"]["interjudge_reliability"]
    role_top = top_models_from_cluster(cluster, 8)
    paired_diffs = cluster["analysis"].get("paired_composite_differences", [])
    residual_top = diagnostics["length_residualized_rank"]["model_results"][:8]
    run_var_sorted = sorted(
        diagnostics["run_to_run_variance"]["model_summary"].items(),
        key=lambda item: item[1]["mean_cv"] if item[1]["mean_cv"] is not None else 999,
    )
    ri_sorted = sorted(
        diagnostics["response_integrity_profile"].items(),
        key=lambda item: (-item[1]["ri_failures"], item[0]),
    )

    lines = [
        "# MARS-Bench Final Statistical Measurements",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        "",
        "## Raw Data Locations",
        "",
        f"- Publication-safe SQLite DB: `{db_path}`",
        f"- Compressed public DB artifact: `{db_path}.gz`",
        f"- Cluster bootstrap output: `{cluster_path}`",
        f"- Source working DB filename: `{diagnostics['export_metadata'].get('source_db_filename')}`",
        f"- Source working DB SHA-256: `{diagnostics['export_metadata'].get('source_db_sha256')}`",
        "",
        "The public DB is release-scoped and redacts local paths and raw long-form source/context packet body text. It retains questions, reference answers, model responses, scores, rubrics, judge notes, audit flags, material hashes, and structured source/audit provenance needed for statistical replication.",
        "",
        "## Corpus and Judging Coverage",
        "",
        f"- Release: `{metadata['release_label']}`",
        f"- Release questions in public DB: `{response_sets['release_questions']}`",
        f"- Release responses in public DB: `{response_sets['total_release_responses']}`",
        f"- Paired-analysis questions: `{response_sets['paired_analysis_questions']}`",
        f"- Paired Codex-Claude response rows: `{coverage['paired_rows']}`",
        f"- Nonpaired release responses outside paired model-selection analysis: `{response_sets['nonpaired_responses']}`",
        f"- Codex rows matching final filter: `{coverage['codex_rows_matching_filter']}`",
        f"- Claude rows matching final filter: `{coverage['claude_rows_matching_filter']}`",
        f"- Paired fraction of Codex rows: `{coverage['paired_fraction_of_codex']:.3f}`",
        f"- Paired fraction of Claude rows: `{coverage['paired_fraction_of_claude']:.3f}`",
        f"- Bootstrap reps: `{metadata['bootstrap_reps']}`; seed: `{metadata['seed']}`; interval: `{metadata['bootstrap_interval_type']}`",
        "",
        "The 924 nonpaired rows are not stray legacy responses. They are current-release D1 structured morphology adjunct responses: 22 questions x 14 models x 3 runs, judged only by `d1-structured-morphology-scorer-2026-04-25`. Because they have no final paired Codex-Claude evaluations, they are excluded from the paired model-selection statistics and should be analyzed only with a morphology-specific metric.",
        "",
        "## Measurement 1: Inter-Judge Reliability",
        "",
        "Purpose: verify that the two independent judge lanes are measuring the same response-quality construct before using averaged scores for model selection.",
        "",
        "Formulae:",
        "",
        "- Mean signed judge difference: `mean(Claude_i - Codex_i)`.",
        "- Mean absolute difference: `mean(abs(Claude_i - Codex_i))`.",
        "- Two-way mixed-effects absolute-agreement ICC uses McGraw and Wong's ICC(A,1) form:",
        "  `ICC(A,1) = (MS_R - MS_E) / (MS_R + (k - 1)MS_E + k(MS_C - MS_E)/n)`.",
        "- For the averaged two-judge score: `ICC(A,k) = (MS_R - MS_E) / (MS_R + (MS_C - MS_E)/n)` with `k = 2`.",
        "- Bland-Altman normal limits: `mean(diff) +/- 1.96 * SD(diff)`.",
        "- Bland-Altman empirical limits: the `2.5th` and `97.5th` percentiles of the paired differences.",
        "",
        "Findings:",
        "",
        f"- ICC(A,1): `{reliability['icc_a1']}` with bootstrap CI `{reliability['bootstrap_ci']['icc_a1']}`.",
        f"- ICC(A,2): `{reliability['icc_a2']}` with bootstrap CI `{reliability['bootstrap_ci']['icc_a2']}`.",
        f"- Mean signed difference, Claude minus Codex: `{reliability['mean_signed_difference_claude_minus_codex']}`.",
        f"- Mean absolute difference: `{reliability['mean_absolute_difference']}`.",
        f"- Pearson / Spearman: `{reliability['pearson_correlation']}` / `{reliability['spearman_correlation']}`.",
        f"- Within 0.50 points: `{reliability['within_0_50_rate']}`; within 1.00 point: `{reliability['within_1_00_rate']}`.",
        f"- Bland-Altman normal limits: `[{reliability['bland_altman_low_normal']}, {reliability['bland_altman_high_normal']}]`.",
        f"- Bland-Altman empirical limits: `[{reliability['bland_altman_low_empirical']}, {reliability['bland_altman_high_empirical']}]`.",
        "",
        "Interpretation: ICC supports aggregate use of the averaged Codex-Claude score. The Bland-Altman tail width means row-level disagreements still deserve qualitative audit, especially in contested source-critical cases.",
        "",
        "## Measurement 2: Role-Audition Weighted Composite",
        "",
        "Purpose: rank models by the domain mix most relevant to Synthetic Theological Agent base-model selection, rather than by raw row count.",
        "",
        "Formula:",
        "",
        "`Composite_m = 0.08*D1_m + 0.17*D2_m + 0.175*D3_m + 0.07*D4_m + 0.14*D5_m + 0.145*D6_m + 0.22*D7_m`.",
        "",
        "Top role-weighted models:",
        "",
        "| Rank | Model | Composite | 95% CI | Top-3 bootstrap rate |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in role_top:
        lines.append(
            f"| {row['composite_observed_rank']} | `{row['model']}` | `{row['composite_observed']}` | "
            f"`[{row['composite_ci_95_low']}, {row['composite_ci_95_high']}]` | `{row['composite_top_3_rate']}` |"
        )
    lines.extend(
        [
            "",
            "Paired composite difference tests:",
            "",
        ]
    )
    for diff in paired_diffs:
        lines.append(
            f"- `{diff['model_a']} - {diff['model_b']}` observed difference `{diff['observed_difference']}`, "
            f"bootstrap CI `[{diff['bootstrap_ci_low']}, {diff['bootstrap_ci_high']}]`, includes zero: `{diff['ci_includes_zero']}`."
        )
    lines.extend(
        [
            "",
            "## Measurement 3: Verbosity Bias and Length Adjustment",
            "",
            "Purpose: detect whether longer responses receive higher scores independent of domain, level, and regime. This matters because verbose theological prose can appear scholarly without being more accurate.",
            "",
            "Formulae:",
            "",
            "- Raw length-score association: Pearson `r(word_count, score)` within each model.",
            "- Verbosity inflation index: `(mean(score | words > median_words) - mean(score | words <= median_words)) / mean(score | words <= median_words)`.",
            "- Length-adjusted score: fit `score ~ log(1 + words) + domain FE + level FE + regime FE`; then adjust each score to the global mean log length: `score_adj_i = score_i - beta_len*(log_words_i - mean(log_words))`.",
            "",
            f"Overall log-length coefficient: `{diagnostics['length_residualized_rank']['log1p_word_count_slope']}`; p-value `{format_p_value(diagnostics['length_residualized_rank']['log1p_word_count_p'])}`.",
            "",
            "Top length-adjusted rankings:",
            "",
            "| Adjusted Rank | Model | Raw Mean | Raw Rank | Length-Adjusted Mean | Rank Shift |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in residual_top:
        lines.append(
            f"| {row['length_adjusted_rank']} | `{row['model']}` | `{row['raw_mean']}` | `{row['raw_rank']}` | "
            f"`{row['length_adjusted_mean']}` | `{row['rank_shift_positive_is_improved']}` |"
        )
    lines.extend(
        [
            "",
            "## Measurement 4: Run-to-Run Variance",
            "",
            "Purpose: assess whether a model is stable across repeated generations for the same question. Stable base models are preferable for STA work because rare erratic behavior can become costly after fine-tuning.",
            "",
            "Formulae:",
            "",
            "- For model `m` and question `q`, `variance_mq = sample_variance(score_mq1, score_mq2, ...)`.",
            "- Coefficient of variation: `CV_mq = SD_mq / mean_mq` when `mean_mq > 0`.",
            "- A high-CV question is flagged at `CV > 0.15`.",
            "",
            "Lowest mean-CV models:",
            "",
            "| Model | Mean CV | Median CV | High-CV Questions |",
            "|---|---:|---:|---:|",
        ]
    )
    for model, item in run_var_sorted[:8]:
        lines.append(f"| `{model}` | `{item['mean_cv']}` | `{item['median_cv']}` | `{item['high_cv_questions_cv_gt_0_15']}` |")
    lines.extend(
        [
            "",
            "## Measurement 5: Trap and Response-Integrity Resilience",
            "",
            "Purpose: distinguish ordinary weak answers from failure modes that make a model risky for scholarly deployment: traps, self-talk, loops, generic-Gnostic confusion, unsupported claims, and hard response-integrity failures.",
            "",
            "Formulae:",
            "",
            "- Trap delta: `mean(trap question scores) - mean(non-trap question scores)` using question-level model means.",
            "- Response-integrity failure rate: `RI_failures / total_release_responses`.",
            "- If a model has zero RI failures, the approximate one-sided 95% upper bound is the rule of three: `3 / n`.",
            "",
            "Response-integrity profile:",
            "",
            "| Model | RI Failures | All Release Responses | RI Rate | Zero-Failure Upper Bound | Domains |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for model, item in ri_sorted:
        lines.append(
            f"| `{model}` | `{item['ri_failures']}` | `{item['all_release_responses']}` | `{item['ri_failure_rate']}` | "
            f"`{item['rule_of_three_upper_95_if_zero']}` | `{json.dumps(item['by_domain'], sort_keys=True)}` |"
        )
    lines.extend(
        [
            "",
            "## Measurement 6: Difficulty Gradient",
            "",
            "Purpose: check whether models collapse from foundational recall (L1) to synthetic expert reasoning (L3). This directly supports the paper's impersonation-versus-improvisation framing.",
            "",
            "Formula: `L1_minus_L3 = mean_L1_score - mean_L3_score`. Larger positive values indicate a larger drop at expert/synthetic difficulty.",
            "",
            "## Measurement 7: Score Distribution and Floor Tail",
            "",
            "Purpose: means can hide brittle models. The distribution analysis reports median, standard deviation, skewness, kurtosis, minimum/maximum, percent below 2, percent at least 3, and percent at least 4 using question-level model means.",
            "",
            "## Measurement 8: Pairwise Wilcoxon Comparisons",
            "",
            "Purpose: provide appendix-level pairwise tests over question-level model means. This is secondary to the role-weighted bootstrap because the paper's selection rule is multi-criteria rather than a p-value leaderboard.",
            "",
            "Formulae:",
            "",
            "- Paired difference for each common question: `d_q = score_Aq - score_Bq`.",
            "- Wilcoxon signed-rank tests whether the median paired difference is zero.",
            "- Benjamini-Hochberg adjusted p-values are reported as an FDR-oriented multiplicity correction under the usual independence/positive-dependence assumptions.",
            "- The reported standardized mean difference is descriptive: `mean(A - B) / pooled_SD`; it is not a paired-samples `d_z`.",
            "",
            "## Files Produced",
            "",
            "- `results/cluster_bootstrap/final_2026_05_07_cluster_bootstrap_results.json`",
            "- `results/cluster_bootstrap/final_2026_05_07_cluster_bootstrap_leaderboard.csv`",
            "- `results/cluster_bootstrap/final_2026_05_07_interjudge_reliability.json`",
            "- `results/diagnostics/final_bias_and_selection_diagnostics.json`",
            "- `results/diagnostics/*.csv` appendix tables",
            "",
            "## References",
            "",
            "- Bland, J. M., and Altman, D. G. 1986. Statistical methods for assessing agreement between two methods of clinical measurement. The Lancet 327:307-310.",
            "- Benjamini, Y., and Hochberg, Y. 1995. Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society B 57:289-300.",
            "- Cohen, J. 1988. Statistical Power Analysis for the Behavioral Sciences. 2nd ed.",
            "- Efron, B., and Tibshirani, R. J. 1993. An Introduction to the Bootstrap.",
            "- Hanley, J. A., and Lippman-Hand, A. 1983. If nothing goes wrong, is everything all right? JAMA 249:1743-1745.",
            "- Koo, T. K., and Li, M. Y. 2016. A guideline of selecting and reporting intraclass correlation coefficients for reliability research. Journal of Chiropractic Medicine 15:155-163.",
            "- Lakens, D. 2017. Equivalence tests: a practical primer for t tests, correlations, and meta-analyses. Social Psychological and Personality Science 8:355-362.",
            "- McGraw, K. O., and Wong, S. P. 1996. Forming inferences about some intraclass correlation coefficients. Psychological Methods 1:30-46.",
            "- Schenker, N., and Gentleman, J. F. 2001. On judging the significance of differences by examining the overlap between confidence intervals. The American Statistician 55:182-186.",
            "- Shrout, P. E., and Fleiss, J. L. 1979. Intraclass correlations: uses in assessing rater reliability. Psychological Bulletin 86:420-428.",
            "- Wilcoxon, F. 1945. Individual comparisons by ranking methods. Biometrics Bulletin 1:80-83.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final MARS-Bench bias and selection diagnostics.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--cluster-json", type=Path, default=DEFAULT_CLUSTER_JSON)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    conn = connect(args.db)
    rows = load_paired_rows(conn)
    question_rows = aggregate_question_means(rows)
    export_metadata = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM publication_export_metadata")}

    cluster = json.loads(args.cluster_json.read_text(encoding="utf-8"))
    diagnostics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "db_path": str(args.db),
        "cluster_json": str(args.cluster_json),
        "export_metadata": export_metadata,
        "row_counts": {
            "paired_rows": len(rows),
            "question_mean_rows": len(question_rows),
            "models": len({row["model"] for row in rows}),
            "questions": len({row["question_id"] for row in rows}),
        },
        "verbosity_bias": verbosity_bias(rows),
        "length_residualized_rank": length_residualized_rank(rows),
        "within_level_verbosity": within_level_verbosity(rows),
        "run_to_run_variance": run_to_run_variance(rows),
        "trap_question_analysis": trap_question_analysis(question_rows),
        "difficulty_gradient": difficulty_gradient(question_rows),
        "score_distribution": score_distribution(question_rows),
        "domain_and_equal_weight_sensitivity": domain_and_equal_weight_sensitivity(question_rows),
        "pairwise_wilcoxon_question_means": pairwise_wilcoxon(question_rows),
        "response_integrity_profile": response_integrity_profile(conn),
        "response_set_summary": response_set_summary(conn),
    }
    conn.close()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = args.results_dir / "final_bias_and_selection_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    write_csv(args.results_dir / "verbosity_bias.csv", flatten_model_dict(diagnostics["verbosity_bias"]))
    write_csv(args.results_dir / "length_residualized_rank.csv", diagnostics["length_residualized_rank"]["model_results"])
    write_csv(args.results_dir / "within_level_verbosity.csv", diagnostics["within_level_verbosity"])
    write_csv(args.results_dir / "run_to_run_variance_by_model.csv", flatten_model_dict(diagnostics["run_to_run_variance"]["model_summary"]))
    write_csv(args.results_dir / "top_high_cv_questions.csv", diagnostics["run_to_run_variance"]["top_high_cv_questions"])
    write_csv(args.results_dir / "trap_question_analysis.csv", flatten_model_dict(diagnostics["trap_question_analysis"]))
    write_csv(args.results_dir / "score_distribution.csv", flatten_model_dict(diagnostics["score_distribution"]))
    write_csv(args.results_dir / "pairwise_wilcoxon_question_means.csv", diagnostics["pairwise_wilcoxon_question_means"])
    write_csv(args.results_dir / "response_integrity_profile.csv", flatten_model_dict(diagnostics["response_integrity_profile"]))

    difficulty_rows = []
    for model, values in diagnostics["difficulty_gradient"].items():
        row = {"model": model, "L1_minus_L3": values["L1_minus_L3"]}
        for level in ["L1", "L2", "L3"]:
            row[f"{level}_mean"] = values[level]["mean"]
            row[f"{level}_n_questions"] = values[level]["n_questions"]
        difficulty_rows.append(row)
    write_csv(args.results_dir / "difficulty_gradient.csv", difficulty_rows)

    write_csv(args.results_dir / "equal_domain_composite.csv", diagnostics["domain_and_equal_weight_sensitivity"]["equal_domain_composite"])
    write_csv(args.results_dir / "role_weighted_observed_from_question_means.csv", diagnostics["domain_and_equal_weight_sensitivity"]["role_weighted_observed_from_question_means"])
    write_csv(args.results_dir / "unweighted_question_mean.csv", diagnostics["domain_and_equal_weight_sensitivity"]["unweighted_question_mean"])

    write_report(args.report, args.db, args.cluster_json, diagnostics, cluster)
    print(f"Wrote diagnostics JSON: {diagnostics_path}")
    print(f"Wrote report:           {args.report}")


if __name__ == "__main__":
    main()
