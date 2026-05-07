#!/usr/bin/env python3
"""
Cluster-aware bootstrap analysis for MARS-Bench judged results.

This script does not call any LLMs. It reads completed evaluation rows from the
SQLite database, keeps Codex/Claude judgments paired by response_id, and
resamples questions within domains to estimate score, rank, and inter-judge
reliability uncertainty.

It is designed to run on partial judging snapshots and on the final locked
snapshot. Partial snapshots are labeled clearly in the output metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sqlite3
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_DB_PATH = Path("data/mars_bench_stats_public.sqlite")
DEFAULT_OUTPUT_DIR = Path("results/cluster_bootstrap")
DEFAULT_CODEX_MODEL = "gpt-5.5-medium-codex-cli"
DEFAULT_CLAUDE_MODEL = "claude-opus-4-7-medium-claude-cli"
DEFAULT_CODEX_VERSION_LIKE = "%codex_gpt_5_5_medium_primary"
DEFAULT_CLAUDE_VERSION_LIKE = "%claude_opus_4_7_medium_comparison"

# Role-audition composite weights, per the philology weighting supplement
# (`supplement_philology_weighting_methodology_2026_05_02.md` §5).
# Sum must equal 1.0 (verified at function entry).
ROLE_AUDITION_WEIGHTS: dict[str, float] = {
    "D1": 0.08,    # Ancient Languages (philology bounded against root-word display)
    "D2": 0.17,    # Biblical Knowledge
    "D3": 0.175,   # Patristic Knowledge
    "D4": 0.07,    # Early Christian History
    "D5": 0.14,    # Theological Knowledge
    "D6": 0.145,   # Heresiological Knowledge
    "D7": 0.22,    # Marcion Studies (highest weight, role-fidelity primary)
}

# Model pairs whose paired-difference CI on the role-audition weighted
# composite is computed during the bootstrap. For each (a, b) pair, the
# script accumulates Δ_b = composite(a) - composite(b) over the B
# bootstrap resamples and reports the percentile 95% CI of Δ_b plus
# whether the CI includes zero.
#
# Why a paired-difference CI rather than overlap of marginal CIs:
# overlapping marginal CIs are necessary-but-not-sufficient for
# non-distinguishability (Schenker and Gentleman 2001, "On Judging the
# Significance of Differences by Examining the Overlap Between Confidence
# Intervals", The American Statistician 55:182-186). The paired-difference
# CI uses the per-resample covariance between the two models' composites
# and is the canonical correct test of distinguishability.
#
# Currently configured for the §6 H-F bridge's two leading STA-eligible
# candidates (Llama 3.3 70B Instruct and Gemma 4 31B-IT). Add further
# pairs here when their paired-difference is methodologically
# load-bearing in the paper.
PAIRED_DIFFERENCE_PAIRS: list[tuple[str, str]] = [
    ("llama-3.3-70b-instruct", "gemma-4-31b-it"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cluster-aware bootstrap analysis on paired MARS-Bench judge scores."
    )
    parser.add_argument("--db", type=Path, default=Path(os.environ.get("MARS_BENCH_DB", DEFAULT_DB_PATH)))
    parser.add_argument("--release-label", help="Benchmark release label. Defaults to latest candidate/frozen release.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="interim", help="Output label prefix, e.g. interim or final.")
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--min-paired-rows", type=int, default=1)
    parser.add_argument("--codex-model", default=DEFAULT_CODEX_MODEL)
    parser.add_argument("--claude-model", default=DEFAULT_CLAUDE_MODEL)
    parser.add_argument("--codex-version-like", default=DEFAULT_CODEX_VERSION_LIKE)
    parser.add_argument("--claude-version-like", default=DEFAULT_CLAUDE_VERSION_LIKE)
    return parser.parse_args()


def connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def latest_release(conn: sqlite3.Connection) -> sqlite3.Row:
    row = conn.execute(
        """
        SELECT id, release_label, status, created_at
        FROM benchmark_releases
        WHERE status IN ('candidate', 'frozen')
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        raise SystemExit("No candidate/frozen benchmark release found.")
    return row


def release_by_label(conn: sqlite3.Connection, release_label: str | None) -> sqlite3.Row:
    if not release_label:
        return latest_release(conn)
    row = conn.execute(
        """
        SELECT id, release_label, status, created_at
        FROM benchmark_releases
        WHERE release_label = ?
        """,
        (release_label,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"Release not found: {release_label}")
    return row


def load_paired_rows(conn: sqlite3.Connection, release_id: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            r.id AS response_id,
            r.run_number AS response_run_number,
            m.name AS model,
            m.family AS model_family,
            q.id AS question_id,
            q.domain_id AS domain_id,
            d.name AS domain_name,
            d.display_name AS domain_display_name,
            q.is_trap AS is_trap,
            c.overall_score AS codex_score,
            cl.overall_score AS claude_score,
            c.judge_prompt_version AS codex_judge_prompt_version,
            cl.judge_prompt_version AS claude_judge_prompt_version
        FROM responses r
        JOIN models m ON r.model_id = m.id
        JOIN questions q ON r.question_id = q.id
        JOIN domains d ON q.domain_id = d.id
        JOIN evaluations c
            ON c.response_id = r.id
           AND c.judge_model = ?
           AND c.judge_prompt_version LIKE ?
        JOIN evaluations cl
            ON cl.response_id = r.id
           AND cl.judge_model = ?
           AND cl.judge_prompt_version LIKE ?
        WHERE r.benchmark_release_id = ?
          AND q.retired_at IS NULL
        ORDER BY d.id, q.id, m.name, r.run_number
        """,
        (
            args.codex_model,
            args.codex_version_like,
            args.claude_model,
            args.claude_version_like,
            release_id,
        ),
    ).fetchall()

    paired = []
    for row in rows:
        item = dict(row)
        item["paired_average_score"] = (float(item["codex_score"]) + float(item["claude_score"])) / 2.0
        item["judge_difference_claude_minus_codex"] = float(item["claude_score"]) - float(item["codex_score"])
        paired.append(item)

    # Guard: the version-LIKE filter must produce exactly one Codex and one Claude
    # evaluation per response_id. If duplicates exist (e.g. a primary judging run
    # plus a rerun whose judge_prompt_version both end with the LIKE suffix), the
    # two self-joins above produce a Cartesian cross-product per response and
    # silently inflate every downstream statistic. Stop loudly if that happens
    # rather than emit biased numbers.
    seen_ids = {row.get("response_id") for row in paired}
    if len(seen_ids) != len(paired):
        raise SystemExit(
            f"load_paired_rows: paired-row count ({len(paired)}) != distinct response_id count "
            f"({len(seen_ids)}). The Codex/Claude version-LIKE filter matched multiple rows for "
            "at least one response_id, producing a Cartesian cross-product. Tighten the filter "
            "(e.g. constrain run_number or use exact judge_prompt_version)."
        )
    return paired


def count_single_judge_rows(conn: sqlite3.Connection, release_id: int, judge_model: str, version_like: str) -> int:
    return int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM responses r
            JOIN evaluations e
              ON e.response_id = r.id
             AND e.judge_model = ?
             AND e.judge_prompt_version LIKE ?
            JOIN questions q ON q.id = r.question_id
            WHERE r.benchmark_release_id = ?
              AND q.retired_at IS NULL
            """,
            (judge_model, version_like, release_id),
        ).fetchone()[0]
    )


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (pct / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def rounded(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def sample_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_denom = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_denom = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    denom = x_denom * y_denom
    return numerator / denom if denom else None


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        end = idx
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[idx][1]:
            end += 1
        avg_rank = (idx + 1 + end + 1) / 2.0
        for pos in range(idx, end + 1):
            original_idx = indexed[pos][0]
            ranks[original_idx] = avg_rank
        idx = end + 1
    return ranks


def spearman_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson_corr(average_ranks(xs), average_ranks(ys))


def icc_absolute_agreement(scores: list[tuple[float, float]]) -> dict[str, float | None]:
    """Two-way absolute-agreement ICC for two fixed judge lanes.

    Uses the McGraw/Wong ICC(A,1) and ICC(A,k) form. With two judges, k=2.
    """
    n = len(scores)
    k = 2
    if n < 2:
        return {"icc_a1": None, "icc_a2": None}

    row_means = [(a + b) / 2.0 for a, b in scores]
    col_means = [mean([a for a, _ in scores]), mean([b for _, b in scores])]
    grand = mean(row_means)

    ss_rows = k * sum((row_mean - grand) ** 2 for row_mean in row_means)
    ss_cols = n * sum((col_mean - grand) ** 2 for col_mean in col_means)
    ss_err = 0.0
    for idx, (a, b) in enumerate(scores):
        ss_err += (a - row_means[idx] - col_means[0] + grand) ** 2
        ss_err += (b - row_means[idx] - col_means[1] + grand) ** 2

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))

    denom_a1 = ms_rows + (k - 1) * ms_err + (k * (ms_cols - ms_err) / n)
    denom_ak = ms_rows + ((ms_cols - ms_err) / n)
    icc_a1 = (ms_rows - ms_err) / denom_a1 if denom_a1 else None
    icc_ak = (ms_rows - ms_err) / denom_ak if denom_ak else None
    return {"icc_a1": icc_a1, "icc_a2": icc_ak}


def group_questions_by_domain(rows: list[dict[str, Any]]) -> dict[int, dict[int, list[dict[str, Any]]]]:
    grouped: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[int(row["domain_id"])][int(row["question_id"])].append(row)
    return grouped


def resample_questions_within_domains(
    grouped: dict[int, dict[int, list[dict[str, Any]]]], rng: random.Random
) -> list[dict[str, Any]]:
    sample: list[dict[str, Any]] = []
    for _domain_id, questions in grouped.items():
        qids = list(questions.keys())
        if not qids:
            continue
        for _ in qids:
            qid = rng.choice(qids)
            sample.extend(questions[qid])
    return sample


def model_means(rows: list[dict[str, Any]], score_key: str = "paired_average_score") -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(float(row[score_key]))
    return {model: mean(scores) for model, scores in grouped.items() if scores}


def domain_model_means(rows: list[dict[str, Any]], score_key: str = "paired_average_score") -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        domain = f"D{int(row['domain_id'])}"
        grouped[domain][str(row["model"])].append(float(row[score_key]))
    return {
        domain: {model: mean(scores) for model, scores in model_scores.items() if scores}
        for domain, model_scores in grouped.items()
    }


def weighted_composite_means(
    domain_model_means_map: dict[str, dict[str, float]],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute the role-audition weighted composite per model from per-domain means.

    For each model, composite = sum_d weights[d] * domain_model_means_map[d][model],
    where d ranges over the weighted domains.

    A model is included in the output only if it has a per-domain mean for every
    weighted domain. Models missing from any weighted domain are excluded — this
    is loud (callers can detect the missing entry) rather than silently treating
    a missing domain as zero.

    The weights argument defaults to ROLE_AUDITION_WEIGHTS (per the philology
    weighting supplement). The function asserts that weights sum to 1.0 within
    float tolerance to prevent silent miscalibration if the weights are edited.
    """
    if weights is None:
        weights = ROLE_AUDITION_WEIGHTS
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-9:
        raise ValueError(
            f"role-audition weights must sum to 1.0; got {total_weight} from {weights}"
        )

    # Collect every model that appears in any domain
    all_models: set[str] = set()
    for model_means in domain_model_means_map.values():
        all_models.update(model_means.keys())

    composites: dict[str, float] = {}
    for model in all_models:
        composite = 0.0
        complete = True
        for domain, weight in weights.items():
            if domain not in domain_model_means_map or model not in domain_model_means_map[domain]:
                complete = False
                break
            composite += weight * domain_model_means_map[domain][model]
        if complete:
            composites[model] = composite
    return composites


def paired_composite_differences(
    composites: dict[str, float],
    pairs: list[tuple[str, str]] | None = None,
) -> dict[str, float]:
    """For each (a, b) in `pairs`, compute composites[a] - composites[b].

    Returns a dict keyed by f"{a}__minus__{b}" → float difference. Pairs
    where either model is missing from `composites` are silently skipped
    (this happens when a model lacks a per-domain mean for some weighted
    domain in a particular bootstrap resample, which weighted_composite_means
    handles by excluding that model from the composite output).

    The defaults to PAIRED_DIFFERENCE_PAIRS when pairs is None.
    """
    if pairs is None:
        pairs = PAIRED_DIFFERENCE_PAIRS
    out: dict[str, float] = {}
    for a, b in pairs:
        if a in composites and b in composites:
            out[f"{a}__minus__{b}"] = composites[a] - composites[b]
    return out


def rank_models(means: dict[str, float]) -> dict[str, int]:
    # Tiebreak by alphabetical model name. At float precision on the production
    # paired-row counts, exact ties on model means do not occur in practice; the
    # alphabetical tiebreak is therefore documented behavior, not a hidden bias.
    ordered = sorted(means.items(), key=lambda item: (-item[1], item[0]))
    return {model: idx + 1 for idx, (model, _score) in enumerate(ordered)}


def reliability_summary(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    diffs = [float(row["judge_difference_claude_minus_codex"]) for row in rows]
    abs_diffs = [abs(diff) for diff in diffs]
    codex_scores = [float(row["codex_score"]) for row in rows]
    claude_scores = [float(row["claude_score"]) for row in rows]
    pairs = [(float(row["codex_score"]), float(row["claude_score"])) for row in rows]
    icc = icc_absolute_agreement(pairs)
    diff_mean = mean_or_none(diffs)
    diff_sd = sample_sd(diffs)
    # Bland-Altman normal-theory limits of agreement: assumes diffs ~ Normal.
    bland_altman_low_normal = diff_mean - 1.96 * diff_sd if diff_mean is not None and diff_sd is not None else None
    bland_altman_high_normal = diff_mean + 1.96 * diff_sd if diff_mean is not None and diff_sd is not None else None
    # Empirical 2.5/97.5 percentiles of diffs: distribution-free analogue.
    bland_altman_low_empirical = percentile(diffs, 2.5)
    bland_altman_high_empirical = percentile(diffs, 97.5)
    n = len(rows)
    # exact_same_score uses an epsilon comparison rather than `== 0.0` so float drift
    # in stored scores (e.g. 2.50000000001) does not silently undercount exact matches.
    exact_eps = 1e-9
    return {
        "paired_rows": n,
        "mean_codex_score": mean_or_none(codex_scores),
        "mean_claude_score": mean_or_none(claude_scores),
        "mean_signed_difference_claude_minus_codex": diff_mean,
        "mean_absolute_difference": mean_or_none(abs_diffs),
        "pearson_correlation": pearson_corr(codex_scores, claude_scores),
        "spearman_correlation": spearman_corr(codex_scores, claude_scores),
        "exact_same_score_rate": sum(1 for diff in abs_diffs if diff < exact_eps) / n if n else None,
        "within_0_25_rate": sum(1 for diff in abs_diffs if diff <= 0.25) / n if n else None,
        "within_0_50_rate": sum(1 for diff in abs_diffs if diff <= 0.50) / n if n else None,
        "within_1_00_rate": sum(1 for diff in abs_diffs if diff <= 1.00) / n if n else None,
        # Legacy keys preserved as aliases of the normal-theory bounds for back-compat;
        # new consumers should prefer the *_normal / *_empirical keys.
        "bland_altman_low": bland_altman_low_normal,
        "bland_altman_high": bland_altman_high_normal,
        "bland_altman_low_normal": bland_altman_low_normal,
        "bland_altman_high_normal": bland_altman_high_normal,
        "bland_altman_low_empirical": bland_altman_low_empirical,
        "bland_altman_high_empirical": bland_altman_high_empirical,
        "icc_a1": icc["icc_a1"],
        "icc_a2": icc["icc_a2"],
    }


def summarize_coverage(
    conn: sqlite3.Connection, release_id: int, rows: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    codex_count = count_single_judge_rows(conn, release_id, args.codex_model, args.codex_version_like)
    claude_count = count_single_judge_rows(conn, release_id, args.claude_model, args.claude_version_like)
    paired_count = len(rows)

    by_domain: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"D{int(row['domain_id'])}"
        by_domain.setdefault(
            key,
            {
                "domain_id": int(row["domain_id"]),
                "domain_name": row["domain_name"],
                "domain_display_name": row["domain_display_name"],
                "paired_rows": 0,
                "questions": set(),
                "models": set(),
            },
        )
        by_domain[key]["paired_rows"] += 1
        by_domain[key]["questions"].add(int(row["question_id"]))
        by_domain[key]["models"].add(str(row["model"]))

    clean_by_domain = {}
    for key, value in sorted(by_domain.items(), key=lambda item: item[1]["domain_id"]):
        clean_by_domain[key] = {
            "domain_id": value["domain_id"],
            "domain_name": value["domain_name"],
            "domain_display_name": value["domain_display_name"],
            "paired_rows": value["paired_rows"],
            "paired_questions": len(value["questions"]),
            "paired_models": len(value["models"]),
        }

    return {
        "codex_rows_matching_filter": codex_count,
        "claude_rows_matching_filter": claude_count,
        "paired_rows": paired_count,
        "is_partial_judge_overlap": paired_count < codex_count or paired_count < claude_count,
        "paired_fraction_of_codex": paired_count / codex_count if codex_count else None,
        "paired_fraction_of_claude": paired_count / claude_count if claude_count else None,
        "by_domain": clean_by_domain,
    }


def run_bootstrap(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    eligible_models = {
        model
        for model, count in Counter(str(row["model"]) for row in rows).items()
        if count >= args.min_paired_rows
    }
    filtered_rows = [row for row in rows if str(row["model"]) in eligible_models]
    grouped = group_questions_by_domain(filtered_rows)
    rng = random.Random(args.seed)

    observed_model_means = model_means(filtered_rows)
    observed_domain_model_means = domain_model_means(filtered_rows)
    observed_ranks = rank_models(observed_model_means)
    observed_reliability = reliability_summary(filtered_rows)
    observed_composites = weighted_composite_means(observed_domain_model_means)
    observed_composite_ranks = rank_models(observed_composites)

    boot_model_means: dict[str, list[float]] = defaultdict(list)
    boot_domain_model_means: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    boot_ranks: dict[str, list[int]] = defaultdict(list)
    boot_reliability: dict[str, list[float]] = defaultdict(list)
    boot_composites: dict[str, list[float]] = defaultdict(list)
    boot_composite_ranks: dict[str, list[int]] = defaultdict(list)
    boot_paired_differences: dict[str, list[float]] = defaultdict(list)
    observed_paired_differences = paired_composite_differences(observed_composites)

    for _ in range(args.bootstrap_reps):
        sample = resample_questions_within_domains(grouped, rng)
        sample_means = model_means(sample)
        sample_domain_means = domain_model_means(sample)
        sample_ranks = rank_models(sample_means)
        sample_reliability = reliability_summary(sample)
        sample_composites = weighted_composite_means(sample_domain_means)
        sample_composite_ranks = rank_models(sample_composites)
        sample_paired_diffs = paired_composite_differences(sample_composites)

        for model, value in sample_means.items():
            boot_model_means[model].append(value)
        for domain, values in sample_domain_means.items():
            for model, value in values.items():
                boot_domain_model_means[domain][model].append(value)
        for model, rank in sample_ranks.items():
            boot_ranks[model].append(rank)
        for model, value in sample_composites.items():
            boot_composites[model].append(value)
        for model, rank in sample_composite_ranks.items():
            boot_composite_ranks[model].append(rank)
        for pair_key, diff_value in sample_paired_diffs.items():
            boot_paired_differences[pair_key].append(diff_value)
        for key in (
            "mean_signed_difference_claude_minus_codex",
            "mean_absolute_difference",
            "pearson_correlation",
            "spearman_correlation",
            "exact_same_score_rate",
            "within_0_25_rate",
            "within_0_50_rate",
            "within_1_00_rate",
            "bland_altman_low",
            "bland_altman_high",
            "bland_altman_low_normal",
            "bland_altman_high_normal",
            "bland_altman_low_empirical",
            "bland_altman_high_empirical",
            "icc_a1",
            "icc_a2",
        ):
            value = sample_reliability.get(key)
            if value is not None:
                boot_reliability[key].append(float(value))

    leaderboard = []
    row_counts = Counter(str(row["model"]) for row in filtered_rows)
    for model, observed_mean in sorted(observed_model_means.items(), key=lambda item: (-item[1], item[0])):
        means = boot_model_means.get(model, [])
        ranks = boot_ranks.get(model, [])
        composites = boot_composites.get(model, [])
        composite_ranks = boot_composite_ranks.get(model, [])
        leaderboard.append(
            {
                "model": model,
                "paired_rows": row_counts[model],
                "observed_mean": rounded(observed_mean, 4),
                "ci_95_low": rounded(percentile(means, 2.5), 4),
                "ci_95_high": rounded(percentile(means, 97.5), 4),
                "observed_rank": observed_ranks.get(model),
                "median_bootstrap_rank": rounded(percentile([float(rank) for rank in ranks], 50), 2),
                "rank_95_low": rounded(percentile([float(rank) for rank in ranks], 2.5), 2),
                "rank_95_high": rounded(percentile([float(rank) for rank in ranks], 97.5), 2),
                "top_1_rate": rounded(sum(1 for rank in ranks if rank <= 1) / len(ranks), 4) if ranks else None,
                "top_3_rate": rounded(sum(1 for rank in ranks if rank <= 3) / len(ranks), 4) if ranks else None,
                "composite_observed": rounded(observed_composites.get(model), 4),
                "composite_ci_95_low": rounded(percentile(composites, 2.5), 4),
                "composite_ci_95_high": rounded(percentile(composites, 97.5), 4),
                "composite_observed_rank": observed_composite_ranks.get(model),
                "composite_median_bootstrap_rank": rounded(percentile([float(rank) for rank in composite_ranks], 50), 2),
                "composite_rank_95_low": rounded(percentile([float(rank) for rank in composite_ranks], 2.5), 2),
                "composite_rank_95_high": rounded(percentile([float(rank) for rank in composite_ranks], 97.5), 2),
                "composite_top_1_rate": rounded(sum(1 for rank in composite_ranks if rank <= 1) / len(composite_ranks), 4) if composite_ranks else None,
                "composite_top_3_rate": rounded(sum(1 for rank in composite_ranks if rank <= 3) / len(composite_ranks), 4) if composite_ranks else None,
            }
        )

    domains = {}
    for domain, model_values in observed_domain_model_means.items():
        domains[domain] = []
        for model, observed_mean in sorted(model_values.items(), key=lambda item: (-item[1], item[0])):
            samples = boot_domain_model_means.get(domain, {}).get(model, [])
            domains[domain].append(
                {
                    "model": model,
                    "observed_mean": rounded(observed_mean, 4),
                    "ci_95_low": rounded(percentile(samples, 2.5), 4),
                    "ci_95_high": rounded(percentile(samples, 97.5), 4),
                }
            )

    reliability = {
        key: rounded(value, 4) if isinstance(value, float) else value
        for key, value in observed_reliability.items()
    }
    reliability["bootstrap_ci"] = {
        key: {
            "ci_95_low": rounded(percentile(values, 2.5), 4),
            "ci_95_high": rounded(percentile(values, 97.5), 4),
        }
        for key, values in boot_reliability.items()
    }

    # Composite-rank-ordered leaderboard for selection-time use.
    # Same models as `leaderboard`, ordered by composite_observed (descending),
    # alphabetical tiebreak.
    composite_leaderboard = sorted(
        [row for row in leaderboard if row["composite_observed"] is not None],
        key=lambda row: (-row["composite_observed"], row["model"]),
    )

    # Paired-difference CIs for each model pair in PAIRED_DIFFERENCE_PAIRS.
    # Replaces the necessary-but-not-sufficient marginal-CI-overlap inference
    # (Schenker and Gentleman 2001) with a properly paired test that uses the
    # per-resample covariance between the two models' composites.
    paired_composite_diffs = []
    for pair_key, diffs in boot_paired_differences.items():
        model_a, model_b = pair_key.split("__minus__", 1)
        observed_diff = observed_paired_differences.get(pair_key)
        ci_low = percentile(diffs, 2.5)
        ci_high = percentile(diffs, 97.5)
        ci_includes_zero = (
            ci_low is not None and ci_high is not None and ci_low <= 0.0 <= ci_high
        )
        paired_composite_diffs.append({
            "model_a": model_a,
            "model_b": model_b,
            "pair_key": pair_key,
            "observed_difference": rounded(observed_diff),
            "bootstrap_mean_difference": rounded(mean_or_none(diffs)),
            "bootstrap_ci_low": rounded(ci_low),
            "bootstrap_ci_high": rounded(ci_high),
            "ci_includes_zero": ci_includes_zero,
            "n_bootstrap_resamples": len(diffs),
            "ci_type": "percentile_95",
        })

    return {
        "eligible_model_count": len(eligible_models),
        "eligible_paired_rows": len(filtered_rows),
        "min_paired_rows": args.min_paired_rows,
        "bootstrap_unit": "questions resampled with replacement within each domain; all model responses and paired judge scores for sampled questions retained",
        "leaderboard": leaderboard,
        "weighted_composite_leaderboard": composite_leaderboard,
        "weighted_composite_weights": ROLE_AUDITION_WEIGHTS,
        "weighted_composite_caveat": (
            "Composite intervals are conditioned on the observed paired-analysis "
            "question counts by domain: D1=66, D2=16, D3=24, D4=24, D5=25, "
            "D6=24, D7=50. Questions are resampled with replacement within "
            "domain, so domains with fewer paired questions have more discrete "
            "bootstrap support than domains with larger question counts. The 924 "
            "D1 structured morphology adjunct responses in the public release "
            "are excluded from this paired composite because they were scored by "
            "the specialized morphology scorer rather than by the final paired "
            "Codex-Claude judging apparatus."
        ),
        "paired_composite_differences": paired_composite_diffs,
        "paired_composite_differences_caveat": (
            "Paired-difference CI on Δ = composite(a) - composite(b), computed "
            "from the same B bootstrap resamples that drive the marginal "
            "composite CIs. The percentile 95% CI uses the per-resample "
            "covariance between the two models' composites and is the canonical "
            "correct test of statistical distinguishability — overlapping "
            "marginal CIs are necessary-but-not-sufficient for non-distinguishability "
            "(Schenker and Gentleman 2001, The American Statistician 55:182-186). "
            "If `ci_includes_zero` is true, the two models are not statistically "
            "distinguishable on the role-weighted composite at 95% on this "
            "snapshot; if false, the difference is statistically distinguishable "
            "under this paired bootstrap procedure. Snapshot reflects the "
            "`eligible_paired_rows` paired-overlap subset and inherits the "
            "domain sample-size and paired-analysis cohort caveats above."
        ),
        "per_domain": domains,
        "interjudge_reliability": reliability,
    }


def write_outputs(output_dir: Path, label: str, payload: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{label}_cluster_bootstrap_results.json"
    leaderboard_csv_path = output_dir / f"{label}_cluster_bootstrap_leaderboard.csv"
    reliability_json_path = output_dir / f"{label}_interjudge_reliability.json"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reliability_json_path.write_text(
        json.dumps(payload["analysis"]["interjudge_reliability"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with leaderboard_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "model",
            "paired_rows",
            "observed_mean",
            "ci_95_low",
            "ci_95_high",
            "observed_rank",
            "median_bootstrap_rank",
            "rank_95_low",
            "rank_95_high",
            "top_1_rate",
            "top_3_rate",
            "composite_observed",
            "composite_ci_95_low",
            "composite_ci_95_high",
            "composite_observed_rank",
            "composite_median_bootstrap_rank",
            "composite_rank_95_low",
            "composite_rank_95_high",
            "composite_top_1_rate",
            "composite_top_3_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["analysis"]["leaderboard"]:
            writer.writerow(row)

    return {
        "json": str(json_path),
        "leaderboard_csv": str(leaderboard_csv_path),
        "reliability_json": str(reliability_json_path),
    }


def script_git_commit() -> dict[str, Any]:
    """Return git commit metadata for this script's repo, or a loud error marker."""
    repo_root = Path(__file__).resolve().parent.parent
    out: dict[str, Any] = {"script_path": "scripts/cluster_bootstrap_mars_results.py"}
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.PIPE
        ).decode().strip()
        out["sha"] = sha
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, stderr=subprocess.PIPE
        ).decode().strip())
        out["dirty"] = dirty
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        out["sha"] = None
        out["error"] = repr(exc)
    return out


def paired_rows_sha256(rows: list[dict[str, Any]]) -> str:
    """Hash a canonical (response_id, codex_score, claude_score) serialization.

    This lets a future rerun detect that the underlying paired-row set has changed
    even when release_label is unchanged (e.g. after rejudging or row additions).
    """
    canonical = sorted(
        ((int(r["response_id"]), float(r["codex_score"]), float(r["claude_score"])) for r in rows),
        key=lambda x: x[0],
    )
    payload = json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    conn = connect(args.db)
    release = release_by_label(conn, args.release_label)
    rows = load_paired_rows(conn, int(release["id"]), args)
    if not rows:
        raise SystemExit("No paired Codex/Claude rows found for the selected filters.")

    coverage = summarize_coverage(conn, int(release["id"]), rows, args)
    analysis = run_bootstrap(rows, args)

    payload = {
        "metadata": {
            "analysis_label": args.label,
            "db_path": str(args.db),
            "release_id": int(release["id"]),
            "release_label": release["release_label"],
            "release_status": release["status"],
            "release_created_at": release["created_at"],
            "codex_model": args.codex_model,
            "claude_model": args.claude_model,
            "codex_version_like": args.codex_version_like,
            "claude_version_like": args.claude_version_like,
            "bootstrap_reps": args.bootstrap_reps,
            "seed": args.seed,
            "bootstrap_interval_type": "percentile",
            "rng_implementation": "random.Random (Mersenne Twister)",
            "script_git": script_git_commit(),
            "paired_rows_sha256": paired_rows_sha256(rows),
            "partial_results_warning": (
                "This is a partial paired-judge snapshot; rerun after Claude judging completes."
                if coverage["is_partial_judge_overlap"]
                else None
            ),
        },
        "coverage": coverage,
        "analysis": analysis,
    }
    paths = write_outputs(args.output_dir, args.label, payload)

    print("Cluster-aware MARS-Bench bootstrap complete.")
    print(f"Release: {release['release_label']} (id={release['id']})")
    print(f"Paired rows: {coverage['paired_rows']} / Codex rows: {coverage['codex_rows_matching_filter']} / Claude rows: {coverage['claude_rows_matching_filter']}")
    if coverage["is_partial_judge_overlap"]:
        print("WARNING: partial paired-judge overlap. Treat leaderboard intervals as interim.")
    print(f"Bootstrap reps: {args.bootstrap_reps}; seed: {args.seed}")
    print("Top models by paired averaged score:")
    for row in analysis["leaderboard"][:10]:
        print(
            f"  {row['observed_rank']:>2}. {row['model']:<42} "
            f"{row['observed_mean']:.3f} [{row['ci_95_low']:.3f}, {row['ci_95_high']:.3f}] "
            f"top3={row['top_3_rate']:.1%}"
        )
    print("Outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
