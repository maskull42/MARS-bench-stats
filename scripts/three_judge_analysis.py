#!/usr/bin/env python3
"""Three-judge inter-rater reliability and re-analysis for MARS-Bench.

Adds DeepSeek V4 Flash (via the OpenCode agentic harness) as a third judge lane
alongside the production Codex and Claude lanes. The production DeepSeek Flash
lane was served through DeepInfra for the original cohort and through
OpenRouter for the later Gemma 4 12B cohort. Excludes both DeepSeek V4 Pro
variants by filtering on the V4 Flash judge_prompt_version pattern. Computes:

  - ICC(A,1) and ICC(A,k=3) two-way mixed-effects, absolute-agreement, by McGraw
    and Wong (1996) Forming Inferences About Some Intraclass Correlation
    Coefficients, Psychological Methods 1(1), 30-46. Pairwise ICCs for the three
    pairs (Codex-Claude, Codex-DeepSeek, Claude-DeepSeek) are reported as well.
  - Pairwise reliability (mean signed and absolute differences, Pearson and
    Spearman correlations, within-tolerance rates, parametric and empirical
    Bland-Altman limits of agreement).
  - Per-model role-weighted composite under the three-judge mean, alongside the
    two-judge baseline; weights are 0.08, 0.17, 0.175, 0.07, 0.14, 0.145, 0.22
    for D1-D7 respectively, matching cluster_bootstrap_mars_results.py.
  - Cluster (by question_id) bootstrap of the paired Llama-Gemma role-weighted
    composite difference under the three-judge mean.
  - Per-candidate-model offset of the DeepSeek-as-judge mean from the
    (Codex+Claude)/2 baseline, useful for diagnosing self-preference bias in
    judge-candidate pairs that share a model family.

Usage:

  python scripts/three_judge_analysis.py \\
      --db data/mars_bench_stats_public.sqlite \\
      --release-label mars_bench_v2_0_d1_d2_d3_d4_d5_d6_d7_rebuild_candidate \\
      --output-dir results/three_judge \\
      --label final_2026_05_10 \\
      --bootstrap-reps 1000 \\
      --seed 20260503

The default values match the v2026.05.10 release configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

WEIGHTS = {1: 0.08, 2: 0.17, 3: 0.175, 4: 0.07, 5: 0.14, 6: 0.145, 7: 0.22}

QUERY = """
SELECT m.name AS model,
       q.domain_id AS domain_id,
       q.id AS question_id,
       r.run_number AS run_number,
       r.id AS response_id,
       e_codex.overall_score AS codex,
       e_claude.overall_score AS claude,
       e_ds.overall_score AS deepseek
FROM responses r
JOIN models m ON m.id = r.model_id
JOIN questions q ON q.id = r.question_id
JOIN evaluations e_codex
       ON e_codex.response_id = r.id
      AND e_codex.judge_model = 'gpt-5.5-medium-codex-cli'
      AND e_codex.judge_prompt_version LIKE '%codex_gpt_5_5_medium_primary'
JOIN evaluations e_claude
       ON e_claude.response_id = r.id
      AND e_claude.judge_model = 'claude-opus-4-7-medium-claude-cli'
      AND e_claude.judge_prompt_version LIKE '%claude_opus_4_7_medium_comparison'
JOIN evaluations e_ds
       ON e_ds.response_id = r.id
      AND (
          (
              e_ds.judge_model = 'deepseek-v4-flash-deepinfra-opencode-cli'
              AND e_ds.judge_prompt_version LIKE '%opencode_deepseek_v4_flash_comparison'
          )
          OR
          (
              e_ds.judge_model = 'deepseek-v4-flash-openrouter-opencode-cli'
              AND e_ds.judge_prompt_version LIKE '%opencode_deepseek_v4_flash_openrouter_comparison'
          )
      )
JOIN release_questions rq ON rq.question_id = q.id AND rq.is_included = 1
JOIN benchmark_releases br ON br.id = rq.release_id
WHERE br.release_label = ?
  AND q.is_supplementary = 0
"""


def icc_two_way_mixed_absolute(scores: np.ndarray) -> Dict[str, float]:
    n, k = scores.shape
    grand = scores.mean()
    row_means = scores.mean(axis=1)
    col_means = scores.mean(axis=0)
    SS_R = k * np.sum((row_means - grand) ** 2)
    SS_C = n * np.sum((col_means - grand) ** 2)
    SS_T = np.sum((scores - grand) ** 2)
    SS_E = SS_T - SS_R - SS_C
    MS_R = SS_R / (n - 1)
    MS_C = SS_C / (k - 1)
    MS_E = SS_E / ((n - 1) * (k - 1))
    icc_a1 = (MS_R - MS_E) / (MS_R + (k - 1) * MS_E + k * (MS_C - MS_E) / n)
    icc_ak = (MS_R - MS_E) / (MS_R + (MS_C - MS_E) / n)
    return {
        "icc_a1": float(icc_a1),
        "icc_ak": float(icc_ak),
        "n_subjects": int(n),
        "k_raters": int(k),
        "ms_r": float(MS_R),
        "ms_c": float(MS_C),
        "ms_e": float(MS_E),
    }


def pairwise_stats(a: np.ndarray, b: np.ndarray, label: str) -> Dict[str, float]:
    diff = a - b
    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1))
    abs_diff = np.abs(diff)
    return {
        "label": label,
        "n": int(a.size),
        "mean_signed_diff": mean_diff,
        "mean_absolute_diff": float(abs_diff.mean()),
        "pearson": float(stats.pearsonr(a, b).statistic),
        "spearman": float(stats.spearmanr(a, b).statistic),
        "within_0_25_rate": float((abs_diff <= 0.25).mean()),
        "within_0_50_rate": float((abs_diff <= 0.50).mean()),
        "within_1_00_rate": float((abs_diff <= 1.00).mean()),
        "loa_low_normal": mean_diff - 1.96 * sd_diff,
        "loa_high_normal": mean_diff + 1.96 * sd_diff,
        "loa_low_empirical": float(np.percentile(diff, 2.5)),
        "loa_high_empirical": float(np.percentile(diff, 97.5)),
        "exact_match_rate": float((diff == 0).mean()),
    }


def composite_per_model(df: pd.DataFrame, score_col: str) -> pd.Series:
    per_md = df.groupby(["model", "domain_id"])[score_col].mean().unstack("domain_id")
    per_md = per_md.reindex(columns=[1, 2, 3, 4, 5, 6, 7])
    return sum(WEIGHTS[d] * per_md[d] for d in [1, 2, 3, 4, 5, 6, 7])


def cluster_bootstrap_paired_diff(
    df: pd.DataFrame,
    score_col: str,
    model_a: str,
    model_b: str,
    n_reps: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    questions = df["question_id"].unique()
    obs = float(
        composite_per_model(df, score_col).loc[model_a]
        - composite_per_model(df, score_col).loc[model_b]
    )
    diffs = np.empty(n_reps)
    q_index = {q: np.where(df["question_id"].values == q)[0] for q in questions}
    for i in range(n_reps):
        sample_qs = rng.choice(questions, size=questions.size, replace=True)
        idx = np.concatenate([q_index[q] for q in sample_qs])
        cmp = composite_per_model(df.iloc[idx], score_col)
        diffs[i] = (
            cmp.loc[model_a] - cmp.loc[model_b]
            if model_a in cmp.index and model_b in cmp.index
            else np.nan
        )
    diffs = diffs[~np.isnan(diffs)]
    return {
        "model_a": model_a,
        "model_b": model_b,
        "observed_diff": obs,
        "bootstrap_mean_diff": float(diffs.mean()),
        "ci_95_low": float(np.percentile(diffs, 2.5)),
        "ci_95_high": float(np.percentile(diffs, 97.5)),
        "ci_includes_zero": bool(
            np.percentile(diffs, 2.5) <= 0 <= np.percentile(diffs, 97.5)
        ),
        "n_bootstrap_resamples": int(diffs.size),
        "seed": seed,
        "ci_type": "percentile_95",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MARS-Bench three-judge re-analysis.")
    parser.add_argument("--db", type=Path, default=Path("data/mars_bench_stats_public.sqlite"))
    parser.add_argument(
        "--release-label",
        default="mars_bench_v2_0_d1_d2_d3_d4_d5_d6_d7_rebuild_candidate",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/three_judge"))
    parser.add_argument("--label", default="final_2026_05_10")
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260503)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(str(args.db))
    df = pd.read_sql_query(QUERY, con, params=(args.release_label,))
    con.close()

    # Deterministic row order for reproducibility of the cluster bootstrap.
    # Without this, the RNG state interacts with whatever order the SQL engine
    # happened to return rows in, and different databases exporting the same
    # data can produce slightly different bootstrap CIs.
    df = (
        df.sort_values(["question_id", "model", "run_number"])
        .reset_index(drop=True)
    )

    if len(df) == 0:
        raise SystemExit(
            "Three-judge query returned zero rows. Confirm the DeepSeek V4 Flash "
            "lane is present in the source DB and the release label is correct."
        )

    if df["response_id"].duplicated().any():
        dupes = sorted(df.loc[df["response_id"].duplicated(), "response_id"].unique().tolist())
        raise SystemExit(
            "Three-judge query matched multiple DeepSeek Flash rows for at least "
            f"one response_id; first duplicates: {dupes[:10]}"
        )

    df["mean3"] = (df["codex"] + df["claude"] + df["deepseek"]) / 3.0
    df["mean2"] = (df["codex"] + df["claude"]) / 2.0
    df["cc_mean"] = df["mean2"]

    icc_3 = icc_two_way_mixed_absolute(df[["codex", "claude", "deepseek"]].values)
    icc_cc = icc_two_way_mixed_absolute(df[["codex", "claude"]].values)
    icc_cd = icc_two_way_mixed_absolute(df[["codex", "deepseek"]].values)
    icc_ld = icc_two_way_mixed_absolute(df[["claude", "deepseek"]].values)

    pw_cc = pairwise_stats(df["claude"].values, df["codex"].values, "claude_minus_codex")
    pw_dc = pairwise_stats(df["deepseek"].values, df["codex"].values, "deepseek_minus_codex")
    pw_dl = pairwise_stats(df["deepseek"].values, df["claude"].values, "deepseek_minus_claude")

    comp2 = composite_per_model(df, "mean2").sort_values(ascending=False)
    comp3 = composite_per_model(df, "mean3").sort_values(ascending=False)

    bs_lg = cluster_bootstrap_paired_diff(
        df, "mean3", "llama-3.3-70b-instruct", "gemma-4-31b-it",
        args.bootstrap_reps, args.seed,
    )

    # DeepSeek-as-judge offset diagnostic for self-preference probe
    per_model_offset = (
        df.groupby("model")
        .agg(
            n=("codex", "size"),
            deepseek_judge_mean=("deepseek", "mean"),
            cc_baseline_mean=("cc_mean", "mean"),
        )
        .assign(deepseek_minus_cc=lambda x: x["deepseek_judge_mean"] - x["cc_baseline_mean"])
        .sort_values("deepseek_minus_cc", ascending=False)
        .reset_index()
    )

    # Per-domain x per-model means (3-judge) — for heatmap regeneration
    cell3 = (
        df.groupby(["model", "domain_id"])["mean3"].mean().unstack("domain_id")
        .reindex(columns=[1, 2, 3, 4, 5, 6, 7])
    )

    # Provenance fingerprint (three-judge version)
    paired_for_hash = (
        df[["model", "domain_id", "question_id", "run_number", "codex", "claude", "deepseek"]]
        .sort_values(["question_id", "model", "run_number"])
        .to_csv(index=False)
        .encode()
    )
    paired_sha = hashlib.sha256(paired_for_hash).hexdigest()

    results = {
        "metadata": {
            "analysis_label": args.label,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "db_path": str(args.db),
            "release_label": args.release_label,
            "n_paired_rows": int(len(df)),
            "judge_lanes": [
                "gpt-5.5-medium-codex-cli (Codex)",
                "claude-opus-4-7-medium-claude-cli (Claude)",
                "DeepSeek V4 Flash via OpenCode (DeepInfra for the original cohort; OpenRouter for Gemma 4 12B)",
            ],
            "excluded_lanes": [
                "deepseek-v4-pro-deepinfra-opencode-cli",
                "deepseek-v4-pro-opencode-cli",
            ],
            "paired_rows_sha256": paired_sha,
            "bootstrap_reps": args.bootstrap_reps,
            "bootstrap_seed": args.seed,
            "bootstrap_interval_type": "percentile",
            "weights": WEIGHTS,
        },
        "icc_three_judges": icc_3,
        "icc_codex_claude": icc_cc,
        "icc_codex_deepseek": icc_cd,
        "icc_claude_deepseek": icc_ld,
        "pairwise_claude_codex": pw_cc,
        "pairwise_deepseek_codex": pw_dc,
        "pairwise_deepseek_claude": pw_dl,
        "composite_2judge_codex_claude": comp2.to_dict(),
        "composite_3judge_with_deepseek": comp3.to_dict(),
        "llama_minus_gemma_3judge_bootstrap": bs_lg,
        "deepseek_judge_offsets_per_model": per_model_offset.to_dict(orient="records"),
        "domain_model_means_3judge": {
            m: cell3.loc[m].to_dict() for m in cell3.index
        },
    }

    json_path = args.output_dir / f"{args.label}_three_judge_results.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)

    csv_path = args.output_dir / f"{args.label}_paired_rows_three_judge.csv"
    df[["model", "domain_id", "question_id", "run_number", "codex", "claude", "deepseek"]].to_csv(
        csv_path, index=False
    )

    offset_csv = args.output_dir / f"{args.label}_deepseek_judge_offsets.csv"
    per_model_offset.to_csv(offset_csv, index=False)

    print(f"Three-judge analysis complete.")
    print(f"  paired rows : {len(df):,}")
    print(f"  ICC(A,1) k=3: {icc_3['icc_a1']:.4f}")
    print(f"  ICC(A,k) k=3: {icc_3['icc_ak']:.4f}")
    print(f"  Llama-Gemma 3-judge: Δ={bs_lg['observed_diff']:+.4f}, "
          f"CI=[{bs_lg['ci_95_low']:+.4f}, {bs_lg['ci_95_high']:+.4f}], "
          f"includes 0={bs_lg['ci_includes_zero']}")
    print(f"Outputs:")
    print(f"  json: {json_path}")
    print(f"  paired_csv: {csv_path}")
    print(f"  offset_csv: {offset_csv}")


if __name__ == "__main__":
    main()
