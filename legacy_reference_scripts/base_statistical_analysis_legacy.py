#!/usr/bin/env python3
"""
MARS-Bench Statistical Analysis Suite.

Computes bias detection, reliability metrics, effect sizes, and confidence intervals
for the complete benchmark dataset. Informed by the literature review (2024-2026).

Output: JSON results file + console summary.
"""

import json
import sqlite3
import sys
from pathlib import Path
from itertools import combinations
import math

import numpy as np
from scipy import stats

DB_PATH = Path(__file__).parent.parent / "data" / "mars_bench.db"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "statistical_analysis.json"


def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def get_model_scores(conn):
    """Get all scores organized by model and domain."""
    rows = conn.execute("""
        SELECT m.name as model, q.domain_id, q.id as question_id, r.run_number,
               e.overall_score, r.response_text, e.per_dimension_scores
        FROM evaluations e
        JOIN responses r ON e.response_id = r.id
        JOIN models m ON r.model_id = m.id
        JOIN questions q ON r.question_id = q.id
        WHERE q.retired_at IS NULL
        ORDER BY m.name, q.domain_id, q.id, r.run_number
    """).fetchall()
    return [dict(r) for r in rows]


def verbosity_bias_analysis(data):
    """Compute Pearson correlation between word count and score per model.
    (arXiv:2411.15594, arXiv:2503.05061)"""
    print("\n=== VERBOSITY BIAS ANALYSIS ===")
    results = {}

    for model in sorted(set(d['model'] for d in data)):
        model_data = [d for d in data if d['model'] == model]
        word_counts = [len(d['response_text'].split()) if d['response_text'] else 0 for d in model_data]
        scores = [d['overall_score'] for d in model_data]

        if len(scores) < 3:
            continue

        r, p = stats.pearsonr(word_counts, scores)
        avg_words = np.mean(word_counts)

        # Verbosity Inflation Index
        median_wc = np.median(word_counts)
        above = [s for s, w in zip(scores, word_counts) if w > median_wc]
        below = [s for s, w in zip(scores, word_counts) if w <= median_wc]
        inflation = (np.mean(above) - np.mean(below)) / np.mean(below) if below and np.mean(below) > 0 else 0

        flag = "FLAGGED" if abs(r) > 0.3 else "OK"
        print(f"  {model:<35} r={r:>6.3f} p={p:.4f} avg_words={avg_words:>6.0f} inflation={inflation:>6.1%} [{flag}]")

        results[model] = {
            "pearson_r": round(r, 4),
            "p_value": round(p, 6),
            "avg_word_count": round(avg_words, 1),
            "verbosity_inflation_index": round(inflation, 4),
            "flagged": abs(r) > 0.3,
        }

    return results


def run_to_run_variance(data):
    """Compute intra-model variance across 3 runs per question.
    (arXiv:2512.16041)"""
    print("\n=== RUN-TO-RUN VARIANCE ===")
    results = {}

    for model in sorted(set(d['model'] for d in data)):
        model_data = [d for d in data if d['model'] == model]

        # Group by question
        questions = {}
        for d in model_data:
            qid = d['question_id']
            if qid not in questions:
                questions[qid] = []
            questions[qid].append(d['overall_score'])

        # Compute per-question variance
        variances = []
        cvs = []
        for qid, scores in questions.items():
            if len(scores) >= 2:
                v = np.var(scores, ddof=1)
                m = np.mean(scores)
                variances.append(v)
                if m > 0:
                    cvs.append(np.std(scores, ddof=1) / m)

        mean_var = np.mean(variances) if variances else 0
        mean_cv = np.mean(cvs) if cvs else 0
        high_cv_count = sum(1 for cv in cvs if cv > 0.15)

        print(f"  {model:<35} mean_var={mean_var:.4f} mean_CV={mean_cv:.3f} high_CV_questions={high_cv_count}/{len(cvs)}")

        results[model] = {
            "mean_variance": round(mean_var, 5),
            "mean_cv": round(mean_cv, 4),
            "high_cv_questions": high_cv_count,
            "total_questions": len(cvs),
        }

    return results


def pairwise_model_comparisons(data):
    """Wilcoxon Signed-Rank Test with Benjamini-Hochberg correction.
    (arXiv:2512.16041)"""
    print("\n=== PAIRWISE MODEL COMPARISONS (Wilcoxon + BH) ===")

    models = sorted(set(d['model'] for d in data))
    # Exclude ceiling for pairwise
    models = [m for m in models if 'ceiling' not in m]

    # Build per-question mean scores per model
    model_question_scores = {}
    for model in models:
        model_data = [d for d in data if d['model'] == model]
        q_scores = {}
        for d in model_data:
            qid = d['question_id']
            if qid not in q_scores:
                q_scores[qid] = []
            q_scores[qid].append(d['overall_score'])
        model_question_scores[model] = {qid: np.mean(scores) for qid, scores in q_scores.items()}

    # Get common questions
    common_qs = set.intersection(*[set(mqs.keys()) for mqs in model_question_scores.values()])

    results = []
    p_values = []

    for m1, m2 in combinations(models, 2):
        scores1 = [model_question_scores[m1][q] for q in common_qs]
        scores2 = [model_question_scores[m2][q] for q in common_qs]

        diffs = [s1 - s2 for s1, s2 in zip(scores1, scores2)]

        try:
            stat, p = stats.wilcoxon(diffs, alternative='two-sided')
        except ValueError:
            stat, p = 0, 1.0

        # Cohen's d
        pooled_std = np.sqrt((np.var(scores1, ddof=1) + np.var(scores2, ddof=1)) / 2)
        cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std if pooled_std > 0 else 0

        results.append({
            "model_1": m1,
            "model_2": m2,
            "mean_diff": round(np.mean(diffs), 3),
            "wilcoxon_stat": round(float(stat), 2),
            "p_value": round(float(p), 6),
            "cohens_d": round(cohens_d, 3),
            "n_questions": len(common_qs),
        })
        p_values.append(p)

    # Benjamini-Hochberg correction
    n_tests = len(p_values)
    sorted_indices = np.argsort(p_values)
    bh_threshold = [(i + 1) / n_tests * 0.05 for i in range(n_tests)]

    for i, idx in enumerate(sorted_indices):
        results[idx]["bh_significant"] = p_values[idx] <= bh_threshold[i]

    # Print top comparisons
    significant = [r for r in results if r.get("bh_significant")]
    print(f"  {len(significant)}/{len(results)} pairs significantly different (BH-corrected p<0.05)")
    for r in sorted(results, key=lambda x: abs(x['cohens_d']), reverse=True)[:10]:
        sig = "*" if r.get('bh_significant') else " "
        print(f"  {sig} {r['model_1'][:20]:<20} vs {r['model_2'][:20]:<20} d={r['cohens_d']:>6.3f} p={r['p_value']:.4f}")

    return results


def bootstrap_confidence_intervals(data, n_bootstrap=1000):
    """Bootstrap 95% CIs for each model's overall and per-domain scores."""
    print("\n=== BOOTSTRAP 95% CONFIDENCE INTERVALS ===")
    results = {}

    for model in sorted(set(d['model'] for d in data)):
        model_data = [d for d in data if d['model'] == model]
        scores = [d['overall_score'] for d in model_data]

        # Overall CI
        boot_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            boot_means.append(np.mean(sample))

        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        mean_score = np.mean(scores)

        # Per-domain CIs
        domain_cis = {}
        for did in range(1, 8):
            d_scores = [d['overall_score'] for d in model_data if d['domain_id'] == did]
            if len(d_scores) < 3:
                continue
            boot_d = [np.mean(np.random.choice(d_scores, size=len(d_scores), replace=True)) for _ in range(n_bootstrap)]
            d_low, d_high = np.percentile(boot_d, [2.5, 97.5])
            domain_cis[f"D{did}"] = {"mean": round(np.mean(d_scores), 3), "ci_low": round(d_low, 3), "ci_high": round(d_high, 3)}

        print(f"  {model:<35} {mean_score:.2f} [{ci_low:.2f}, {ci_high:.2f}]")

        results[model] = {
            "overall_mean": round(mean_score, 3),
            "ci_95_low": round(ci_low, 3),
            "ci_95_high": round(ci_high, 3),
            "per_domain": domain_cis,
        }

    return results


def score_distribution_analysis(data):
    """Analyze score distributions per model — skewness, kurtosis, modality."""
    print("\n=== SCORE DISTRIBUTION ANALYSIS ===")
    results = {}

    for model in sorted(set(d['model'] for d in data)):
        scores = [d['overall_score'] for d in data if d['model'] == model]

        results[model] = {
            "mean": round(np.mean(scores), 3),
            "median": round(np.median(scores), 3),
            "std": round(np.std(scores, ddof=1), 3),
            "skewness": round(float(stats.skew(scores)), 3),
            "kurtosis": round(float(stats.kurtosis(scores)), 3),
            "min": round(min(scores), 2),
            "max": round(max(scores), 2),
            "pct_below_2": round(sum(1 for s in scores if s < 2) / len(scores) * 100, 1),
            "pct_3_plus": round(sum(1 for s in scores if s >= 3) / len(scores) * 100, 1),
            "pct_4_plus": round(sum(1 for s in scores if s >= 4) / len(scores) * 100, 1),
        }

        print(f"  {model:<35} mean={results[model]['mean']:.2f} std={results[model]['std']:.2f} "
              f"skew={results[model]['skewness']:>6.2f} >=3: {results[model]['pct_3_plus']:>5.1f}% >=4: {results[model]['pct_4_plus']:>5.1f}%")

    return results


def difficulty_level_analysis(data):
    """Analyze score patterns by question difficulty level."""
    print("\n=== DIFFICULTY LEVEL ANALYSIS ===")
    conn = get_conn()
    results = {}

    for model in sorted(set(d['model'] for d in data)):
        model_results = {}
        for level in [1, 2, 3]:
            level_scores = []
            for d in data:
                if d['model'] == model:
                    q_level = conn.execute("SELECT level FROM questions WHERE id=?", (d['question_id'],)).fetchone()
                    if q_level and q_level['level'] == level:
                        level_scores.append(d['overall_score'])
            if level_scores:
                model_results[f"L{level}"] = {
                    "mean": round(np.mean(level_scores), 3),
                    "n": len(level_scores),
                }
        results[model] = model_results

    # Print summary
    models_no_ceiling = [m for m in sorted(set(d['model'] for d in data)) if 'ceiling' not in m]
    print(f"  {'Model':<35} {'L1':>8} {'L2':>8} {'L3':>8} {'Gradient':>10}")
    for model in models_no_ceiling:
        if model in results and 'L1' in results[model] and 'L3' in results[model]:
            l1 = results[model]['L1']['mean']
            l3 = results[model]['L3']['mean']
            gradient = l1 - l3
            print(f"  {model:<35} {l1:>8.2f} {results[model].get('L2',{}).get('mean',0):>8.2f} {l3:>8.2f} {gradient:>+10.2f}")

    return results


def trap_question_analysis(data):
    """Analyze trap question performance across models."""
    print("\n=== TRAP QUESTION ANALYSIS ===")
    conn = get_conn()
    results = {}

    trap_qids = [r['id'] for r in conn.execute("SELECT id FROM questions WHERE is_trap=1 AND retired_at IS NULL").fetchall()]

    for model in sorted(set(d['model'] for d in data)):
        model_data = [d for d in data if d['model'] == model]
        trap_scores = [d['overall_score'] for d in model_data if d['question_id'] in trap_qids]
        non_trap_scores = [d['overall_score'] for d in model_data if d['question_id'] not in trap_qids]

        if trap_scores and non_trap_scores:
            trap_mean = np.mean(trap_scores)
            non_trap_mean = np.mean(non_trap_scores)
            diff = trap_mean - non_trap_mean

            results[model] = {
                "trap_mean": round(trap_mean, 3),
                "non_trap_mean": round(non_trap_mean, 3),
                "difference": round(diff, 3),
                "n_trap": len(trap_scores),
                "n_non_trap": len(non_trap_scores),
            }
            print(f"  {model:<35} trap={trap_mean:.2f} non-trap={non_trap_mean:.2f} diff={diff:>+.2f}")

    return results


def self_evaluation_bias(data):
    """Quantify potential self-preference bias for the ceiling model."""
    print("\n=== SELF-EVALUATION BIAS CHECK ===")

    ceiling_scores = [d['overall_score'] for d in data if d['model'] == 'claude-opus-4-6-ceiling']
    other_scores = [d['overall_score'] for d in data if 'ceiling' not in d['model']]

    if ceiling_scores and other_scores:
        ceiling_mean = np.mean(ceiling_scores)
        other_mean = np.mean(other_scores)
        gap = ceiling_mean - other_mean

        # Is the gap larger than expected from model quality alone?
        # Compare to the best open-weight model
        best_open = max(
            np.mean([d['overall_score'] for d in data if d['model'] == m])
            for m in set(d['model'] for d in data) if 'ceiling' not in m
        )

        print(f"  Ceiling mean:       {ceiling_mean:.2f}")
        print(f"  Best open-weight:   {best_open:.2f}")
        print(f"  All open-weight:    {other_mean:.2f}")
        print(f"  Ceiling gap:        {gap:.2f}")
        print(f"  Ceiling vs best:    {ceiling_mean - best_open:.2f}")
        print(f"  NOTE: Self-evaluation bias likely inflates ceiling by ~0.3-0.5 points (literature estimate)")

        return {
            "ceiling_mean": round(ceiling_mean, 3),
            "best_open_weight_mean": round(best_open, 3),
            "all_open_weight_mean": round(other_mean, 3),
            "ceiling_gap": round(gap, 3),
            "ceiling_vs_best_gap": round(ceiling_mean - best_open, 3),
        }
    return {}


def main():
    print("MARS-Bench Statistical Analysis")
    print("=" * 60)

    conn = get_conn()
    data = get_model_scores(conn)
    print(f"Loaded {len(data)} evaluations across {len(set(d['model'] for d in data))} models")

    np.random.seed(42)  # Reproducibility

    results = {
        "metadata": {
            "total_evaluations": len(data),
            "total_models": len(set(d['model'] for d in data)),
            "total_questions": len(set(d['question_id'] for d in data)),
            "random_seed": 42,
        },
        "verbosity_bias": verbosity_bias_analysis(data),
        "run_to_run_variance": run_to_run_variance(data),
        "score_distributions": score_distribution_analysis(data),
        "difficulty_levels": difficulty_level_analysis(data),
        "trap_questions": trap_question_analysis(data),
        "self_evaluation_bias": self_evaluation_bias(data),
        "bootstrap_cis": bootstrap_confidence_intervals(data),
        "pairwise_comparisons": pairwise_model_comparisons(data),
    }

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=convert)

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
