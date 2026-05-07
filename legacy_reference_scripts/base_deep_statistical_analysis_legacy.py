#!/usr/bin/env python3
"""
MARS-Bench Deep Statistical Analyses.

Six analyses requested:
1. Residualized verbosity scores
2. Within-level verbosity correlations
3. D7-specific pairwise test (qwen3.5-27b-dense vs qwen3.5-397b-a17b-moe)
4. Contamination proxy via L1/L3 differential
5. High-CV question analysis for qwen3.5-27b-dense
6. Inflation index formula (from statistical_analysis.py)
"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from scipy import stats

DB_PATH = Path(__file__).parent.parent / "data" / "mars_bench.db"
STATS_JSON = Path(__file__).parent.parent / "data" / "statistical_analysis.json"

def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def load_eval_data(conn):
    """Load all evaluation data with model, domain, level, word count."""
    rows = conn.execute("""
        SELECT e.overall_score, e.response_word_count, e.response_id,
               m.name as model_name, q.domain_id, q.level, q.id as question_id,
               r.run_number, r.response_text
        FROM evaluations e
        JOIN responses r ON e.response_id = r.id
        JOIN models m ON r.model_id = m.id
        JOIN questions q ON r.question_id = q.id
        WHERE q.retired_at IS NULL
    """).fetchall()
    return [dict(r) for r in rows]


# ============================================================
# ANALYSIS 1: Residualized Verbosity Scores
# ============================================================
def analysis_1_residualized_scores(data):
    print("=" * 70)
    print("ANALYSIS 1: RESIDUALIZED VERBOSITY SCORES")
    print("=" * 70)
    print("Regress overall_score on response_word_count, compute residuals + grand mean.\n")

    scores = np.array([d['overall_score'] for d in data])
    # Use response_word_count from evaluations; fall back to len(split) if null
    word_counts = np.array([
        d['response_word_count'] if d['response_word_count'] is not None
        else len(d['response_text'].split()) if d['response_text'] else 0
        for d in data
    ])

    grand_mean = np.mean(scores)

    # Simple OLS: score = a + b * word_count + residual
    slope, intercept, r_value, p_value, std_err = stats.linregress(word_counts, scores)
    predicted = intercept + slope * word_counts
    residuals = scores - predicted
    residualized = residuals + grand_mean

    print(f"  Regression: score = {intercept:.4f} + {slope:.6f} * word_count")
    print(f"  R^2 = {r_value**2:.4f}, p = {p_value:.2e}")
    print(f"  Grand mean: {grand_mean:.3f}\n")

    # Compute per-model means for raw and residualized
    models = sorted(set(d['model_name'] for d in data))
    raw_means = {}
    resid_means = {}
    for model in models:
        indices = [i for i, d in enumerate(data) if d['model_name'] == model]
        raw_means[model] = np.mean([scores[i] for i in indices])
        resid_means[model] = np.mean([residualized[i] for i in indices])

    raw_ranking = sorted(raw_means.items(), key=lambda x: -x[1])
    resid_ranking = sorted(resid_means.items(), key=lambda x: -x[1])

    raw_rank_map = {name: i+1 for i, (name, _) in enumerate(raw_ranking)}
    resid_rank_map = {name: i+1 for i, (name, _) in enumerate(resid_ranking)}

    print(f"  {'Model':<40} {'Raw':>7} {'Rank':>5} {'Resid':>7} {'Rank':>5} {'Shift':>6}")
    print(f"  {'-'*40} {'-'*7} {'-'*5} {'-'*7} {'-'*5} {'-'*6}")
    for model, resid_score in resid_ranking:
        raw_score = raw_means[model]
        raw_r = raw_rank_map[model]
        res_r = resid_rank_map[model]
        shift = raw_r - res_r  # positive = moved up after residualization
        shift_str = f"+{shift}" if shift > 0 else str(shift)
        print(f"  {model:<40} {raw_score:>7.3f} {raw_r:>5} {resid_score:>7.3f} {res_r:>5} {shift_str:>6}")

    print()
    return raw_ranking, resid_ranking


# ============================================================
# ANALYSIS 2: Within-Level Verbosity Correlations
# ============================================================
def analysis_2_within_level_verbosity(data):
    print("=" * 70)
    print("ANALYSIS 2: WITHIN-LEVEL VERBOSITY CORRELATIONS (Pearson r)")
    print("=" * 70)
    print("Pearson r between word_count and overall_score, by level, per model.\n")

    models = sorted(set(d['model_name'] for d in data))
    # Exclude ceiling for this analysis
    models_no_ceil = [m for m in models if 'ceiling' not in m]

    print(f"  {'Model':<40} {'L1 r':>8} {'L1 p':>8} {'L2 r':>8} {'L2 p':>8} {'L3 r':>8} {'L3 p':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for model in models_no_ceil:
        parts = []
        for level in [1, 2, 3]:
            subset = [d for d in data if d['model_name'] == model and d['level'] == level]
            if len(subset) < 3:
                parts.append(("  N/A", "  N/A"))
                continue
            wc = [d['response_word_count'] or len(d['response_text'].split()) for d in subset]
            sc = [d['overall_score'] for d in subset]
            r, p = stats.pearsonr(wc, sc)
            parts.append((f"{r:>8.3f}", f"{p:>8.4f}"))

        print(f"  {model:<40} {parts[0][0]} {parts[0][1]} {parts[1][0]} {parts[1][1]} {parts[2][0]} {parts[2][1]}")

    print()


# ============================================================
# ANALYSIS 3: D7 Pairwise — qwen3.5-27b-dense vs qwen3.5-397b-a17b-moe
# ============================================================
def analysis_3_d7_pairwise(data):
    print("=" * 70)
    print("ANALYSIS 3: D7 PAIRWISE — qwen3.5-27b-dense vs qwen3.5-397b-a17b-moe")
    print("=" * 70)
    print("Wilcoxon Signed-Rank Test on Marcion Studies (D7) questions only.\n")

    m1_name = "qwen3.5-27b-dense"
    m2_name = "qwen3.5-397b-a17b-moe"

    d7_data = [d for d in data if d['domain_id'] == 7]

    # Build per-question mean scores
    def get_question_means(model_name, subset):
        q_scores = {}
        for d in subset:
            if d['model_name'] == model_name:
                qid = d['question_id']
                if qid not in q_scores:
                    q_scores[qid] = []
                q_scores[qid].append(d['overall_score'])
        return {qid: np.mean(s) for qid, s in q_scores.items()}

    m1_means = get_question_means(m1_name, d7_data)
    m2_means = get_question_means(m2_name, d7_data)

    common_qs = sorted(set(m1_means.keys()) & set(m2_means.keys()))
    n = len(common_qs)

    scores1 = np.array([m1_means[q] for q in common_qs])
    scores2 = np.array([m2_means[q] for q in common_qs])
    diffs = scores1 - scores2

    print(f"  Common D7 questions: {n}")
    print(f"  {m1_name} mean: {np.mean(scores1):.3f}")
    print(f"  {m2_name} mean: {np.mean(scores2):.3f}")
    print(f"  Mean difference: {np.mean(diffs):+.3f}\n")

    # Wilcoxon signed-rank test
    try:
        stat, p = stats.wilcoxon(diffs, alternative='two-sided')
    except ValueError as e:
        print(f"  Wilcoxon test failed: {e}")
        return

    # Rank-biserial correlation r = 1 - (2T / (n*(n+1)/2))
    # where T is the smaller of W+ and W- (the test statistic from scipy is T)
    # scipy.stats.wilcoxon returns the sum of ranks assigned to the differences with
    # the smaller absolute value. We need to compute rank-biserial properly.
    # r_rb = 1 - (2*T) / (n_nonzero * (n_nonzero + 1) / 2)
    nonzero_diffs = diffs[diffs != 0]
    n_nonzero = len(nonzero_diffs)
    rank_sum_total = n_nonzero * (n_nonzero + 1) / 2
    # stat from scipy is the smaller of W+ and W-
    r_rb = 1 - (2 * stat) / rank_sum_total if rank_sum_total > 0 else 0

    # More robust: compute W+ and W- directly
    abs_diffs = np.abs(nonzero_diffs)
    ranks = stats.rankdata(abs_diffs)
    w_plus = np.sum(ranks[nonzero_diffs > 0])
    w_minus = np.sum(ranks[nonzero_diffs < 0])
    r_rb_direct = (w_plus - w_minus) / (w_plus + w_minus) if (w_plus + w_minus) > 0 else 0

    print(f"  Wilcoxon T statistic: {stat:.2f}")
    print(f"  p-value: {p:.6f}")
    print(f"  W+ (favoring {m1_name}): {w_plus:.1f}")
    print(f"  W- (favoring {m2_name}): {w_minus:.1f}")
    print(f"  Rank-biserial r: {r_rb_direct:+.4f}")

    # Interpret
    if r_rb_direct > 0:
        favored = m1_name
    else:
        favored = m2_name
    magnitude = abs(r_rb_direct)
    if magnitude < 0.1:
        effect = "negligible"
    elif magnitude < 0.3:
        effect = "small"
    elif magnitude < 0.5:
        effect = "medium"
    else:
        effect = "large"

    sig = "YES" if p < 0.05 else "NO"
    print(f"\n  Significant at p<0.05? {sig}")
    print(f"  Effect size: {effect} ({favored} favored)")
    print()


# ============================================================
# ANALYSIS 4: Contamination Proxy — L1/L3 Differential
# ============================================================
def analysis_4_contamination_proxy(data):
    print("=" * 70)
    print("ANALYSIS 4: CONTAMINATION PROXY — L1/L3 DIFFERENTIAL")
    print("=" * 70)
    print("(L1 mean - L3 mean) per model. Large positive = possible contamination.\n")

    models = sorted(set(d['model_name'] for d in data))
    models_no_ceil = [m for m in models if 'ceiling' not in m]

    results = []
    for model in models_no_ceil:
        l1_scores = [d['overall_score'] for d in data if d['model_name'] == model and d['level'] == 1]
        l3_scores = [d['overall_score'] for d in data if d['model_name'] == model and d['level'] == 3]
        if l1_scores and l3_scores:
            l1_mean = np.mean(l1_scores)
            l3_mean = np.mean(l3_scores)
            diff = l1_mean - l3_mean
            results.append((model, l1_mean, l3_mean, diff))

    results.sort(key=lambda x: -x[3])

    print(f"  {'Model':<40} {'L1 Mean':>8} {'L3 Mean':>8} {'L1-L3':>8} {'Flag':>12}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")
    for model, l1, l3, diff in results:
        flag = "INVESTIGATE" if diff > 0.5 else ("monitor" if diff > 0.3 else "")
        print(f"  {model:<40} {l1:>8.3f} {l3:>8.3f} {diff:>+8.3f} {flag:>12}")

    print()


# ============================================================
# ANALYSIS 5: High-CV Question Analysis for qwen3.5-27b-dense
# ============================================================
def analysis_5_high_cv_questions(data, conn):
    print("=" * 70)
    print("ANALYSIS 5: HIGH-CV QUESTION ANALYSIS — qwen3.5-27b-dense")
    print("=" * 70)
    print("Identifying the 26 high-CV questions (CV > 0.15) and their patterns.\n")

    model = "qwen3.5-27b-dense"
    model_data = [d for d in data if d['model_name'] == model]

    # Group by question
    questions = {}
    for d in model_data:
        qid = d['question_id']
        if qid not in questions:
            questions[qid] = []
        questions[qid].append(d['overall_score'])

    # Compute CV per question
    high_cv = []
    all_cvs = []
    for qid, scores in questions.items():
        if len(scores) >= 2:
            m = np.mean(scores)
            sd = np.std(scores, ddof=1)
            cv = sd / m if m > 0 else 0
            all_cvs.append((qid, cv, m, sd, scores))
            if cv > 0.15:
                high_cv.append((qid, cv, m, sd, scores))

    print(f"  Total questions with 2+ runs: {len(all_cvs)}")
    print(f"  High-CV questions (CV > 0.15): {len(high_cv)}")

    # Get domain and level info
    domain_names = dict(conn.execute("SELECT id, display_name FROM domains").fetchall())

    # Enrich with domain/level
    enriched = []
    for qid, cv, m, sd, scores in high_cv:
        row = conn.execute("SELECT domain_id, level, text FROM questions WHERE id = ?", (qid,)).fetchone()
        if row:
            enriched.append({
                'qid': qid,
                'cv': cv,
                'mean': m,
                'std': sd,
                'scores': scores,
                'domain_id': row['domain_id'],
                'domain': domain_names.get(row['domain_id'], f"D{row['domain_id']}"),
                'level': row['level'],
                'text_preview': row['text'][:80] + '...' if len(row['text']) > 80 else row['text'],
            })

    enriched.sort(key=lambda x: -x['cv'])

    # Domain distribution
    print(f"\n  --- Domain distribution of high-CV questions ---")
    domain_counts = {}
    for e in enriched:
        d = f"D{e['domain_id']}: {e['domain']}"
        domain_counts[d] = domain_counts.get(d, 0) + 1
    for d, c in sorted(domain_counts.items()):
        print(f"    {d:<40} {c:>3} questions")

    # Level distribution
    print(f"\n  --- Level distribution of high-CV questions ---")
    level_counts = {1: 0, 2: 0, 3: 0}
    for e in enriched:
        level_counts[e['level']] += 1
    for lvl in [1, 2, 3]:
        total_at_level = len([d for d in model_data if d['level'] == lvl]) // 3  # approx questions
        pct = level_counts[lvl] / total_at_level * 100 if total_at_level > 0 else 0
        print(f"    L{lvl}: {level_counts[lvl]:>3} high-CV questions ({pct:.1f}% of L{lvl} questions)")

    # Domain x Level crosstab
    print(f"\n  --- Domain x Level crosstab ---")
    print(f"  {'Domain':<30} {'L1':>5} {'L2':>5} {'L3':>5}")
    for did in sorted(set(e['domain_id'] for e in enriched)):
        dname = domain_names.get(did, f"D{did}")
        l1 = sum(1 for e in enriched if e['domain_id'] == did and e['level'] == 1)
        l2 = sum(1 for e in enriched if e['domain_id'] == did and e['level'] == 2)
        l3 = sum(1 for e in enriched if e['domain_id'] == did and e['level'] == 3)
        print(f"  {dname:<30} {l1:>5} {l2:>5} {l3:>5}")

    # Top 10 highest CV questions
    print(f"\n  --- Top 10 highest-CV questions ---")
    for e in enriched[:10]:
        print(f"    Q{e['qid']:>3} (D{e['domain_id']} L{e['level']}) CV={e['cv']:.3f} "
              f"mean={e['mean']:.2f} scores={[round(s,2) for s in e['scores']]}")
        print(f"          {e['text_preview']}")

    # Pattern analysis
    print(f"\n  --- Pattern Summary ---")
    mean_cv_high = np.mean([e['cv'] for e in enriched])
    mean_score_high = np.mean([e['mean'] for e in enriched])
    mean_score_all = np.mean([m for _, _, m, _, _ in all_cvs])
    print(f"    Mean CV of high-CV questions: {mean_cv_high:.3f}")
    print(f"    Mean score of high-CV questions: {mean_score_high:.3f}")
    print(f"    Mean score of all questions: {mean_score_all:.3f}")
    print(f"    High-CV questions tend to be {'lower-scoring' if mean_score_high < mean_score_all else 'higher-scoring'} "
          f"(diff: {mean_score_high - mean_score_all:+.3f})")

    print()


# ============================================================
# ANALYSIS 6: Inflation Index Formula
# ============================================================
def analysis_6_inflation_index():
    print("=" * 70)
    print("ANALYSIS 6: VERBOSITY INFLATION INDEX — FORMULA DEFINITION")
    print("=" * 70)
    print()
    print("  Source: scripts/statistical_analysis.py, lines 64-67")
    print("  Function: verbosity_bias_analysis()")
    print()
    print("  FORMULA:")
    print("  --------")
    print("  1. Compute median_wc = median of word counts for all responses of a model")
    print("  2. Split responses into two groups:")
    print("       'above': responses where word_count > median_wc")
    print("       'below': responses where word_count <= median_wc")
    print("  3. inflation = (mean(above_scores) - mean(below_scores)) / mean(below_scores)")
    print()
    print("  In Python:")
    print("    median_wc = np.median(word_counts)")
    print("    above = [score for score, wc in zip(scores, word_counts) if wc > median_wc]")
    print("    below = [score for score, wc in zip(scores, word_counts) if wc <= median_wc]")
    print("    inflation = (np.mean(above) - np.mean(below)) / np.mean(below)")
    print()
    print("  INTERPRETATION:")
    print("  - This is a median-split relative difference measure")
    print("  - A value of 0.05 means responses above median length score 5% higher")
    print("  - It measures how much 'extra credit' longer responses receive")
    print("  - Does NOT control for question difficulty or model quality")
    print()

    # Print actual values from the JSON
    with open(STATS_JSON) as f:
        stats_data = json.load(f)

    vb = stats_data.get("verbosity_bias", {})
    print(f"  {'Model':<40} {'Inflation':>10} {'Pearson r':>10} {'Flagged':>8}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*8}")
    for model in sorted(vb.keys()):
        v = vb[model]
        print(f"  {model:<40} {v['verbosity_inflation_index']:>10.4f} {v['pearson_r']:>10.4f} {'YES' if v['flagged'] else '':>8}")

    print()


# ============================================================
# MAIN
# ============================================================
def main():
    print()
    print("*" * 70)
    print("  MARS-Bench Deep Statistical Analysis")
    print("  6 analyses on benchmark data")
    print("*" * 70)
    print()

    conn = get_conn()
    data = load_eval_data(conn)
    print(f"Loaded {len(data)} evaluations, "
          f"{len(set(d['model_name'] for d in data))} models, "
          f"{len(set(d['question_id'] for d in data))} questions\n")

    analysis_1_residualized_scores(data)
    analysis_2_within_level_verbosity(data)
    analysis_3_d7_pairwise(data)
    analysis_4_contamination_proxy(data)
    analysis_5_high_cv_questions(data, conn)
    analysis_6_inflation_index()

    print("=" * 70)
    print("ALL ANALYSES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
