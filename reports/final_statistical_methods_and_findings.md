# MARS-Bench Final Statistical Measurements

Generated: `2026-05-07T08:58:35+00:00`

## Raw Data Locations

- Publication-safe SQLite DB: `data/mars_bench_stats_public.sqlite`
- Compressed public DB artifact: `data/mars_bench_stats_public.sqlite.gz`
- Cluster bootstrap output: `results/cluster_bootstrap/final_2026_05_07_cluster_bootstrap_results.json`
- Source working DB filename: `mars_bench.db`
- Source working DB SHA-256: `02635670449b41b8aadba3b4ce22a08911ef75b5ca3f0a0582e1ce53d41f2ab8`

The public DB is release-scoped and redacts local paths and raw long-form source/context packet body text. It retains questions, reference answers, model responses, scores, rubrics, judge notes, audit flags, material hashes, and structured source/audit provenance needed for statistical replication.

## Corpus and Judging Coverage

- Release: `mars_bench_v2_0_d1_d2_d3_d4_d5_d6_d7_rebuild_candidate`
- Release questions in public DB: `251`
- Release responses in public DB: `10542`
- Paired-analysis questions: `229`
- Paired Codex-Claude response rows: `9618`
- Nonpaired release responses outside paired model-selection analysis: `924`
- Codex rows matching final filter: `9618`
- Claude rows matching final filter: `9618`
- Paired fraction of Codex rows: `1.000`
- Paired fraction of Claude rows: `1.000`
- Bootstrap reps: `1000`; seed: `20260503`; interval: `percentile`

The 924 nonpaired rows are not stray legacy responses. They are current-release D1 structured morphology adjunct responses: 22 questions x 14 models x 3 runs, judged only by `d1-structured-morphology-scorer-2026-04-25`. Because they have no final paired Codex-Claude evaluations, they are excluded from the paired model-selection statistics and should be analyzed only with a morphology-specific metric.

## Measurement 1: Inter-Judge Reliability

Purpose: verify that the two independent judge lanes are measuring the same response-quality construct before using averaged scores for model selection.

Formulae:

- Mean signed judge difference: `mean(Claude_i - Codex_i)`.
- Mean absolute difference: `mean(abs(Claude_i - Codex_i))`.
- Two-way mixed-effects absolute-agreement ICC uses McGraw and Wong's ICC(A,1) form:
  `ICC(A,1) = (MS_R - MS_E) / (MS_R + (k - 1)MS_E + k(MS_C - MS_E)/n)`.
- For the averaged two-judge score: `ICC(A,k) = (MS_R - MS_E) / (MS_R + (MS_C - MS_E)/n)` with `k = 2`.
- Bland-Altman normal limits: `mean(diff) +/- 1.96 * SD(diff)`.
- Bland-Altman empirical limits: the `2.5th` and `97.5th` percentiles of the paired differences.

Findings:

- ICC(A,1): `0.9267` with bootstrap CI `{'ci_95_high': 0.9342, 'ci_95_low': 0.9182}`.
- ICC(A,2): `0.962` with bootstrap CI `{'ci_95_high': 0.966, 'ci_95_low': 0.9573}`.
- Mean signed difference, Claude minus Codex: `0.0271`.
- Mean absolute difference: `0.2261`.
- Pearson / Spearman: `0.9312` / `0.9218`.
- Within 0.50 points: `0.8539`; within 1.00 point: `0.9833`.
- Bland-Altman normal limits: `[-0.6921, 0.7464]`.
- Bland-Altman empirical limits: `[-0.7, 0.9]`.

Interpretation: ICC supports aggregate use of the averaged Codex-Claude score. The Bland-Altman tail width means row-level disagreements still deserve qualitative audit, especially in contested source-critical cases.

## Measurement 2: Role-Audition Weighted Composite

Purpose: rank models by the domain mix most relevant to Synthetic Theological Agent base-model selection, rather than by raw row count.

Formula:

`Composite_m = 0.08*D1_m + 0.17*D2_m + 0.175*D3_m + 0.07*D4_m + 0.14*D5_m + 0.145*D6_m + 0.22*D7_m`.

Top role-weighted models:

| Rank | Model | Composite | 95% CI | Top-3 bootstrap rate |
|---:|---|---:|---:|---:|
| 1 | `hermes-3-llama-3.1-405b` | `2.7995` | `[2.7222, 2.8796]` | `1.0` |
| 2 | `llama4-maverick-17b-128e-moe` | `2.5557` | `[2.4975, 2.6176]` | `0.999` |
| 3 | `llama-3.3-70b-instruct` | `2.4952` | `[2.4335, 2.5568]` | `0.873` |
| 4 | `gemma-4-31b-it` | `2.4566` | `[2.4048, 2.5051]` | `0.128` |
| 5 | `qwen3.5-397b-a17b-moe` | `2.3978` | `[2.3595, 2.4372]` | `0.0` |
| 6 | `deepseek-v4-flash` | `2.383` | `[2.3434, 2.4245]` | `0.0` |
| 7 | `glm-5.1` | `2.3681` | `[2.3214, 2.4108]` | `0.0` |
| 8 | `command-a-03-2025` | `2.3452` | `[2.2916, 2.406]` | `0.0` |

Paired composite difference tests:

- `llama-3.3-70b-instruct - gemma-4-31b-it` observed difference `0.0386`, bootstrap CI `[-0.0251, 0.1052]`, includes zero: `True`.

## Measurement 3: Verbosity Bias and Length Adjustment

Purpose: detect whether longer responses receive higher scores independent of domain, level, and regime. This matters because verbose theological prose can appear scholarly without being more accurate.

Formulae:

- Raw length-score association: Pearson `r(word_count, score)` within each model.
- Verbosity inflation index: `(mean(score | words > median_words) - mean(score | words <= median_words)) / mean(score | words <= median_words)`.
- Length-adjusted score: fit `score ~ log(1 + words) + domain FE + level FE + regime FE`; then adjust each score to the global mean log length: `score_adj_i = score_i - beta_len*(log_words_i - mean(log_words))`.

Overall log-length coefficient: `-0.450366`; p-value `1.29e-297`.

Top length-adjusted rankings:

| Adjusted Rank | Model | Raw Mean | Raw Rank | Length-Adjusted Mean | Rank Shift |
|---:|---|---:|---:|---:|---:|
| 1 | `qwen3.5-397b-a17b-moe` | `2.7725` | `3` | `2.9063` | `2` |
| 2 | `deepseek-v4-flash` | `2.7421` | `5` | `2.8655` | `3` |
| 3 | `glm-5.1` | `2.719` | `7` | `2.785` | `4` |
| 4 | `qwen3.5-122b-a10b` | `2.614` | `9` | `2.772` | `5` |
| 5 | `hermes-3-llama-3.1-405b` | `3.0119` | `1` | `2.6866` | `-4` |
| 6 | `qwen3.6-27b-fp8` | `2.5323` | `12` | `2.6744` | `6` |
| 7 | `qwen3.6-35b-a3b-q8_0` | `2.5386` | `11` | `2.6587` | `4` |
| 8 | `gemma-4-31b-it` | `2.7542` | `4` | `2.6502` | `-4` |

## Measurement 4: Run-to-Run Variance

Purpose: assess whether a model is stable across repeated generations for the same question. Stable base models are preferable for STA work because rare erratic behavior can become costly after fine-tuning.

Formulae:

- For model `m` and question `q`, `variance_mq = sample_variance(score_mq1, score_mq2, ...)`.
- Coefficient of variation: `CV_mq = SD_mq / mean_mq` when `mean_mq > 0`.
- A high-CV question is flagged at `CV > 0.15`.

Lowest mean-CV models:

| Model | Mean CV | Median CV | High-CV Questions |
|---|---:|---:|---:|
| `qwen3.5-397b-a17b-moe` | `0.0688` | `0.0455` | `28` |
| `deepseek-v4-flash` | `0.0733` | `0.0535` | `30` |
| `glm-5.1` | `0.0833` | `0.0591` | `33` |
| `mistral-medium-3.5` | `0.0845` | `0.0639` | `34` |
| `gemma-4-31b-it` | `0.086` | `0.0595` | `35` |
| `llada2.1-flash` | `0.0889` | `0.0702` | `42` |
| `qwen3.6-27b-fp8` | `0.091` | `0.0675` | `36` |
| `qwen3.5-122b-a10b` | `0.0937` | `0.0596` | `49` |

## Measurement 5: Trap and Response-Integrity Resilience

Purpose: distinguish ordinary weak answers from failure modes that make a model risky for scholarly deployment: traps, self-talk, loops, generic-Gnostic confusion, unsupported claims, and hard response-integrity failures.

Formulae:

- Trap delta: `mean(trap question scores) - mean(non-trap question scores)` using question-level model means.
- Response-integrity failure rate: `RI_failures / total_release_responses`.
- If a model has zero RI failures, the approximate one-sided 95% upper bound is the rule of three: `3 / n`.

Response-integrity profile:

| Model | RI Failures | All Release Responses | RI Rate | Zero-Failure Upper Bound | Domains |
|---|---:|---:|---:|---:|---|
| `qwen3.6-35b-a3b-q8_0` | `5` | `753` | `0.00664` | `None` | `{"D7": 5}` |
| `command-a-03-2025` | `2` | `753` | `0.002656` | `None` | `{"D3": 2}` |
| `qwen3.5-9b` | `2` | `753` | `0.002656` | `None` | `{"D1": 2}` |
| `qwen3.6-27b-fp8` | `2` | `753` | `0.002656` | `None` | `{"D4": 1, "D7": 1}` |
| `qwen3.5-122b-a10b` | `1` | `753` | `0.001328` | `None` | `{"D7": 1}` |
| `deepseek-v4-flash` | `0` | `753` | `0.0` | `0.003984` | `{}` |
| `gemma-4-31b-it` | `0` | `753` | `0.0` | `0.003984` | `{}` |
| `glm-5.1` | `0` | `753` | `0.0` | `0.003984` | `{}` |
| `hermes-3-llama-3.1-405b` | `0` | `753` | `0.0` | `0.003984` | `{}` |
| `llada2.1-flash` | `0` | `753` | `0.0` | `0.003984` | `{}` |
| `llama-3.3-70b-instruct` | `0` | `753` | `0.0` | `0.003984` | `{}` |
| `llama4-maverick-17b-128e-moe` | `0` | `753` | `0.0` | `0.003984` | `{}` |
| `mistral-medium-3.5` | `0` | `753` | `0.0` | `0.003984` | `{}` |
| `qwen3.5-397b-a17b-moe` | `0` | `753` | `0.0` | `0.003984` | `{}` |

## Measurement 6: Difficulty Gradient

Purpose: check whether models collapse from foundational recall (L1) to synthetic expert reasoning (L3). This directly supports the paper's impersonation-versus-improvisation framing.

Formula: `L1_minus_L3 = mean_L1_score - mean_L3_score`. Larger positive values indicate a larger drop at expert/synthetic difficulty.

## Measurement 7: Score Distribution and Floor Tail

Purpose: means can hide brittle models. The distribution analysis reports median, standard deviation, skewness, kurtosis, minimum/maximum, percent below 2, percent at least 3, and percent at least 4 using question-level model means.

## Measurement 8: Pairwise Wilcoxon Comparisons

Purpose: provide appendix-level pairwise tests over question-level model means. This is secondary to the role-weighted bootstrap because the paper's selection rule is multi-criteria rather than a p-value leaderboard.

Formulae:

- Paired difference for each common question: `d_q = score_Aq - score_Bq`.
- Wilcoxon signed-rank tests whether the median paired difference is zero.
- Benjamini-Hochberg adjusted p-values are reported as an FDR-oriented multiplicity correction under the usual independence/positive-dependence assumptions.
- The reported standardized mean difference is descriptive: `mean(A - B) / pooled_SD`; it is not a paired-samples `d_z`.

## Files Produced

- `results/cluster_bootstrap/final_2026_05_07_cluster_bootstrap_results.json`
- `results/cluster_bootstrap/final_2026_05_07_cluster_bootstrap_leaderboard.csv`
- `results/cluster_bootstrap/final_2026_05_07_interjudge_reliability.json`
- `results/diagnostics/final_bias_and_selection_diagnostics.json`
- `results/diagnostics/*.csv` appendix tables

## References

- Bland, J. M., and Altman, D. G. 1986. Statistical methods for assessing agreement between two methods of clinical measurement. The Lancet 327:307-310.
- Benjamini, Y., and Hochberg, Y. 1995. Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society B 57:289-300.
- Cohen, J. 1988. Statistical Power Analysis for the Behavioral Sciences. 2nd ed.
- Efron, B., and Tibshirani, R. J. 1993. An Introduction to the Bootstrap.
- Hanley, J. A., and Lippman-Hand, A. 1983. If nothing goes wrong, is everything all right? JAMA 249:1743-1745.
- Koo, T. K., and Li, M. Y. 2016. A guideline of selecting and reporting intraclass correlation coefficients for reliability research. Journal of Chiropractic Medicine 15:155-163.
- Lakens, D. 2017. Equivalence tests: a practical primer for t tests, correlations, and meta-analyses. Social Psychological and Personality Science 8:355-362.
- McGraw, K. O., and Wong, S. P. 1996. Forming inferences about some intraclass correlation coefficients. Psychological Methods 1:30-46.
- Schenker, N., and Gentleman, J. F. 2001. On judging the significance of differences by examining the overlap between confidence intervals. The American Statistician 55:182-186.
- Shrout, P. E., and Fleiss, J. L. 1979. Intraclass correlations: uses in assessing rater reliability. Psychological Bulletin 86:420-428.
- Wilcoxon, F. 1945. Individual comparisons by ranking methods. Biometrics Bulletin 1:80-83.
