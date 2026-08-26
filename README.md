# MARS-Bench Statistical Replication Package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20067936.svg)](https://doi.org/10.5281/zenodo.20067936)
[![Tests](https://github.com/maskull42/MARS-bench-stats/actions/workflows/tests.yml/badge.svg)](https://github.com/maskull42/MARS-bench-stats/actions/workflows/tests.yml)

**Paper:** ["As If, Not As Was: MARS-Bench and the Agentic Audition of LLMs for Marcionite Surrogation" (PDF)](paper/As-If-Not-As-Was_MARS-Bench.pdf) — A.G. Elrod, presented at the Expert Meeting on Agentic AI and the Humanities, VU Amsterdam, 12 May 2026.

**Slides:** [talk deck, public edition (PDF)](paper/MARS-Bench_talk_public-edition.pdf) — the 12 May 2026 talk, including the post-freeze three-judge re-analysis; two film images are replaced by a citation card in this edition.

The repository contains two distinct public snapshots:

- **Frozen paper archive, v2026.05.07:** 14 models and 10,542 release
  responses. The accompanying paper, "As If, Not As Was: MARS-Bench and the
  Agentic Audition of LLMs for Marcionite Surrogation," cites this snapshot at
  Zenodo DOI
  [10.5281/zenodo.20067936](https://doi.org/10.5281/zenodo.20067936).
- **Current analysis tag, v2026.06.05:** 15 models and 11,295 release responses.
  This snapshot adds the Gemma 4 12B A6000 run, its completed primary
  Codex/Claude/DeepSeek judging rows, and fully recalculated two-judge,
  three-judge, and diagnostic statistics.

The Zenodo DOI and `CITATION.cff` remain anchored to the frozen v2026.05.07
paper archive. See `CHANGELOG.md` for the full revision history.

This repository contains the publication-safe data export, statistical code,
and final measurement outputs for the MARS-Bench agentic digital humanities
benchmark analysis. The package is designed to support reproducibility for the
accompanying paper: the public database includes benchmark questions, reference
answers, model responses, judge scores, judge notes, audit notes, audit flags,
prompt/rubric materials, and provenance hashes needed to regenerate the final
statistics. Machine-local paths, private worker/account tables, worker-claim
state, and raw long-form source/context packet bodies have been removed or
redacted. Structured lexical, syntactic, and provenance metadata needed for
replication is retained.

## Repository Contents

- `data/mars_bench_stats_public.sqlite.gz`: compressed publication-safe SQLite
  database. This is the tracked public data artifact.
- `data/SHA256SUMS`: checksums for the public database and final result files.
- `scripts/export_publication_safe_db.py`: maintainer script that builds the
  public export from the private working SQLite database.
- `scripts/cluster_bootstrap_mars_results.py`: final paired Codex-Claude,
  cluster-aware bootstrap analysis.
- `scripts/run_bias_and_selection_diagnostics.py`: final model-selection,
  bias, stability, distribution, and appendix diagnostics.
- `scripts/parameter_scaling_analysis.py`: parameter-count correlation and
  residual diagnostics using the three-judge model scores.
- `scripts/three_judge_analysis.py` (added in v2026.05.10, updated in
  v2026.06.05): three-judge re-analysis adding DeepSeek V4 Flash via OpenCode
  CLI to the original
  Codex+Claude pair. Computes ICC(A,1) and ICC(A,k=3), pairwise reliability
  for the three lane pairs, three-judge role-weighted composites, paired
  Llama-Gemma cluster bootstrap under the three-judge mean, and a per-model
  DeepSeek-as-judge offset diagnostic for self-preference probing.
- `results/cluster_bootstrap/`: final cluster bootstrap JSON/CSV outputs.
- `results/diagnostics/`: final diagnostic JSON/CSV outputs.
- `results/diagnostics/model_parameter_counts_2026_06_05.json`: audited
  provenance for `models.size_b` parameter-count metadata.
- `results/three_judge/` (added in v2026.05.10): three-judge analysis outputs
  for the v2026.05.10 and v2026.06.05 releases.
- `reports/final_statistical_methods_and_findings.md`: detailed methods,
  formulas, data locations, and numerical findings.
- `reports/three_judge_analysis_2026_05_10.md` (added in v2026.05.10):
  methods and findings for the three-judge re-analysis.
- `reports/three_judge_analysis_2026_06_05.md` (added in v2026.06.05):
  methods and findings for the three-judge re-analysis after the Gemma 4 12B
  addition.
- `reports/parameter_scaling_2026_06_05.md`: parameter-count scaling findings
  after filling audited `models.size_b` metadata.
- `CHANGELOG.md`: release history.
- `legacy_reference_scripts/`: earlier statistical scripts preserved for
  audit history and method continuity.
- `docs/`: data-export, response-set, and publication-safety notes.

## Reproducing the Final Statistics

Create an environment and install the small dependency set:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Uncompress the public database:

```bash
gzip -dk data/mars_bench_stats_public.sqlite.gz
```

Run the final cluster bootstrap:

```bash
python scripts/cluster_bootstrap_mars_results.py \
  --db data/mars_bench_stats_public.sqlite \
  --release-label mars_bench_v2_0_d1_d2_d3_d4_d5_d6_d7_rebuild_candidate \
  --output-dir results/cluster_bootstrap \
  --label final_2026_06_05 \
  --bootstrap-reps 1000 \
  --seed 20260503
```

Run the final bias, selection, and appendix diagnostics:

```bash
python scripts/run_bias_and_selection_diagnostics.py \
  --db data/mars_bench_stats_public.sqlite \
  --cluster-json results/cluster_bootstrap/final_2026_06_05_cluster_bootstrap_results.json \
  --results-dir results/diagnostics \
  --report reports/final_statistical_methods_and_findings.md
```

Run the three-judge re-analysis (v2026.05.10 addition):

```bash
python scripts/three_judge_analysis.py \
  --db data/mars_bench_stats_public.sqlite \
  --release-label mars_bench_v2_0_d1_d2_d3_d4_d5_d6_d7_rebuild_candidate \
  --output-dir results/three_judge \
  --label final_2026_06_05 \
  --bootstrap-reps 1000 \
  --seed 20260503
```

The public database contains 251 release questions and 11,295 release
responses. The paired model-selection analysis uses 229 questions and 10,305
paired Codex-Claude response rows across 15 models. The remaining 990 responses
are current release D1 structured morphology adjunct rows: 22 questions x 15
models x 3 runs. Those rows were scored only by
`d1-structured-morphology-scorer-2026-04-25` and have no final paired
Codex-Claude evaluations, so they are not included in the paired
model-selection statistics.

The final paired filter uses the Codex judge model
`gpt-5.5-medium-codex-cli` and the Claude judge model
`claude-opus-4-7-medium-claude-cli` with the final primary/comparison judge
version filters recorded in the cluster bootstrap JSON metadata. The
`evaluations` table also preserves supplementary/earlier evaluation rows for
auditability; the final analyses select only the explicitly filtered paired
Codex-Claude rows.

## Raw Data Location

The raw public data for replication is:

```text
data/mars_bench_stats_public.sqlite.gz
```

After decompression, use:

```text
data/mars_bench_stats_public.sqlite
```

The private working database is not distributed. Its source filename and
SHA-256 hash are recorded in `publication_export_metadata` inside the public
database so the export can be audited without exposing a local machine path.

## Citation

For the archived `v2026.05.07` release, cite the Zenodo version DOI:

```text
https://doi.org/10.5281/zenodo.20067936
```

The GitHub release archived by Zenodo is:

```text
https://github.com/maskull42/MARS-bench-stats/releases/tag/v2026.05.07
```

## Statistical Measurements

The model-selection scripts use paired average response scores. For model `m`,
question `q`, run `r`, and judge `j`, let `s_{mqrj}` be the score. The paired
response score is

```text
S_{mqr} = (s_{mqr,Codex} + s_{mqr,Claude}) / 2
```

Question-level model means are:

```text
Q_{mq} = mean_r(S_{mqr})
```

### 1. Inter-Judge Reliability

Purpose: check whether the Codex and Claude judge lanes measure the same
response-quality construct before their scores are averaged.

Measurements:

```text
mean signed difference = mean(s_Claude - s_Codex)
mean absolute difference = mean(abs(s_Claude - s_Codex))
```

The main reliability coefficient is a two-way mixed-effects,
absolute-agreement intraclass correlation coefficient, ICC(A,1):

```text
ICC(A,1) = (MS_R - MS_E) /
           (MS_R + (k - 1)MS_E + k(MS_C - MS_E)/n)
```

For the averaged two-judge score:

```text
ICC(A,k) = (MS_R - MS_E) / (MS_R + (MS_C - MS_E)/n)
```

Here `MS_R` is the response mean square, `MS_C` is the judge-column mean
square, `MS_E` is residual mean square, `n` is the number of judged responses,
and `k = 2` judges. Bland-Altman limits are also reported:

```text
mean(diff) +/- 1.96 * SD(diff)
```

Non-expert interpretation: if ICC(A,k) is high, the average of the two judge
lanes is stable enough to use as the main score. Bland-Altman limits show how
large individual row disagreements can still be.

Final finding: ICC(A,2) is 0.9617 with a bootstrap 95% CI of 0.9572 to 0.9656.

### 2. Cluster-Aware Bootstrap Ranking

Purpose: estimate ranking uncertainty without pretending that every response
row is independent. Rows are clustered by question because multiple model runs
answer the same question.

For each bootstrap replicate, the 229 paired-analysis questions are sampled
with replacement within each domain. All model/run rows attached to sampled
questions are carried into the replicate. The scripts recompute observed means,
ranks, top-1 rates, top-3 rates, and percentile 95% confidence intervals.

Non-expert interpretation: this asks how much the leaderboard would move if
the benchmark had sampled a slightly different set of questions from the same
domain structure.

### 3. Role-Audition Weighted Composite

Purpose: rank models by the domain mix most relevant to Synthetic Theological
Agent base-model selection, rather than by raw row counts.

Let `D1_m` through `D7_m` be a model's domain means. The composite is:

```text
Composite_m =
  0.08  * D1_m +
  0.17  * D2_m +
  0.175 * D3_m +
  0.07  * D4_m +
  0.14  * D5_m +
  0.145 * D6_m +
  0.22  * D7_m
```

Non-expert interpretation: this gives more influence to domains that matter
more for the intended scholarly role, especially Marcion studies and patristic
knowledge, while still retaining all seven benchmark domains.

Final finding: `hermes-3-llama-3.1-405b` ranks first on the weighted composite;
`llama4-maverick-17b-128e-moe` ranks second; `llama-3.3-70b-instruct` ranks
third. The targeted paired composite difference between
`llama-3.3-70b-instruct` and `gemma-4-31b-it` includes zero.

### 4. Pairwise Composite Difference Tests

Purpose: test specific model comparisons using paired bootstrap differences,
not overlap between two separate confidence intervals.

For models `a` and `b`:

```text
Delta_b = Composite_a - Composite_b
```

The 95% interval is the 2.5th and 97.5th percentiles of bootstrap `Delta_b`.
If the interval includes zero, the observed advantage is not treated as
stable under this benchmark sample.

### 5. Verbosity Bias and Length Adjustment

Purpose: detect whether longer answers are rewarded independently of accuracy,
domain, difficulty level, and response regime.

Raw verbosity association is reported with Pearson correlation:

```text
r = cov(words, score) / (SD_words * SD_score)
```

The verbosity inflation index is:

```text
inflation =
  (mean(score | words > median_words) -
   mean(score | words <= median_words)) /
   mean(score | words <= median_words)
```

The length-adjustment regression is:

```text
score_i =
  alpha + beta_len * log(1 + words_i) +
  domain fixed effects +
  level fixed effects +
  regime fixed effects +
  error_i
```

Scores are adjusted to the global mean log length:

```text
score_adj_i = score_i - beta_len * (log_words_i - mean(log_words))
```

Non-expert interpretation: this separates real performance from a tendency to
write more words. It is especially relevant because long theological prose can
sound scholarly even when it is not more accurate.

Final finding: the overall adjusted log-word coefficient is -0.447318
(`p < 0.001` in the OLS diagnostic), so longer answers were not generally being
rewarded after the covariates used here.

### 6. Run-to-Run Variance

Purpose: measure whether a model is stable across repeated generations for
the same question.

For each model-question pair:

```text
variance_mq = sample_variance(S_{mq1}, S_{mq2}, ...)
CV_mq = SD_mq / mean_mq
```

High-CV questions are flagged at `CV > 0.15`.

Non-expert interpretation: a model with the same average score can still be
riskier if it sometimes answers the same kind of question very well and other
times very poorly.

### 7. Trap and Response-Integrity Resilience

Purpose: distinguish low-scoring answers from failure modes that matter for
scholarly deployment, including trap susceptibility, loops, self-talk, generic
Gnostic confusion, unsupported claims, and hard response-integrity failures.

Trap sensitivity:

```text
trap_delta_m =
  mean(score on trap questions)_m -
  mean(score on non-trap questions)_m
```

Response-integrity failure rate:

```text
RI_rate_m = RI_failures_m / total_release_responses_m
```

For a model with zero observed RI failures, the approximate one-sided 95%
upper bound is the rule of three:

```text
upper_bound ~= 3 / n
```

Non-expert interpretation: zero observed failures is not proof of zero risk;
the rule of three gives a conservative upper bound for rare failures in the
observed sample.

### 8. Difficulty Gradient

Purpose: measure whether models collapse from foundational questions to
synthetic expert reasoning.

```text
L1_minus_L3_m = mean_L1_score_m - mean_L3_score_m
```

Non-expert interpretation: a larger positive value means a larger drop from
basic recall to difficult synthesis.

### 9. Score Distribution and Floor Tail

Purpose: inspect whether means hide brittle behavior.

Reported statistics include mean, median, standard deviation, skewness,
kurtosis, minimum, maximum, percent below 2, percent at least 3, and percent
at least 4 using question-level model means.

Non-expert interpretation: two models can have similar means while one has a
larger low-score tail.

### 10. Pairwise Wilcoxon Tests and False Discovery Rate Control

Purpose: provide appendix-level paired comparisons over question-level model
means.

For models `a` and `b` on common questions:

```text
d_q = Q_{aq} - Q_{bq}
```

The Wilcoxon signed-rank test evaluates whether the median paired difference
is zero. Benjamini-Hochberg adjusted p-values are reported as an FDR-oriented
multiplicity correction under the usual independence/positive-dependence
assumptions.

Non-expert interpretation: this is a secondary statistical appendix. It is
not the selection rule; the role-weighted bootstrap and qualitative failure
profiles are more directly tied to model selection. The reported standardized
mean difference uses pooled SD and is descriptive; it is not a paired-samples
`d_z`.

## Publication Safety

The public export intentionally excludes private user/account tables, active
worker claims, and raw long-form source/context body text. Local filesystem
paths are redacted, while hashes and structured provenance are retained. See
`docs/publication_safety.md`, `docs/data_dictionary.md`, and
`docs/response_sets.md` for details.

## Citations

- Benjamini, Y., and Hochberg, Y. 1995. Controlling the false discovery rate:
  a practical and powerful approach to multiple testing. Journal of the Royal
  Statistical Society B 57:289-300. doi:10.1111/j.2517-6161.1995.tb02031.x.
- Bland, J. M., and Altman, D. G. 1986. Statistical methods for assessing
  agreement between two methods of clinical measurement. The Lancet
  327:307-310. doi:10.1016/S0140-6736(86)90837-8.
- Cohen, J. 1988. Statistical Power Analysis for the Behavioral Sciences.
  2nd ed. Hillsdale, NJ: Lawrence Erlbaum Associates.
- Efron, B., and Tibshirani, R. J. 1993. An Introduction to the Bootstrap.
  New York: Chapman and Hall/CRC.
- Hanley, J. A., and Lippman-Hand, A. 1983. If nothing goes wrong, is
  everything all right? JAMA 249:1743-1745.
  doi:10.1001/jama.1983.03330370053031.
- Koo, T. K., and Li, M. Y. 2016. A guideline of selecting and reporting
  intraclass correlation coefficients for reliability research. Journal of
  Chiropractic Medicine 15:155-163. doi:10.1016/j.jcm.2016.02.012.
- Lakens, D. 2017. Equivalence tests: a practical primer for t tests,
  correlations, and meta-analyses. Social Psychological and Personality
  Science 8:355-362. doi:10.1177/1948550617697177.
- McGraw, K. O., and Wong, S. P. 1996. Forming inferences about some
  intraclass correlation coefficients. Psychological Methods 1:30-46.
  doi:10.1037/1082-989X.1.1.30.
- Schenker, N., and Gentleman, J. F. 2001. On judging the significance of
  differences by examining the overlap between confidence intervals. The
  American Statistician 55:182-186. doi:10.1198/000313001317097960.
- Shrout, P. E., and Fleiss, J. L. 1979. Intraclass correlations: uses in
  assessing rater reliability. Psychological Bulletin 86:420-428.
  doi:10.1037/0033-2909.86.2.420.
- Wilcoxon, F. 1945. Individual comparisons by ranking methods. Biometrics
  Bulletin 1:80-83. doi:10.2307/3001968.
