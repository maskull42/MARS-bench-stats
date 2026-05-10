# MARS-Bench Three-Judge Re-Analysis (v2026.05.10)

Generated: `2026-05-10`. Companion document to `final_statistical_methods_and_findings.md`, which remains frozen at the v2026.05.07 release. This document reports the three-judge re-analysis added at the v2026.05.10 release and is intended for citation in materials produced after the paper was finalized.

## Provenance

- Source database: `data/mars_bench_stats_public.sqlite` at the v2026.05.10 release (SHA-256 `4f587ca90b0ffc533389a5b889d016f4e946e4b680209787815e761414593a0c`).
- Three-judge paired-rows SHA-256: `c7058eddba411d9585270f44fbe99d1c4be7ed24dd89ea22aed7045d5e143775`.
- Bootstrap parameters: 1000 percentile-interval replicates, seed `20260503`. Identical to the two-judge configuration.
- Script: `scripts/three_judge_analysis.py`.

The two-judge cluster bootstrap (Codex+Claude) at v2026.05.07 reproduces byte-identically against the v2026.05.10 database when invoked with the same parameters. Only the new three-judge analysis is materially affected by the database refresh.

## Judge lanes included and excluded

Three production lanes are joined per response:

- Codex CLI: `judge_model = 'gpt-5.5-medium-codex-cli'`, `judge_prompt_version LIKE '%codex_gpt_5_5_medium_primary'`.
- Claude Code CLI: `judge_model = 'claude-opus-4-7-medium-claude-cli'`, `judge_prompt_version LIKE '%claude_opus_4_7_medium_comparison'`.
- DeepSeek V4 Flash via OpenCode CLI: `judge_model = 'deepseek-v4-flash-deepinfra-opencode-cli'`, `judge_prompt_version LIKE '%opencode_deepseek_v4_flash_comparison'`.

Both DeepSeek V4 Pro variants (`deepseek-v4-pro-deepinfra-opencode-cli` and `deepseek-v4-pro-opencode-cli`) are excluded by virtue of the V4 Flash–specific prompt-version filter; their evaluation rows remain in the database but do not contribute to this analysis.

The three-way join returns the same 9,618 paired rows the v2026.05.07 analysis used.

## Measurement: Inter-Judge Reliability under three lanes

Two-way mixed-effects, absolute-agreement intraclass correlation (McGraw and Wong 1996), with the third judge included.

| Statistic | v2026.05.07 (k=2) | v2026.05.10 (k=3) | Change |
| --- | ---: | ---: | ---: |
| ICC(A,1) | 0.9267 | 0.9073 | −0.0194 |
| ICC(A,k) | 0.9620 | 0.9671 | +0.0051 |

Pairwise ICC(A,1):

- Codex–Claude: 0.9267
- Codex–DeepSeek V4 Flash: 0.8965
- Claude–DeepSeek V4 Flash: 0.9000
- All three (k=3): 0.9073

The single-judge reliability drops slightly with the third lane because DeepSeek V4 Flash introduces an offset shift; the averaged-judge reliability rises slightly because averaging over three raters is more reliable than averaging over two. Both directions are consistent with classical psychometric theory under the addition of a moderately-correlated third rater.

## Measurement: Pairwise reliability (Bland-Altman, mean differences, correlations)

| Pair | mean signed diff | mean absolute diff | Pearson r | Spearman ρ | within ±0.50 | within ±1.00 | parametric LoA | empirical 2.5–97.5 percentile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Claude − Codex | +0.0271 | 0.2261 | 0.9312 | 0.9218 | 85.39% | 98.33% | [−0.6921, +0.7464] | [−0.70, +0.90] |
| DeepSeek − Codex | +0.1292 | 0.2623 | 0.9101 | 0.9013 | 81.41% | 96.29% | [−0.7030, +0.9614] | [−0.55, +1.20] |
| DeepSeek − Claude | +0.1020 | 0.2719 | 0.9047 | 0.8903 | 81.75% | 96.37% | [−0.7671, +0.9712] | [−0.70, +1.15] |

DeepSeek V4 Flash, when serving as judge, applies a systematic upward shift of approximately 0.10–0.13 on the 0–5 rubric relative to Codex and Claude. The shape of agreement (Pearson, Spearman, within-tolerance rates) is similar across pairs; the offset is the distinguishing feature.

## Measurement: Role-weighted composite under the three-judge mean

The role-audition weighted composite uses the same weights as the v2026.05.07 release:

`Composite_m = 0.08·D1_m + 0.17·D2_m + 0.175·D3_m + 0.07·D4_m + 0.14·D5_m + 0.145·D6_m + 0.22·D7_m`

The two-judge baseline averages over Codex and Claude scores; the three-judge composite averages over Codex, Claude, and DeepSeek V4 Flash. Both use the same paired rows.

Top role-weighted models, three-judge mean (descending):

| Rank | Model | 2-judge composite | 3-judge composite | Δ |
| ---: | --- | ---: | ---: | ---: |
| 1 | hermes-3-llama-3.1-405b | 2.7995 | 2.8483 | +0.0488 |
| 2 | llama4-maverick-17b-128e-moe | 2.5557 | 2.5781 | +0.0224 |
| 3 | llama-3.3-70b-instruct | 2.4952 | 2.5347 | +0.0395 |
| 4 | gemma-4-31b-it | 2.4566 | 2.4932 | +0.0366 |
| 5 | qwen3.5-397b-a17b-moe | 2.3978 | 2.3931 | −0.0047 |
| 6 | command-a-03-2025 | 2.3452 | 2.3915 | +0.0463 |
| 7 | glm-5.1 | 2.3681 | 2.3847 | +0.0166 |
| 8 | deepseek-v4-flash | 2.3830 | 2.3736 | −0.0094 |
| 9 | qwen3.5-122b-a10b | 2.2771 | 2.2874 | +0.0103 |
| 10 | qwen3.6-35b-a3b-q8_0 | 2.2626 | 2.2693 | +0.0067 |
| 11 | mistral-medium-3.5 | 2.2370 | 2.2604 | +0.0234 |
| 12 | qwen3.6-27b-fp8 | 2.2252 | 2.2574 | +0.0322 |
| 13 | llada2.1-flash | 2.0532 | 2.1053 | +0.0521 |
| 14 | qwen3.5-9b | 1.9451 | 1.9911 | +0.0460 |

The top four ranks are preserved in order. Mid-leaderboard reordering: command-a-03-2025 rises from rank 8 to 6 and deepseek-v4-flash drops from 6 to 8 as a candidate.

## Measurement: Llama-Gemma paired bootstrap under the three-judge mean

Cluster bootstrap (1000 percentile-interval replicates, seed 20260503), question-clustered, paired:

- Observed Δ (Llama 3.3 − Gemma 4, role-weighted): +0.0415
- Bootstrap mean Δ: +0.0431
- 95% percentile CI: [−0.0241, +0.1136]
- CI includes zero: True

The selection-bridge conclusion is unchanged from v2026.05.07: Llama 3.3 has a small numerical role-weighted advantage over Gemma 4 that does not survive the paired bootstrap, and non-score criteria continue to legitimately drive the Gemma selection.

## Diagnostic: DeepSeek-as-judge offset by candidate model

For each candidate model, the DeepSeek V4 Flash judge mean and the (Codex+Claude)/2 baseline mean were computed. The offset (DeepSeek minus baseline) probes whether DeepSeek V4 Flash exhibits self-preference bias toward its own family's candidate model (`deepseek-v4-flash`).

| Candidate model | DeepSeek-as-judge mean | (Codex+Claude)/2 mean | Offset (DS − CC/2) |
| --- | ---: | ---: | ---: |
| llada2.1-flash | 2.4191 | 2.2025 | +0.2166 |
| command-a-03-2025 | 2.7960 | 2.6213 | +0.1747 |
| qwen3.5-9b | 2.3122 | 2.1386 | +0.1735 |
| llama-3.3-70b-instruct | 2.8752 | 2.7342 | +0.1410 |
| hermes-3-llama-3.1-405b | 3.1476 | 3.0119 | +0.1357 |
| gemma-4-31b-it | 2.8897 | 2.7542 | +0.1356 |
| qwen3.6-27b-fp8 | 2.6598 | 2.5323 | +0.1274 |
| mistral-medium-3.5 | 2.6937 | 2.5826 | +0.1110 |
| glm-5.1 | 2.8148 | 2.7190 | +0.0957 |
| qwen3.5-122b-a10b | 2.7044 | 2.6140 | +0.0904 |
| qwen3.6-35b-a3b-q8_0 | 2.6243 | 2.5386 | +0.0857 |
| llama4-maverick-17b-128e-moe | 2.8772 | 2.8218 | +0.0555 |
| **deepseek-v4-flash** | 2.7822 | 2.7421 | **+0.0401** |
| qwen3.5-397b-a17b-moe | 2.8081 | 2.7725 | +0.0356 |

DeepSeek V4 Flash applies a positive offset to every candidate (every offset above zero), with a cohort mean of approximately +0.116. Its offset for its own family's candidate (`deepseek-v4-flash`) is the second-smallest in the cohort at +0.0401, larger than only `qwen3.5-397b-a17b-moe` at +0.0356. This is the inverse of self-preference bias as documented by Panickssery, Bowman, and Feng (2024); the same-family judge–candidate pair shows a smaller upward adjustment than nearly every cross-family pair, which is consistent with the absence of self-preference under the source-grounded, schema-constrained, agentic adjudication apparatus described in the paper. Further study with additional own-family judge–candidate pairs is warranted to confirm the effect's generality.

## References

Method citations match those of the v2026.05.07 methods document. The three-judge analysis additionally references:

- Panickssery, Arjun, Samuel R. Bowman, and Shi Feng. "LLM Evaluators Recognize and Favor Their Own Generations." *NeurIPS 2024*. arXiv:2404.13076.
- Wataoka, Koki, Tsubasa Takahashi, and Ryokan Ri. "Self-Preference Bias in LLM-as-a-Judge." Paper presented at the NeurIPS 2024 Safe Generative AI Workshop. arXiv:2410.21819.
