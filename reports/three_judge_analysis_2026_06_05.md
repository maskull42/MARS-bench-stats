# MARS-Bench Three-Judge Re-Analysis (v2026.06.05)

Generated: `2026-06-05`. This document reports the three-judge re-analysis after adding the Gemma 4 12B A6000 benchmark run and its completed primary judging rows.

## Provenance

- Source database: `data/mars_bench_stats_public.sqlite` at the v2026.06.05 release (SHA-256 `9983b4223ef5bf000d5ad3f6b14c2968f6de6e2af749e29b7263978715554dda`).
- Three-judge paired-rows SHA-256: `018606274c259fe5496d4af17fe16feffd0db78639e0fbad632d17d92e5759af`.
- Bootstrap parameters: 1000 percentile-interval replicates, seed `20260503`.
- Script: `scripts/three_judge_analysis.py`.

The three-way join returns 10,305 paired rows: 229 paired-analysis questions x 15 candidate models x 3 runs.

## Judge Lanes Included And Excluded

Three production lanes are joined per response:

- Codex CLI: `judge_model = 'gpt-5.5-medium-codex-cli'`, `judge_prompt_version LIKE '%codex_gpt_5_5_medium_primary'`.
- Claude Code CLI: `judge_model = 'claude-opus-4-7-medium-claude-cli'`, `judge_prompt_version LIKE '%claude_opus_4_7_medium_comparison'`.
- DeepSeek V4 Flash via OpenCode CLI: DeepInfra for the original cohort (`deepseek-v4-flash-deepinfra-opencode-cli`) and OpenRouter for Gemma 4 12B (`deepseek-v4-flash-openrouter-opencode-cli`).

Both DeepSeek V4 Pro variants (`deepseek-v4-pro-deepinfra-opencode-cli` and `deepseek-v4-pro-opencode-cli`) are excluded by the V4 Flash-specific judge filters.

## Inter-Judge Reliability

Two-way mixed-effects, absolute-agreement intraclass correlation (McGraw and Wong 1996), with the third judge included:

| Statistic | Two judges (Codex-Claude) | Three judges | Change |
| --- | ---: | ---: | ---: |
| ICC(A,1) | 0.9262 | 0.8819 | -0.0443 |
| ICC(A,k) | 0.9617 | 0.9573 | -0.0044 |

Pairwise ICC(A,1):

- Codex-Claude: 0.9262
- Codex-DeepSeek V4 Flash: 0.8595
- Claude-DeepSeek V4 Flash: 0.8636
- All three (k=3): 0.8819

## Pairwise Reliability

| Pair | mean signed diff | mean absolute diff | Pearson r | Spearman rho | within +/-0.50 | within +/-1.00 | parametric LoA | empirical 2.5-97.5 percentile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Claude - Codex | +0.0259 | 0.2254 | 0.9306 | 0.9190 | 85.40% | 98.34% | [-0.6900, +0.7418] | [-0.72, +0.90] |
| DeepSeek - Codex | +0.1590 | 0.2962 | 0.8779 | 0.8681 | 79.51% | 94.68% | [-0.8064, +1.1245] | [-0.55, +1.40] |
| DeepSeek - Claude | +0.1331 | 0.3081 | 0.8715 | 0.8564 | 79.52% | 94.71% | [-0.8748, +1.1410] | [-0.75, +1.45] |

DeepSeek V4 Flash applies a larger upward shift after the Gemma 4 12B addition than in the v2026.05.10 three-judge snapshot, driven primarily by the Gemma 4 12B OpenRouter DeepSeek lane.

## Role-Weighted Composite

The role-audition weighted composite uses the same weights as the two-judge analysis:

`Composite_m = 0.08*D1_m + 0.17*D2_m + 0.175*D3_m + 0.07*D4_m + 0.14*D5_m + 0.145*D6_m + 0.22*D7_m`

Top role-weighted models under the three-judge mean:

| Rank | Model | 2-judge composite | 3-judge composite | Delta |
| ---: | --- | ---: | ---: | ---: |
| 1 | hermes-3-llama-3.1-405b | 2.7995 | 2.8483 | +0.0488 |
| 2 | llama4-maverick-17b-128e-moe | 2.5557 | 2.5781 | +0.0223 |
| 3 | llama-3.3-70b-instruct | 2.4952 | 2.5347 | +0.0395 |
| 4 | gemma-4-31b-it | 2.4566 | 2.4932 | +0.0366 |
| 5 | qwen3.5-397b-a17b-moe | 2.3978 | 2.3931 | -0.0047 |
| 6 | command-a-03-2025 | 2.3452 | 2.3915 | +0.0463 |
| 7 | glm-5.1 | 2.3681 | 2.3847 | +0.0166 |
| 8 | deepseek-v4-flash | 2.3830 | 2.3736 | -0.0094 |
| 9 | qwen3.5-122b-a10b | 2.2771 | 2.2874 | +0.0103 |
| 10 | qwen3.6-35b-a3b-q8_0 | 2.2626 | 2.2693 | +0.0067 |
| 11 | mistral-medium-3.5 | 2.2370 | 2.2604 | +0.0234 |
| 12 | qwen3.6-27b-fp8 | 2.2252 | 2.2574 | +0.0321 |
| 13 | gemma-4-12b-it-q8_0-llamacpp-a6000 | 1.9993 | 2.2267 | +0.2274 |
| 14 | llada2.1-flash | 2.0532 | 2.1053 | +0.0521 |
| 15 | qwen3.5-9b | 1.9451 | 1.9911 | +0.0460 |

The top four ranks are preserved in order. Gemma 4 12B ranks 14th on the two-judge composite and 13th on the three-judge composite.

## Llama-Gemma Paired Bootstrap

Cluster bootstrap (1000 percentile-interval replicates, seed `20260503`), question-clustered and paired:

- Observed Delta (Llama 3.3 - Gemma 4 31B, role-weighted): +0.0415
- Bootstrap mean Delta: +0.0431
- 95% percentile CI: [-0.0241, +0.1136]
- CI includes zero: True

The selection-bridge conclusion is unchanged: Llama 3.3 has a small numerical role-weighted advantage over Gemma 4 31B that does not survive the paired bootstrap.

## DeepSeek-As-Judge Offset

For each candidate model, the DeepSeek V4 Flash judge mean and the (Codex+Claude)/2 baseline mean were computed. The offset is DeepSeek minus the two-judge baseline.

| Candidate model | DeepSeek-as-judge mean | (Codex+Claude)/2 mean | Offset |
| --- | ---: | ---: | ---: |
| gemma-4-12b-it-q8_0-llamacpp-a6000 | 2.6475 | 2.0751 | +0.5724 |
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
| deepseek-v4-flash | 2.7822 | 2.7421 | +0.0401 |
| qwen3.5-397b-a17b-moe | 2.8081 | 2.7725 | +0.0356 |

The Gemma 4 12B offset is substantially larger than every other model's offset and should be interpreted as a provider-lane/judge-calibration diagnostic rather than as a general shift in the original DeepInfra DeepSeek Flash lane.

## References

Method citations match those of `reports/final_statistical_methods_and_findings.md`.
