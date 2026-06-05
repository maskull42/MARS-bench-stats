# Parameter Scaling Diagnostics

Generated: `2026-06-05`

## Inputs

- Public DB: `data/mars_bench_stats_public.sqlite`
- Three-judge results: `results/three_judge/final_2026_06_05_three_judge_results.json`
- Parameter provenance: `results/diagnostics/model_parameter_counts_2026_06_05.json`
- Script: `scripts/parameter_scaling_analysis.py`

The analysis uses only the three production judge lanes: Codex/GPT
(`gpt-5.5-medium-codex-cli`), Claude Code
(`claude-opus-4-7-medium-claude-cli`), and DeepSeek V4 Flash
(`deepseek-v4-flash-*opencode-cli`). It joins the three-judge model scores to
`models.size_b` in the public DB.

`size_b` is interpreted as total parameters in billions. For sparse MoE models,
the report also computes a sensitivity check using active parameters where the
official source disclosed them.

## Headline Results

Across the 15-model public cohort, total parameter count has a moderate positive
relationship with three-judge performance:

| Target | Pearson r on log10(total params) | p | Spearman rho on total params | p |
| --- | ---: | ---: | ---: | ---: |
| All-domain mean | 0.626 | 0.0126 | 0.614 | 0.0148 |
| Role-weighted composite | 0.576 | 0.0245 | 0.604 | 0.0172 |

The active-parameter sensitivity check is weaker and not significant at this
sample size:

| Target | Pearson r on log10(active/dense params) | p | Spearman rho on active/dense params | p |
| --- | ---: | ---: | ---: | ---: |
| All-domain mean | 0.413 | 0.1263 | 0.345 | 0.2080 |
| Role-weighted composite | 0.411 | 0.1277 | 0.377 | 0.1658 |

This means there is a real size signal in this snapshot when total parameter
count is used, but it is not a simple "more active compute always wins" story.

## Domain Split

| Domain | Pearson r on log10(total params) | p | Spearman rho on total params | p |
| --- | ---: | ---: | ---: | ---: |
| D1 ancient_languages | 0.635 | 0.0109 | 0.529 | 0.0428 |
| D2 biblical_knowledge | 0.738 | 0.0017 | 0.818 | 0.0002 |
| D3 patristic_knowledge | 0.456 | 0.0878 | 0.400 | 0.1396 |
| D4 early_christian_history | 0.232 | 0.4047 | 0.225 | 0.4201 |
| D5 theological_knowledge | 0.058 | 0.8386 | 0.038 | 0.8944 |
| D6 heresiological_knowledge | 0.194 | 0.4891 | 0.275 | 0.3212 |
| D7 marcion_studies | 0.419 | 0.1196 | 0.436 | 0.1045 |

The clearest size relationship is in D2 biblical knowledge, followed by D1
ancient languages. D5 theological knowledge is essentially uncorrelated with
parameter count in this snapshot. D4 and D6 are also weak.

## Residuals

Against a linear fit of all-domain mean on `log10(size_b)`, the largest positive
residuals are:

| Model | Size B | Observed | Predicted | Residual |
| --- | ---: | ---: | ---: | ---: |
| Hermes405 | 405 | 2.917 | 2.595 | +0.321 |
| Gemma31 | 31 | 2.599 | 2.358 | +0.241 |
| Llama70 | 70 | 2.608 | 2.433 | +0.174 |
| Llama4 | 400 | 2.637 | 2.594 | +0.043 |
| Qwen27 | 27 | 2.384 | 2.345 | +0.039 |

The largest negative residuals are:

| Model | Size B | Observed | Predicted | Residual |
| --- | ---: | ---: | ---: | ---: |
| LLaDA100 | 100 | 2.192 | 2.466 | -0.274 |
| Qwen9 | 9 | 2.081 | 2.244 | -0.163 |
| GLM5.1 | 754 | 2.519 | 2.653 | -0.133 |
| Mistral128 | 128 | 2.382 | 2.489 | -0.107 |
| Qwen122 | 122 | 2.419 | 2.485 | -0.066 |

## Interpretation

What stands out is not that size is irrelevant; it is that size is domain
specific and model-family mediated. Larger total-parameter models tend to do
better overall, but the signal is concentrated in Bible/ancient-language
recall-heavy domains. The more synthetic theological and heresiological domains
do not reward size nearly as reliably.

Hermes 405B is the strongest overperformer relative to its size trend. Gemma 4
31B and Llama 3.3 70B also sit well above the regression line, which matches the
earlier leaderboard observation that they remain highly competitive despite much
smaller total parameter counts than the largest MoEs.

Conversely, GLM 5.1, Qwen 122B-A10B, Mistral Medium 3.5, and LLaDA2.1 Flash do
not convert their size into proportional MARS-Bench gains. LLaDA2.1 Flash is the
clearest negative residual in this cohort.

The active-parameter sensitivity check is important. Once sparse MoEs are
compared by activated parameters instead of total stored parameters, the
correlation drops and no longer reaches conventional significance. That argues
against treating total parameter count as a stand-alone proxy for expected
STA-relevant performance.
