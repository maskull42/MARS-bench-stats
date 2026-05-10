# Changelog

All notable changes to the MARS-Bench Statistical Replication Package are recorded in this file.

The format follows the conventions of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the package adheres to a date-based versioning scheme (`vYYYY.MM.DD`).

## [v2026.05.10] — 2026-05-10

### Added

- Third production judge lane: DeepSeek V4 Flash via the OpenCode agentic harness, identified in the `evaluations` table by `judge_model = 'deepseek-v4-flash-deepinfra-opencode-cli'` and `judge_prompt_version LIKE '%opencode_deepseek_v4_flash_comparison'`. The lane covers all 9,618 paired rows of the v2.0 release at full fidelity.
- `scripts/three_judge_analysis.py`: deterministic three-judge re-analysis script. Computes ICC(A,1), ICC(A,k=3), pairwise reliability for the three lane pairs (Codex–Claude, Codex–DeepSeek, Claude–DeepSeek), per-model role-weighted composites under the three-judge mean, paired Llama–Gemma cluster bootstrap, and a per-candidate-model offset diagnostic for self-preference probing.
- `results/three_judge/`: three-judge analysis outputs at the v2026.05.10 release tag.
- `reports/three_judge_analysis_2026_05_10.md`: methods-and-findings document for the three-judge re-analysis. Frozen at this release; sits alongside the v2026.05.07 methods document, which is unchanged.

### Changed

- `data/mars_bench_stats_public.sqlite` and `.sqlite.gz`: re-exported from the working database to include the DeepSeek V4 Flash lane and a small number of high-effort Codex/Claude rows that were excluded from the v2026.05.07 export. The two-judge cluster bootstrap continues to reproduce byte-identically against the new database; only the new three-judge analysis is materially affected by the change.
- `data/SHA256SUMS`: updated to reflect the new public database fingerprints and the three-judge result hashes; v2026.05.07 result hashes (cluster bootstrap, diagnostics, methods report) are preserved unchanged.
- `README.md`: brief mention of the third judge and pointer to the three-judge re-analysis script and report.

### Unchanged (frozen at v2026.05.07)

- `results/cluster_bootstrap/final_2026_05_07_*`
- `results/diagnostics/*` (all v2026.05.07 diagnostic outputs)
- `reports/final_statistical_methods_and_findings.md`
- `scripts/cluster_bootstrap_mars_results.py`
- `scripts/run_bias_and_selection_diagnostics.py`

### Notes for citation

The accompanying paper, "As If, Not As Way: MARS-Bench and the Agentic Audition of LLMs for Marcionite Surrogation," cites the v2026.05.07 release at Zenodo DOI [10.5281/zenodo.20067936](https://doi.org/10.5281/zenodo.20067936). That citation does not change. The v2026.05.10 release is intended for citation in materials produced after the paper was finalized (presentation slides, follow-up work, audit notes); it will be assigned its own Zenodo DOI on release.

## [v2026.05.07] — 2026-05-07

### Initial public statistical replication package for the MARS-Bench v2.0 candidate release

- Two production judge lanes: Codex (`gpt-5.5-medium-codex-cli`) and Claude Code (`claude-opus-4-7-medium-claude-cli`), each covering 9,618 paired rows.
- `scripts/cluster_bootstrap_mars_results.py`: paired Codex–Claude cluster-aware bootstrap analysis (1000 percentile-interval replicates, seed 20260503).
- `scripts/run_bias_and_selection_diagnostics.py`: model-selection, bias, stability, distribution, and appendix diagnostics.
- `data/mars_bench_stats_public.sqlite.gz`: publication-safe database export.
- `reports/final_statistical_methods_and_findings.md`: methods and headline findings.
- This release is the citation anchor for the published paper.
