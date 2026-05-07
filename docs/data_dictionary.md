# Public SQLite Data Dictionary

The public database is `data/mars_bench_stats_public.sqlite` after
decompressing `data/mars_bench_stats_public.sqlite.gz`.

## Tables

- `publication_export_metadata`: export timestamp, release label, row counts,
  source database filename, source database hash, and publication-safety policy.
- `domains`: seven benchmark domains.
- `question_categories`: release-relevant question categories.
- `question_subcategories`: present for schema compatibility; this export has
  zero rows in this table.
- `rubrics`: scoring rubrics used by the judging apparatus.
- `prompt_templates`: public prompt and judging-template materials included in
  the replication package.
- `benchmark_releases`: release-level metadata with local manifest paths
  redacted.
- `release_questions`: membership of questions in the benchmark release.
- `questions`: question text, level, trap flag, grounding type, reference fields,
  and rubric linkage.
- `models`: model registry rows for release responses, with notes sanitized.
- `benchmark_runs`: model generation run metadata for release responses.
- `reference_authoring_runs`: reference-answer authoring run metadata with local
  paths redacted.
- `question_reference_answer_versions`: reference answers, answer hashes,
  confidence metadata, and source notes.
- `question_source_resolution`: present for schema compatibility; this export
  has zero rows in this table.
- `question_corpus_context`: source/context metadata with raw long-form packet
  bodies redacted and hashed. Structured lexical, syntactic, and provenance
  metadata may remain where needed for replication.
- `question_grounding_audit`: question-level grounding audit notes and flags.
- `responses`: release model responses, run numbers, response text, response
  hashes, generation metadata, and redacted raw response metadata.
- `evaluations`: all exported evaluation rows for release responses, including
  final paired Codex-Claude rows, the D1 structured morphology scorer rows, and
  retained supplementary/earlier judging rows. Final model-selection analyses
  filter this table to `gpt-5.5-medium-codex-cli` rows matching
  `%codex_gpt_5_5_medium_primary` and `claude-opus-4-7-medium-claude-cli` rows
  matching `%claude_opus_4_7_medium_comparison`.
- `response_grounding_audits`: response-level grounding, hallucination,
  unsupported-claim, and fatal-error audit outputs for the 9,618 paired-analysis
  responses.

## View

- `paired_codex_claude_scores`: convenience view joining Codex and Claude final
  judgment rows on `response_id`. This is the main input used by the final
  statistical scripts. It exposes the paired average score, individual judge
  scores, question metadata, model name, response text, and word count.

## Response Sets

The public DB contains 251 release questions and 10,542 release responses.
The paired Codex-Claude model-selection analysis uses 229 questions and 9,618
responses. The 924 responses outside the paired analysis are current-release D1
structured morphology adjunct responses: 22 questions x 14 models x 3 runs.
They were scored by `d1-structured-morphology-scorer-2026-04-25` and have no
final paired Codex-Claude evaluations.
