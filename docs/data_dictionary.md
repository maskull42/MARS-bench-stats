# Public SQLite Data Dictionary

The public database is `data/mars_bench_stats_public.sqlite` after
decompressing `data/mars_bench_stats_public.sqlite.gz`.

## Tables

- `publication_export_metadata`: export timestamp, release label, row counts,
  source database filename, source database hash, and publication-safety policy.
- `domains`: seven benchmark domains.
- `question_categories`: release-relevant question categories.
- `question_subcategories`: release-relevant question subcategories.
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
- `question_source_resolution`: structured source-resolution metadata and
  provenance for questions.
- `question_corpus_context`: source/context metadata with raw packet bodies
  redacted and hashed.
- `question_grounding_audit`: question-level grounding audit notes and flags.
- `responses`: release model responses, run numbers, response text, response
  hashes, and response-integrity flags.
- `evaluations`: final judge scores, judge models, judge prompt versions,
  score rationales, notes, and judge flags.
- `response_grounding_audits`: response-level grounding, hallucination,
  unsupported-claim, and fatal-error audit outputs.

## View

- `paired_codex_claude_scores`: convenience view joining Codex and Claude final
  judgment rows on `response_id`. This is the main input used by the final
  statistical scripts. It exposes the paired average score, individual judge
  scores, question metadata, model name, response text, and word count.

