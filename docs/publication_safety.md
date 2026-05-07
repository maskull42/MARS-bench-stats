# Publication Safety Notes

This repository is a release-scoped public export. It is not a dump of the
private judging database.

## Included

- Benchmark domains, categories, subcategories, rubrics, and prompt templates.
- Release membership metadata for
  `mars_bench_v2_0_d1_d2_d3_d4_d5_d6_d7_rebuild_candidate`.
- Question text, reference answers, and reference-answer provenance needed for
  statistical replication.
- Model registry rows for models represented in the release.
- Model responses for the release.
- Codex and Claude final judging rows used by the paired statistical analyses,
  including scores, judge notes, audit notes, and flags.
- D1 structured morphology scorer rows and retained supplementary/earlier
  evaluation rows for auditability. These rows are not used by the paired
  model-selection scripts unless explicitly selected by a future analysis.
- Grounding and response-integrity audit outputs for the 9,618 paired-analysis
  responses.
- Hashes for source materials and provenance packets where raw bodies are not
  included.

## Redacted or Omitted

- Private user/account tables.
- Worker-claim state used during live judging.
- Machine-local filesystem paths.
- Raw long-form source/context packet bodies and long source snippets.
- Any local source database path; only the source filename and SHA-256 hash are
  retained in `publication_export_metadata`.

## Verification Performed

The export was regenerated after sanitizing model notes and script defaults.
The uncompressed SQLite artifact was scanned for machine-local path strings
such as `/Users/`, known local workspace patterns, and private adjacent
repository names. No such strings remained in the SQLite artifact after the
final export.
Structured lexical, syntactic, and provenance metadata remains where needed for
replication; this is distinct from raw long-form source-packet body text.

The only `/Users/` text intentionally present in the repository is the regular
expression inside `scripts/export_publication_safe_db.py` that detects and
redacts local filesystem paths.
