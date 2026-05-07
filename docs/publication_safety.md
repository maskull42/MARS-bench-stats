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
- Codex and Claude final judging rows, including scores, judge notes, audit
  notes, and flags used by the statistical analyses.
- Grounding and response-integrity audit outputs for release responses.
- Hashes for source materials and provenance packets where raw bodies are not
  included.

## Redacted or Omitted

- Private user/account tables.
- Worker-claim state used during live judging.
- Machine-local filesystem paths.
- Raw source/context packet bodies and long source snippets.
- Any local source database path; only the source filename and SHA-256 hash are
  retained in `publication_export_metadata`.

## Verification Performed

The export was regenerated after sanitizing model notes and script defaults.
The uncompressed SQLite artifact was scanned for machine-local path strings
such as `/Users/`, project desktop paths, and private adjacent repository
names. No such strings remained in the SQLite artifact after the final export.

The only `/Users/` text intentionally present in the repository is the regular
expression inside `scripts/export_publication_safe_db.py` that detects and
redacts local filesystem paths.

