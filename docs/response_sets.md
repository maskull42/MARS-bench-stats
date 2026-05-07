# Response Set Accounting

This note explains why the public database has 10,542 release responses while
the paired model-selection analyses use 9,618 responses.

## Release Response Universe

The public DB contains:

- 251 release questions.
- 14 model cohorts.
- 3 response runs per model-question pair.
- 10,542 total release responses.

This follows:

```text
251 questions x 14 models x 3 runs = 10,542 responses
```

## Paired Model-Selection Cohort

The final paired Codex-Claude model-selection analysis uses:

- 229 questions.
- 14 model cohorts.
- 3 response runs per model-question pair.
- 9,618 paired response rows.

This follows:

```text
229 questions x 14 models x 3 runs = 9,618 responses
```

Each paired row has both:

- `gpt-5.5-medium-codex-cli` with prompt version matching
  `%codex_gpt_5_5_medium_primary`.
- `claude-opus-4-7-medium-claude-cli` with prompt version matching
  `%claude_opus_4_7_medium_comparison`.

## The 924 Nonpaired Responses

The difference is:

```text
10,542 - 9,618 = 924
924 = 22 questions x 14 models x 3 runs
```

Those 924 rows are not stray legacy responses. They are current-release D1
Ancient Languages structured morphology adjunct responses:

- Track: `philology_adjunct`.
- Tier: `D`.
- Authoring input type: `structured_key_spec`.
- Level: `3`.
- Question composition: 10 Greek New Testament, 5 Septuagint Greek, 5 Biblical
  Hebrew, and 2 Biblical Aramaic morphology questions.

They have evaluations from:

```text
d1-structured-morphology-scorer
d1-structured-morphology-scorer-2026-04-25
```

They have no final paired Codex-Claude evaluations and are therefore excluded
from paired model-selection statistics. They can be analyzed separately with a
morphology-specific metric, but should not be mixed into the paired
Codex-Claude composite without an explicit methodological bridge.

