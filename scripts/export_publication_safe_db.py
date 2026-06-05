#!/usr/bin/env python3
"""Build a publication-safe MARS-Bench statistical export database.

The source database is the private working SQLite database. The output database
is release-scoped and suitable for public statistical replication:

- includes benchmark questions, reference answers, model responses, rubrics,
  scores, judge notes, audit flags, and provenance hashes;
- excludes private user/account tables and worker claims;
- redacts local filesystem paths and raw long-form source/context packet body
  text while retaining hashes and structured provenance metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCE_DB = Path(os.environ.get("MARS_BENCH_SOURCE_DB", "data/source-private/mars_bench.db"))
DEFAULT_RELEASE_LABEL = "mars_bench_v2_0_d1_d2_d3_d4_d5_d6_d7_rebuild_candidate"
DEFAULT_OUTPUT_DB = Path("data/mars_bench_stats_public.sqlite")

LOCAL_PATH_RE = re.compile(r"/Users/[^\s\"',;)]+")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_unicode(value: str) -> str:
    return value.encode("utf-8", errors="replace").decode("utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(clean_unicode(value).encode("utf-8")).hexdigest()


def redact_local_paths(value: str) -> str:
    value = clean_unicode(value)

    def repl(match: re.Match[str]) -> str:
        raw = match.group(0)
        return f"[redacted-local-path:{Path(raw).name}]"

    return LOCAL_PATH_RE.sub(repl, value)


def maybe_json(value: str | None) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


STRICT_TEXT_KEYS = (
    "source_text",
    "passage_text",
    "text_content",
    "quote",
    "quotation",
    "snippet",
    "source_passage",
    "translated_text",
    "translation",
    "ocr_text",
    "body",
)

PATH_KEYS = (
    "path",
    "local_path",
    "workspace_path",
    "bundle_path",
    "manifest_path",
    "approval_decision_memo",
    "batch_manifest_path",
)


def sanitize_json(value: Any, *, strict_packet: bool = False, parent_key: str = "") -> Any:
    """Recursively sanitize JSON-like structures for public release.

    strict_packet=True is used for source/context packets, where raw source text
    and long passage bodies are not included in the public export.
    """
    key_l = parent_key.lower()
    if isinstance(value, dict):
        return {
            str(k): sanitize_json(v, strict_packet=strict_packet, parent_key=str(k))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [sanitize_json(v, strict_packet=strict_packet, parent_key=parent_key) for v in value]
    if isinstance(value, str):
        redacted = redact_local_paths(value)
        is_path_key = any(pattern in key_l for pattern in PATH_KEYS)
        if is_path_key:
            return {
                "redacted": True,
                "reason": "local filesystem path removed",
                "basename": Path(value).name if "/" in value else None,
                "sha256": sha256_text(value),
            }
        is_text_key = any(pattern in key_l for pattern in STRICT_TEXT_KEYS)
        if strict_packet and (is_text_key or len(redacted) > 1200):
            return {
                "redacted": True,
                "reason": "raw source/context body omitted from publication-safe export",
                "char_count": len(value),
                "sha256": sha256_text(value),
            }
        return redacted
    return value


def sanitize_json_text(value: str | None, *, strict_packet: bool = False) -> str | None:
    parsed = maybe_json(value)
    if parsed is None:
        return redact_local_paths(value) if value is not None else None
    return json.dumps(sanitize_json(parsed, strict_packet=strict_packet), ensure_ascii=False, sort_keys=True)


def redact_text(value: str | None) -> str | None:
    return redact_local_paths(value) if value is not None else None


def placeholders(values: Iterable[Any]) -> str:
    vals = list(values)
    if not vals:
        raise ValueError("cannot build placeholders for empty list")
    return ",".join("?" for _ in vals)


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys = OFF;

        CREATE TABLE publication_export_metadata (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );

        CREATE TABLE domains (
          id INTEGER PRIMARY KEY,
          name TEXT,
          display_name TEXT,
          description TEXT,
          priority INTEGER
        );

        CREATE TABLE question_categories (
          id INTEGER PRIMARY KEY,
          domain_id INTEGER,
          code TEXT,
          name TEXT,
          description TEXT
        );

        CREATE TABLE question_subcategories (
          id INTEGER PRIMARY KEY,
          category_id INTEGER,
          code TEXT,
          name TEXT,
          description TEXT
        );

        CREATE TABLE rubrics (
          id INTEGER PRIMARY KEY,
          domain_id INTEGER,
          version TEXT,
          dimensions_json TEXT,
          judge_instructions TEXT,
          persona_prompt TEXT,
          created_at TEXT,
          is_locked INTEGER
        );

        CREATE TABLE prompt_templates (
          id INTEGER PRIMARY KEY,
          domain_id INTEGER,
          name TEXT,
          version TEXT,
          template_text TEXT,
          description TEXT,
          created_at TEXT
        );

        CREATE TABLE benchmark_releases (
          id INTEGER PRIMARY KEY,
          release_label TEXT,
          base_release_id INTEGER,
          manifest_path_redacted TEXT,
          manifest_hash TEXT,
          generation_prompt_bundle_version TEXT,
          reference_authoring_bundle_version TEXT,
          judge_prompt_version TEXT,
          rubric_bundle_version TEXT,
          status TEXT,
          notes TEXT,
          created_at TEXT,
          frozen_at TEXT
        );

        CREATE TABLE release_questions (
          id INTEGER PRIMARY KEY,
          release_id INTEGER,
          question_id INTEGER,
          source_question_id INTEGER,
          parent_release_question_id INTEGER,
          lineage_action TEXT,
          track TEXT,
          tier TEXT,
          authoring_input_type TEXT,
          is_included INTEGER,
          inclusion_notes TEXT,
          manifest_section TEXT,
          sort_order INTEGER,
          created_at TEXT
        );

        CREATE TABLE questions (
          id INTEGER PRIMARY KEY,
          domain_id INTEGER,
          level INTEGER,
          text TEXT,
          ground_truth TEXT,
          expected_characteristics TEXT,
          rubric_id INTEGER,
          source_refs TEXT,
          grounding_type TEXT,
          is_trap INTEGER,
          created_at TEXT,
          created_by TEXT,
          is_supplementary INTEGER,
          category_id INTEGER,
          subcategory_id INTEGER,
          retired_at TEXT,
          retired_reason TEXT
        );

        CREATE TABLE models (
          id INTEGER PRIMARY KEY,
          name TEXT,
          family TEXT,
          version TEXT,
          size_b REAL,
          quantization TEXT,
          inference_engine TEXT,
          api_provider TEXT,
          is_open_weight INTEGER,
          is_ceiling_ref INTEGER,
          notes TEXT
        );

        CREATE TABLE benchmark_runs (
          id INTEGER PRIMARY KEY,
          name TEXT,
          description TEXT,
          model_id INTEGER,
          domain_id INTEGER,
          temperature REAL,
          top_p REAL,
          max_tokens INTEGER,
          prompt_template_version TEXT,
          judge_model TEXT,
          judge_prompt_version TEXT,
          started_at TEXT,
          completed_at TEXT,
          status TEXT,
          notes TEXT,
          random_seed INTEGER,
          benchmark_release_id INTEGER
        );

        CREATE TABLE reference_authoring_runs (
          id INTEGER PRIMARY KEY,
          release_id INTEGER,
          run_label TEXT,
          workspace_path_redacted TEXT,
          batch_manifest_path_redacted TEXT,
          batch_manifest_hash TEXT,
          author_model TEXT,
          status TEXT,
          started_at TEXT,
          completed_at TEXT,
          notes TEXT
        );

        CREATE TABLE question_reference_answer_versions (
          id INTEGER PRIMARY KEY,
          release_id INTEGER,
          question_id INTEGER,
          release_question_id INTEGER,
          reference_answer_version TEXT,
          reference_answer_text TEXT,
          expected_characteristics_json TEXT,
          uncertainty_notes TEXT,
          authoring_input_hash TEXT,
          authoring_run_id INTEGER,
          author_model TEXT,
          review_status TEXT,
          bundle_path_redacted TEXT,
          bundle_hash TEXT,
          is_frozen INTEGER,
          created_at TEXT,
          frozen_at TEXT
        );

        CREATE TABLE question_source_resolution (
          id INTEGER PRIMARY KEY,
          release_id INTEGER,
          question_id INTEGER,
          release_question_id INTEGER,
          resolution_version TEXT,
          source_refs_raw TEXT,
          candidate_sources_json TEXT,
          resolved_sources_json TEXT,
          unresolved_refs_json TEXT,
          epistemic_type TEXT,
          resolution_status TEXT,
          reviewer_notes TEXT,
          created_at TEXT,
          approved_at TEXT
        );

        CREATE TABLE question_corpus_context (
          id INTEGER PRIMARY KEY,
          release_id INTEGER,
          question_id INTEGER,
          release_question_id INTEGER,
          context_version TEXT,
          context_type TEXT,
          packet_json_public_redacted TEXT,
          input_hash TEXT,
          retrieval_notes TEXT,
          context_status TEXT,
          created_at TEXT,
          approved_at TEXT
        );

        CREATE TABLE question_grounding_audit (
          id INTEGER PRIMARY KEY,
          release_id INTEGER,
          question_id INTEGER,
          release_question_id INTEGER,
          audit_version TEXT,
          audit_type TEXT,
          audit_status TEXT,
          findings_json TEXT,
          reviewer TEXT,
          created_at TEXT,
          approved_at TEXT
        );

        CREATE TABLE responses (
          id INTEGER PRIMARY KEY,
          benchmark_run_id INTEGER,
          benchmark_release_id INTEGER,
          question_id INTEGER,
          model_id INTEGER,
          run_number INTEGER,
          response_text TEXT,
          latency_ms INTEGER,
          prompt_tokens INTEGER,
          completion_tokens INTEGER,
          total_tokens INTEGER,
          cost_usd REAL,
          temperature REAL,
          top_p REAL,
          max_tokens INTEGER,
          raw_metadata_redacted TEXT,
          created_at TEXT
        );

        CREATE TABLE evaluations (
          id INTEGER PRIMARY KEY,
          response_id INTEGER,
          judge_model TEXT,
          judge_prompt_version TEXT,
          rubric_id INTEGER,
          per_dimension_scores TEXT,
          overall_score REAL,
          overall_rationale TEXT,
          factual_errors TEXT,
          anachronisms TEXT,
          confusions TEXT,
          scholarly_credibility TEXT,
          error_flags TEXT,
          judge_latency_ms INTEGER,
          judge_tokens INTEGER,
          judge_cost_usd REAL,
          created_at TEXT,
          judge_run_number INTEGER,
          dimension_order_seed INTEGER,
          prompt_hash TEXT,
          trap_caps_applied TEXT,
          response_word_count INTEGER,
          benchmark_release_id INTEGER,
          release_question_id INTEGER,
          reference_answer_version TEXT,
          context_version TEXT,
          grounding_audit_version TEXT,
          evaluation_regime TEXT,
          materials_hash TEXT
        );

        CREATE TABLE response_grounding_audits (
          id INTEGER PRIMARY KEY,
          response_id INTEGER,
          benchmark_release_id INTEGER,
          release_question_id INTEGER,
          question_id INTEGER,
          task_type TEXT,
          stream TEXT,
          audit_version TEXT,
          audit_status TEXT,
          failure_status TEXT,
          claim_manifest_json TEXT,
          source_material_hashes_json TEXT,
          hallucination_flags_json TEXT,
          unsupported_but_noncentral_claims_json TEXT,
          fatal_audit_errors_json TEXT,
          materials_hash TEXT,
          created_at TEXT,
          approved_at TEXT
        );

        CREATE INDEX idx_pub_responses_release_model ON responses(benchmark_release_id, model_id);
        CREATE INDEX idx_pub_responses_question ON responses(question_id);
        CREATE INDEX idx_pub_evaluations_response ON evaluations(response_id);
        CREATE INDEX idx_pub_evaluations_release ON evaluations(benchmark_release_id);
        CREATE INDEX idx_pub_release_questions_release ON release_questions(release_id);

        CREATE VIEW paired_codex_claude_scores AS
        SELECT
          r.id AS response_id,
          m.name AS model,
          q.id AS question_id,
          q.domain_id AS domain_id,
          q.level AS level,
          q.is_trap AS is_trap,
          rq.authoring_input_type AS authoring_input_type,
          r.run_number AS run_number,
          r.response_text AS response_text,
          COALESCE(c.response_word_count, cl.response_word_count, length(r.response_text) - length(replace(r.response_text, ' ', '')) + 1) AS response_word_count,
          c.overall_score AS codex_score,
          cl.overall_score AS claude_score,
          (c.overall_score + cl.overall_score) / 2.0 AS paired_average_score,
          cl.overall_score - c.overall_score AS judge_difference_claude_minus_codex,
          c.error_flags AS codex_error_flags,
          cl.error_flags AS claude_error_flags,
          c.trap_caps_applied AS codex_trap_caps_applied,
          cl.trap_caps_applied AS claude_trap_caps_applied
        FROM responses r
        JOIN models m ON m.id = r.model_id
        JOIN questions q ON q.id = r.question_id
        JOIN release_questions rq ON rq.release_id = r.benchmark_release_id AND rq.question_id = q.id
        JOIN evaluations c
          ON c.response_id = r.id
         AND c.judge_model = 'gpt-5.5-medium-codex-cli'
         AND c.judge_prompt_version LIKE '%codex_gpt_5_5_medium_primary'
        JOIN evaluations cl
          ON cl.response_id = r.id
         AND cl.judge_model = 'claude-opus-4-7-medium-claude-cli'
         AND cl.judge_prompt_version LIKE '%claude_opus_4_7_medium_comparison';
        """
    )


def insert_many(conn: sqlite3.Connection, table: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
    conn.executemany(sql, [[row.get(col) for col in cols] for row in rows])


def fetch_dicts(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def transform_rows(rows: list[dict[str, Any]], transforms: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        for col, fn in transforms.items():
            if col in item:
                item[col] = fn(item[col])
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create publication-safe MARS-Bench stats DB.")
    parser.add_argument("--source-db", type=Path, default=DEFAULT_SOURCE_DB)
    parser.add_argument("--release-label", default=DEFAULT_RELEASE_LABEL)
    parser.add_argument("--output-db", type=Path, default=DEFAULT_OUTPUT_DB)
    args = parser.parse_args()

    source_db = args.source_db.resolve()
    output_db = args.output_db.resolve()
    if not source_db.exists():
        raise SystemExit(f"source DB not found: {source_db}")
    if output_db.exists():
        output_db.unlink()
    output_db.parent.mkdir(parents=True, exist_ok=True)

    src = connect(source_db)
    dst = sqlite3.connect(str(output_db))
    dst.row_factory = sqlite3.Row
    create_schema(dst)

    release = src.execute(
        "SELECT * FROM benchmark_releases WHERE release_label = ?",
        (args.release_label,),
    ).fetchone()
    if release is None:
        raise SystemExit(f"release not found: {args.release_label}")
    release_id = int(release["id"])

    rq_rows = fetch_dicts(src, "SELECT * FROM release_questions WHERE release_id = ? ORDER BY id", (release_id,))
    question_ids = sorted({int(row["question_id"]) for row in rq_rows})
    release_question_ids = sorted({int(row["id"]) for row in rq_rows})
    q_ph = placeholders(question_ids)

    response_rows = fetch_dicts(src, "SELECT * FROM responses WHERE benchmark_release_id = ? ORDER BY id", (release_id,))
    response_ids = sorted({int(row["id"]) for row in response_rows})
    response_ph = placeholders(response_ids)
    model_ids = sorted({int(row["model_id"]) for row in response_rows})
    model_ph = placeholders(model_ids)
    run_ids = sorted({int(row["benchmark_run_id"]) for row in response_rows})
    run_ph = placeholders(run_ids)

    eval_rows = fetch_dicts(src, "SELECT * FROM evaluations WHERE benchmark_release_id = ? ORDER BY id", (release_id,))
    rubric_ids = sorted({int(row["rubric_id"]) for row in eval_rows})
    rubric_ph = placeholders(rubric_ids)

    question_rows = fetch_dicts(src, f"SELECT * FROM questions WHERE id IN ({q_ph}) ORDER BY id", tuple(question_ids))
    category_ids = sorted({int(row["category_id"]) for row in question_rows if row.get("category_id") is not None})
    subcategory_ids = sorted({int(row["subcategory_id"]) for row in question_rows if row.get("subcategory_id") is not None})
    domain_ids = sorted({int(row["domain_id"]) for row in question_rows})
    domain_ph = placeholders(domain_ids)

    insert_many(dst, "domains", fetch_dicts(src, f"SELECT * FROM domains WHERE id IN ({domain_ph}) ORDER BY id", tuple(domain_ids)))
    if category_ids:
        insert_many(dst, "question_categories", fetch_dicts(src, f"SELECT * FROM question_categories WHERE id IN ({placeholders(category_ids)}) ORDER BY id", tuple(category_ids)))
    if subcategory_ids:
        insert_many(dst, "question_subcategories", fetch_dicts(src, f"SELECT * FROM question_subcategories WHERE id IN ({placeholders(subcategory_ids)}) ORDER BY id", tuple(subcategory_ids)))

    insert_many(dst, "rubrics", fetch_dicts(src, f"SELECT * FROM rubrics WHERE id IN ({rubric_ph}) ORDER BY id", tuple(rubric_ids)))
    insert_many(dst, "prompt_templates", transform_rows(
        fetch_dicts(src, "SELECT * FROM prompt_templates ORDER BY id"),
        {"template_text": redact_text, "description": redact_text},
    ))

    release_public = dict(release)
    release_public["manifest_path_redacted"] = redact_text(release_public.pop("manifest_path"))
    release_public["notes"] = redact_text(release_public["notes"])
    insert_many(dst, "benchmark_releases", [release_public])
    insert_many(dst, "release_questions", rq_rows)

    insert_many(dst, "questions", transform_rows(
        question_rows,
        {
            "expected_characteristics": lambda v: sanitize_json_text(v, strict_packet=False),
            "source_refs": lambda v: sanitize_json_text(v, strict_packet=True),
            "ground_truth": redact_text,
            "retired_reason": redact_text,
        },
    ))

    insert_many(dst, "models", transform_rows(
        fetch_dicts(src, f"SELECT * FROM models WHERE id IN ({model_ph}) ORDER BY id", tuple(model_ids)),
        {"notes": redact_text},
    ))

    insert_many(dst, "benchmark_runs", transform_rows(
        fetch_dicts(src, f"SELECT * FROM benchmark_runs WHERE id IN ({run_ph}) ORDER BY id", tuple(run_ids)),
        {"description": redact_text, "notes": redact_text},
    ))

    ref_runs = fetch_dicts(src, "SELECT * FROM reference_authoring_runs WHERE release_id = ? ORDER BY id", (release_id,))
    ref_runs_public = []
    for row in ref_runs:
        row["workspace_path_redacted"] = redact_text(row.pop("workspace_path"))
        row["batch_manifest_path_redacted"] = redact_text(row.pop("batch_manifest_path"))
        row["notes"] = redact_text(row["notes"])
        ref_runs_public.append(row)
    insert_many(dst, "reference_authoring_runs", ref_runs_public)

    ref_rows = fetch_dicts(src, "SELECT * FROM question_reference_answer_versions WHERE release_id = ? ORDER BY id", (release_id,))
    ref_public = []
    for row in ref_rows:
        row["bundle_path_redacted"] = redact_text(row.pop("bundle_path"))
        row["expected_characteristics_json"] = sanitize_json_text(row["expected_characteristics_json"])
        row["uncertainty_notes"] = redact_text(row["uncertainty_notes"])
        ref_public.append(row)
    insert_many(dst, "question_reference_answer_versions", ref_public)

    insert_many(dst, "question_source_resolution", transform_rows(
        fetch_dicts(src, "SELECT * FROM question_source_resolution WHERE release_id = ? ORDER BY id", (release_id,)),
        {
            "source_refs_raw": lambda v: sanitize_json_text(v, strict_packet=True),
            "candidate_sources_json": lambda v: sanitize_json_text(v, strict_packet=True),
            "resolved_sources_json": lambda v: sanitize_json_text(v, strict_packet=True),
            "unresolved_refs_json": lambda v: sanitize_json_text(v, strict_packet=True),
            "reviewer_notes": redact_text,
        },
    ))

    context_rows = fetch_dicts(src, "SELECT * FROM question_corpus_context WHERE release_id = ? ORDER BY id", (release_id,))
    context_public = []
    for row in context_rows:
        row["packet_json_public_redacted"] = sanitize_json_text(row.pop("packet_json"), strict_packet=True)
        row["retrieval_notes"] = redact_text(row["retrieval_notes"])
        context_public.append(row)
    insert_many(dst, "question_corpus_context", context_public)

    insert_many(dst, "question_grounding_audit", transform_rows(
        fetch_dicts(src, "SELECT * FROM question_grounding_audit WHERE release_id = ? ORDER BY id", (release_id,)),
        {"findings_json": lambda v: sanitize_json_text(v, strict_packet=False), "reviewer": redact_text},
    ))

    response_public = []
    for row in response_rows:
        row["raw_metadata_redacted"] = sanitize_json_text(row.pop("raw_metadata"), strict_packet=True)
        response_public.append(row)
    insert_many(dst, "responses", response_public)

    insert_many(dst, "evaluations", transform_rows(
        eval_rows,
        {
            "per_dimension_scores": lambda v: sanitize_json_text(v, strict_packet=False),
            "overall_rationale": redact_text,
            "factual_errors": lambda v: sanitize_json_text(v, strict_packet=False),
            "anachronisms": lambda v: sanitize_json_text(v, strict_packet=False),
            "confusions": lambda v: sanitize_json_text(v, strict_packet=False),
            "scholarly_credibility": redact_text,
            "error_flags": lambda v: sanitize_json_text(v, strict_packet=False),
            "trap_caps_applied": lambda v: sanitize_json_text(v, strict_packet=False),
        },
    ))

    insert_many(dst, "response_grounding_audits", transform_rows(
        fetch_dicts(src, "SELECT * FROM response_grounding_audits WHERE benchmark_release_id = ? ORDER BY id", (release_id,)),
        {
            "claim_manifest_json": lambda v: sanitize_json_text(v, strict_packet=False),
            "source_material_hashes_json": lambda v: sanitize_json_text(v, strict_packet=True),
            "hallucination_flags_json": lambda v: sanitize_json_text(v, strict_packet=False),
            "unsupported_but_noncentral_claims_json": lambda v: sanitize_json_text(v, strict_packet=False),
            "fatal_audit_errors_json": lambda v: sanitize_json_text(v, strict_packet=False),
        },
    ))

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_db_filename": source_db.name,
        "source_db_sha256": sha256_file(source_db),
        "release_label": args.release_label,
        "release_id": str(release_id),
        "publication_safety_policy": (
            "Release-scoped export. Private user/account tables, worker claims, and raw source-packet bodies "
            "are omitted. Local filesystem paths are redacted. Hashes and structured provenance are retained."
        ),
        "rows_responses": str(len(response_rows)),
        "rows_evaluations": str(len(eval_rows)),
        "rows_release_questions": str(len(rq_rows)),
        "rows_response_grounding_audits": str(src.execute("SELECT COUNT(*) FROM response_grounding_audits WHERE benchmark_release_id = ?", (release_id,)).fetchone()[0]),
    }
    dst.executemany(
        "INSERT INTO publication_export_metadata (key, value) VALUES (?, ?)",
        sorted(metadata.items()),
    )

    dst.commit()
    dst.execute("VACUUM")
    dst.close()
    src.close()

    # Keep a compressed copy for mirrors that prefer smaller artifacts.
    gz_path = output_db.with_suffix(output_db.suffix + ".gz")
    if gz_path.exists():
        gz_path.unlink()
    with output_db.open("rb") as src_handle, __import__("gzip").open(gz_path, "wb", compresslevel=9) as gz_handle:
        shutil.copyfileobj(src_handle, gz_handle)

    print(f"Wrote publication-safe DB: {output_db}")
    print(f"Wrote compressed DB:       {gz_path}")
    print(f"Output DB sha256:          {sha256_file(output_db)}")


if __name__ == "__main__":
    main()
