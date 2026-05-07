#!/usr/bin/env python3
"""
Build a release-aware v2.0 statistical snapshot for completed model cohorts.

This is not the full dissertation-grade statistical rebuild yet.
It is the first safe scaffold:
- release-aware
- v2-only
- judge-run aware
- completed-cohort only

Outputs:
- JSON snapshot
- markdown leaderboard/report
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = Path(os.environ.get("MARS_BENCH_DB", "data/mars_bench_stats_public.sqlite"))
DEFAULT_RELEASE_LABEL = "mars_bench_v2_0"
EXPECTED_RESPONSES_PER_MODEL = 627


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def get_release_id(conn: sqlite3.Connection, release_label: str) -> int:
    row = conn.execute(
        "SELECT id FROM benchmark_releases WHERE release_label = ?",
        (release_label,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"release not found: {release_label}")
    return int(row["id"])


def get_completed_models(conn: sqlite3.Connection, release_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        WITH response_counts AS (
          SELECT r.model_id, COUNT(*) AS response_count
          FROM responses r
          JOIN benchmark_runs br ON br.id = r.benchmark_run_id
          WHERE br.benchmark_release_id = ?
          GROUP BY r.model_id
        ),
        evaluation_counts AS (
          SELECT r.model_id, COUNT(*) AS evaluation_count
          FROM evaluations e
          JOIN responses r ON r.id = e.response_id
          WHERE e.benchmark_release_id = ?
            AND e.judge_prompt_version LIKE 'v2.%'
            AND e.judge_run_number = 1
          GROUP BY r.model_id
        )
        SELECT m.id AS model_id, m.name AS model_name
        FROM models m
        JOIN response_counts rc ON rc.model_id = m.id
        JOIN evaluation_counts ec ON ec.model_id = m.id
        WHERE rc.response_count = ? AND ec.evaluation_count = ?
        ORDER BY m.name
        """,
        (release_id, release_id, EXPECTED_RESPONSES_PER_MODEL, EXPECTED_RESPONSES_PER_MODEL),
    ).fetchall()


def build_snapshot(conn: sqlite3.Connection, release_id: int, release_label: str) -> dict:
    completed_models = get_completed_models(conn, release_id)
    model_names = [row["model_name"] for row in completed_models]

    placeholders = ",".join("?" for _ in model_names)
    if not placeholders:
        raise SystemExit("no completed models found for snapshot")

    overall_rows = conn.execute(
        f"""
        SELECT
          m.name AS model_name,
          COUNT(*) AS evaluation_count,
          ROUND(AVG(e.overall_score), 3) AS overall_avg
        FROM evaluations e
        JOIN responses r ON r.id = e.response_id
        JOIN models m ON m.id = r.model_id
        WHERE e.benchmark_release_id = ?
          AND e.judge_prompt_version LIKE 'v2.%'
          AND e.judge_run_number = 1
          AND m.name IN ({placeholders})
        GROUP BY m.name
        ORDER BY overall_avg DESC, m.name
        """,
        (release_id, *model_names),
    ).fetchall()

    regime_rows = conn.execute(
        f"""
        SELECT
          m.name AS model_name,
          e.evaluation_regime,
          COUNT(*) AS evaluation_count,
          ROUND(AVG(e.overall_score), 3) AS regime_avg
        FROM evaluations e
        JOIN responses r ON r.id = e.response_id
        JOIN models m ON m.id = r.model_id
        WHERE e.benchmark_release_id = ?
          AND e.judge_prompt_version LIKE 'v2.%'
          AND e.judge_run_number = 1
          AND m.name IN ({placeholders})
        GROUP BY m.name, e.evaluation_regime
        ORDER BY m.name, e.evaluation_regime
        """,
        (release_id, *model_names),
    ).fetchall()

    domain_rows = conn.execute(
        f"""
        SELECT
          m.name AS model_name,
          d.name AS domain_name,
          COUNT(*) AS evaluation_count,
          ROUND(AVG(e.overall_score), 3) AS domain_avg
        FROM evaluations e
        JOIN responses r ON r.id = e.response_id
        JOIN models m ON m.id = r.model_id
        JOIN questions q ON q.id = r.question_id
        JOIN domains d ON d.id = q.domain_id
        WHERE e.benchmark_release_id = ?
          AND e.judge_prompt_version LIKE 'v2.%'
          AND e.judge_run_number = 1
          AND m.name IN ({placeholders})
        GROUP BY m.name, d.name
        ORDER BY m.name, d.name
        """,
        (release_id, *model_names),
    ).fetchall()

    regimes_by_model: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in regime_rows:
        regimes_by_model[row["model_name"]][row["evaluation_regime"]] = {
            "evaluation_count": row["evaluation_count"],
            "avg_score": row["regime_avg"],
        }

    domains_by_model: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in domain_rows:
        domains_by_model[row["model_name"]][row["domain_name"]] = {
            "evaluation_count": row["evaluation_count"],
            "avg_score": row["domain_avg"],
        }

    leaderboard = []
    for rank, row in enumerate(overall_rows, start=1):
        leaderboard.append(
            {
                "rank": rank,
                "model_name": row["model_name"],
                "evaluation_count": row["evaluation_count"],
                "overall_avg": row["overall_avg"],
                "regimes": regimes_by_model[row["model_name"]],
                "domains": domains_by_model[row["model_name"]],
            }
        )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "release_label": release_label,
        "benchmark_release_id": release_id,
        "completed_model_count": len(leaderboard),
        "expected_responses_per_model": EXPECTED_RESPONSES_PER_MODEL,
        "leaderboard": leaderboard,
    }


def write_markdown(snapshot: dict, output_path: Path) -> None:
    lines = [
        "# MARS-Bench v2.0 Release-Aware Stats Snapshot",
        "",
        f"- generated_at: `{snapshot['generated_at']}`",
        f"- release_label: `{snapshot['release_label']}`",
        f"- completed_model_count: `{snapshot['completed_model_count']}`",
        "",
        "## Overall Leaderboard",
        "",
        "| Rank | Model | Avg | Evaluations |",
        "|---|---|---:|---:|",
    ]
    for row in snapshot["leaderboard"]:
        lines.append(
            f"| {row['rank']} | `{row['model_name']}` | `{row['overall_avg']}` | `{row['evaluation_count']}` |"
        )

    lines.extend(
        [
            "",
            "## Regime Breakdown",
            "",
            "| Model | Grounded | Ungrounded | Structured |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in snapshot["leaderboard"]:
        regimes = row["regimes"]
        grounded = regimes.get("grounded_source_packet", {}).get("avg_score", "—")
        ungrounded = regimes.get("ungrounded_brief", {}).get("avg_score", "—")
        structured = regimes.get("structured_key_spec", {}).get("avg_score", "—")
        lines.append(
            f"| `{row['model_name']}` | `{grounded}` | `{ungrounded}` | `{structured}` |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a release-aware v2.0 leaderboard snapshot.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--release-label", default=DEFAULT_RELEASE_LABEL)
    args = parser.parse_args()

    db_path = Path(args.db_path).resolve()
    conn = connect(db_path)
    try:
        release_id = get_release_id(conn, args.release_label)
        snapshot = build_snapshot(conn, release_id, args.release_label)
    finally:
        conn.close()

    stamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    json_path = ROOT / "documentation" / "analysis" / f"{args.release_label}_stats_snapshot_{stamp}.json"
    md_path = ROOT / "documentation" / "analysis" / f"{args.release_label}_stats_snapshot_{stamp}.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(snapshot, md_path)

    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
