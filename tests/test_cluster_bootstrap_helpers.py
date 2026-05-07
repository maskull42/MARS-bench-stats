"""Tests for scripts/cluster_bootstrap_mars_results.py.

Covers:
  - icc_absolute_agreement: hand-derived golden case
  - average_ranks: tie-handling
  - percentile: boundary + interpolation
  - run_bootstrap: determinism under fixed seed
  - load_paired_rows: SQL-pairing happy path AND uniqueness guard

The ICC golden case below is reproducible by hand from the McGraw/Wong
absolute-agreement formulas. See the docstring of test_icc_golden for the
derivation. If that test fails, either the formula in
icc_absolute_agreement was edited or the call site changed.
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from argparse import Namespace
from pathlib import Path

import pytest

# Load the script as a module without requiring it to be a package.
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "cluster_bootstrap_mars_results.py"
_spec = importlib.util.spec_from_file_location("cluster_bootstrap", _SCRIPT)
cluster_bootstrap = importlib.util.module_from_spec(_spec)
sys.modules["cluster_bootstrap"] = cluster_bootstrap
_spec.loader.exec_module(cluster_bootstrap)


# -----------------------------------------------------------------------------
# percentile
# -----------------------------------------------------------------------------

def test_percentile_empty_returns_none():
    assert cluster_bootstrap.percentile([], 50) is None


def test_percentile_single_value():
    assert cluster_bootstrap.percentile([3.7], 50) == 3.7
    assert cluster_bootstrap.percentile([3.7], 0) == 3.7
    assert cluster_bootstrap.percentile([3.7], 100) == 3.7


def test_percentile_boundaries_odd_length():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert cluster_bootstrap.percentile(xs, 0) == 1.0
    assert cluster_bootstrap.percentile(xs, 100) == 5.0
    assert cluster_bootstrap.percentile(xs, 50) == 3.0  # exact median


def test_percentile_linear_interpolation_even_length():
    xs = [1.0, 2.0, 3.0, 4.0]
    # median of 4 values is interpolated halfway between 2nd and 3rd ordered
    assert cluster_bootstrap.percentile(xs, 50) == 2.5
    # pct=25: position = (4-1) * 0.25 = 0.75 -> lerp(1.0, 2.0, 0.75) = 1.75
    assert cluster_bootstrap.percentile(xs, 25) == 1.75
    # pct=75: position = 3 * 0.75 = 2.25 -> lerp(3.0, 4.0, 0.25) = 3.25
    assert cluster_bootstrap.percentile(xs, 75) == 3.25


# -----------------------------------------------------------------------------
# average_ranks
# -----------------------------------------------------------------------------

def test_average_ranks_no_ties():
    # Sorted-position ranks: 1->1, 3->3, 2->2
    assert cluster_bootstrap.average_ranks([1.0, 3.0, 2.0]) == [1.0, 3.0, 2.0]


def test_average_ranks_all_tied():
    # n=3, all equal -> all rank (1+2+3)/3 = 2.0
    assert cluster_bootstrap.average_ranks([5.0, 5.0, 5.0]) == [2.0, 2.0, 2.0]


def test_average_ranks_partial_tie():
    # Values 1, 2, 2, 4 -> ranks 1, 2.5, 2.5, 4
    assert cluster_bootstrap.average_ranks([1.0, 2.0, 2.0, 4.0]) == [1.0, 2.5, 2.5, 4.0]


def test_average_ranks_tie_at_end():
    # 5, 1, 5, 3 -> sorted positions: 1->1, 3->2, 5->avg(3,4)=3.5, 5->3.5
    assert cluster_bootstrap.average_ranks([5.0, 1.0, 5.0, 3.0]) == [3.5, 1.0, 3.5, 2.0]


def test_average_ranks_single_element():
    assert cluster_bootstrap.average_ranks([42.0]) == [1.0]


# -----------------------------------------------------------------------------
# icc_absolute_agreement — hand-derived golden case
# -----------------------------------------------------------------------------

def test_icc_golden():
    """Hand-derivation:

    Pairs (codex, claude): (1,2), (3,2), (2,4), (4,4); n=4, k=2.
    Row means: 1.5, 2.5, 3.0, 4.0; col means: 2.5, 3.0; grand: 2.75.
    SS_rows = 2 * sum((rm - 2.75)^2) = 2 * 3.25 = 6.5.
    SS_cols = 4 * sum((cm - 2.75)^2) = 4 * 0.125 = 0.5.
    SS_err  = sum_{ij}(x_ij - rm_i - cm_j + 2.75)^2 = 2.5.

    MS_rows = 6.5/3 = 13/6.
    MS_cols = 0.5/1 = 1/2.
    MS_err  = 2.5/3 = 5/6.

    ICC(A,1) = (MS_rows - MS_err) / (MS_rows + (k-1)*MS_err + (k/n)*(MS_cols - MS_err))
             = (4/3) / (3 + (1/2)*(-1/3))
             = (4/3) / (17/6)
             = 8/17
             ~= 0.47058823529411764

    ICC(A,2) = (MS_rows - MS_err) / (MS_rows + (MS_cols - MS_err)/n)
             = (4/3) / (13/6 - 1/12)
             = (4/3) / (25/12)
             = 16/25
             = 0.64 (exact)
    """
    pairs = [(1.0, 2.0), (3.0, 2.0), (2.0, 4.0), (4.0, 4.0)]
    icc = cluster_bootstrap.icc_absolute_agreement(pairs)
    assert icc["icc_a1"] == pytest.approx(8 / 17, abs=1e-12)
    assert icc["icc_a2"] == pytest.approx(0.64, abs=1e-12)


def test_icc_perfect_agreement():
    # Identical scores -> SS_err = 0, ICC = 1.0 for both lanes.
    pairs = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
    icc = cluster_bootstrap.icc_absolute_agreement(pairs)
    assert icc["icc_a1"] == pytest.approx(1.0, abs=1e-12)
    assert icc["icc_a2"] == pytest.approx(1.0, abs=1e-12)


def test_icc_too_few_subjects_returns_none():
    icc = cluster_bootstrap.icc_absolute_agreement([(1.0, 2.0)])
    assert icc == {"icc_a1": None, "icc_a2": None}


# -----------------------------------------------------------------------------
# load_paired_rows — SQL fixture (happy path + uniqueness guard)
# -----------------------------------------------------------------------------

def _build_fixture_db(path: Path) -> sqlite3.Connection:
    """Create a minimal SQLite DB with the columns load_paired_rows joins on."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE benchmark_releases (
            id INTEGER PRIMARY KEY,
            release_label TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE domains (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            display_name TEXT NOT NULL
        );
        CREATE TABLE questions (
            id INTEGER PRIMARY KEY,
            domain_id INTEGER NOT NULL,
            is_trap INTEGER DEFAULT 0,
            retired_at TEXT
        );
        CREATE TABLE models (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            family TEXT
        );
        CREATE TABLE responses (
            id INTEGER PRIMARY KEY,
            model_id INTEGER NOT NULL,
            question_id INTEGER NOT NULL,
            run_number INTEGER DEFAULT 1,
            benchmark_release_id INTEGER NOT NULL
        );
        CREATE TABLE evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            response_id INTEGER NOT NULL,
            judge_model TEXT NOT NULL,
            judge_prompt_version TEXT NOT NULL,
            overall_score REAL NOT NULL
        );
        """
    )
    return conn


def _populate_fixture(conn: sqlite3.Connection, *, with_duplicate: bool = False) -> None:
    cur = conn.cursor()
    cur.execute("INSERT INTO benchmark_releases VALUES (1, 'test_release', 'candidate', '2026-05-03')")
    cur.execute("INSERT INTO domains VALUES (1, 'test_domain', 'Test Domain')")
    cur.executemany(
        "INSERT INTO questions (id, domain_id, is_trap, retired_at) VALUES (?, ?, 0, NULL)",
        [(101, 1), (102, 1), (103, 1)],
    )
    cur.execute("INSERT INTO models VALUES (1, 'model-A', 'fam')")
    cur.executemany(
        "INSERT INTO responses (id, model_id, question_id, run_number, benchmark_release_id) VALUES (?, 1, ?, 1, 1)",
        [(201, 101), (202, 102), (203, 103)],
    )
    codex_label = "test-grounded-v2026_05_03__codex_gpt_5_5_medium_primary"
    claude_label = "test-grounded-v2026_05_03__claude_opus_4_7_medium_comparison"
    eval_rows = [
        (201, "gpt-5.5-medium-codex-cli", codex_label, 3.5),
        (201, "claude-opus-4-7-medium-claude-cli", claude_label, 3.0),
        (202, "gpt-5.5-medium-codex-cli", codex_label, 4.0),
        (202, "claude-opus-4-7-medium-claude-cli", claude_label, 4.5),
        (203, "gpt-5.5-medium-codex-cli", codex_label, 2.5),
        (203, "claude-opus-4-7-medium-claude-cli", claude_label, 2.5),
    ]
    if with_duplicate:
        # Inject a second Codex evaluation for response 201 with the SAME
        # version string (realistic re-evaluation pass). Both rows match the
        # LIKE filter, so the self-join produces a Cartesian cross-product
        # for response 201 — exactly the Cartesian-product trigger the
        # load_paired_rows guard is supposed to catch.
        eval_rows.append(
            (201, "gpt-5.5-medium-codex-cli", codex_label, 3.7)
        )
    cur.executemany(
        "INSERT INTO evaluations (response_id, judge_model, judge_prompt_version, overall_score) VALUES (?, ?, ?, ?)",
        eval_rows,
    )
    conn.commit()


def _args_for_load() -> Namespace:
    return Namespace(
        codex_model="gpt-5.5-medium-codex-cli",
        claude_model="claude-opus-4-7-medium-claude-cli",
        codex_version_like="%codex_gpt_5_5_medium_primary",
        claude_version_like="%claude_opus_4_7_medium_comparison",
    )


def test_load_paired_rows_happy_path(tmp_path: Path):
    db_path = tmp_path / "fixture.db"
    conn = _build_fixture_db(db_path)
    _populate_fixture(conn, with_duplicate=False)
    rows = cluster_bootstrap.load_paired_rows(conn, release_id=1, args=_args_for_load())
    assert len(rows) == 3, f"expected 3 paired rows, got {len(rows)}"
    response_ids = sorted(r["response_id"] for r in rows)
    assert response_ids == [201, 202, 203]
    # Derived fields should be present and correct for response 201
    r201 = next(r for r in rows if r["response_id"] == 201)
    assert r201["paired_average_score"] == pytest.approx(3.25)
    assert r201["judge_difference_claude_minus_codex"] == pytest.approx(-0.5)


def test_load_paired_rows_duplicate_match_fires_guard(tmp_path: Path):
    """If two Codex evaluations match the LIKE suffix, the SQL produces a
    Cartesian cross-product; the guard added in load_paired_rows must
    raise SystemExit rather than emit silently inflated paired counts."""
    db_path = tmp_path / "fixture_dup.db"
    conn = _build_fixture_db(db_path)
    _populate_fixture(conn, with_duplicate=True)
    with pytest.raises(SystemExit) as excinfo:
        cluster_bootstrap.load_paired_rows(conn, release_id=1, args=_args_for_load())
    msg = str(excinfo.value)
    assert "Cartesian" in msg or "distinct response_id count" in msg


def test_load_paired_rows_respects_retired_at(tmp_path: Path):
    db_path = tmp_path / "fixture_retired.db"
    conn = _build_fixture_db(db_path)
    _populate_fixture(conn, with_duplicate=False)
    conn.execute("UPDATE questions SET retired_at = '2026-05-01' WHERE id = 102")
    conn.commit()
    rows = cluster_bootstrap.load_paired_rows(conn, release_id=1, args=_args_for_load())
    response_ids = sorted(r["response_id"] for r in rows)
    assert response_ids == [201, 203]


# -----------------------------------------------------------------------------
# run_bootstrap — determinism under fixed seed
# -----------------------------------------------------------------------------

def _bootstrap_args(seed: int = 12345, reps: int = 50, min_paired_rows: int = 1) -> Namespace:
    return Namespace(seed=seed, bootstrap_reps=reps, min_paired_rows=min_paired_rows)


def _bootstrap_fixture_rows(tmp_path: Path) -> list[dict]:
    db_path = tmp_path / "fixture_boot.db"
    conn = _build_fixture_db(db_path)
    _populate_fixture(conn, with_duplicate=False)
    return cluster_bootstrap.load_paired_rows(conn, release_id=1, args=_args_for_load())


def test_run_bootstrap_deterministic_with_fixed_seed(tmp_path: Path):
    rows = _bootstrap_fixture_rows(tmp_path)
    a1 = cluster_bootstrap.run_bootstrap(rows, _bootstrap_args(seed=42))
    a2 = cluster_bootstrap.run_bootstrap(rows, _bootstrap_args(seed=42))
    # Leaderboard structure must be identical
    assert a1["leaderboard"] == a2["leaderboard"]
    # Reliability summary including bootstrap CIs must be identical
    assert a1["interjudge_reliability"] == a2["interjudge_reliability"]
    # Per-domain block must be identical
    assert a1["per_domain"] == a2["per_domain"]


def test_run_bootstrap_differs_with_different_seeds(tmp_path: Path):
    rows = _bootstrap_fixture_rows(tmp_path)
    a1 = cluster_bootstrap.run_bootstrap(rows, _bootstrap_args(seed=42))
    a2 = cluster_bootstrap.run_bootstrap(rows, _bootstrap_args(seed=43))
    # Sanity: with only 3 questions in 1 domain the resampling distribution is
    # narrow but two different seeds should still produce different bootstrap
    # CIs at the configured 50 reps. If this assertion ever fails, check that
    # rng was seeded inside run_bootstrap, not at module level.
    assert a1["interjudge_reliability"]["bootstrap_ci"] != a2["interjudge_reliability"]["bootstrap_ci"]


# -----------------------------------------------------------------------------
# paired_rows_sha256 — provenance helper sanity
# -----------------------------------------------------------------------------

def test_paired_rows_sha256_stable_under_reorder():
    """The hash should be invariant to row ordering since it sorts by response_id."""
    rows_a = [
        {"response_id": 1, "codex_score": 2.5, "claude_score": 3.0},
        {"response_id": 2, "codex_score": 1.0, "claude_score": 1.5},
    ]
    rows_b = list(reversed(rows_a))
    assert cluster_bootstrap.paired_rows_sha256(rows_a) == cluster_bootstrap.paired_rows_sha256(rows_b)


def test_paired_rows_sha256_changes_with_score():
    rows_a = [{"response_id": 1, "codex_score": 2.5, "claude_score": 3.0}]
    rows_b = [{"response_id": 1, "codex_score": 2.5, "claude_score": 3.5}]
    assert cluster_bootstrap.paired_rows_sha256(rows_a) != cluster_bootstrap.paired_rows_sha256(rows_b)


# -----------------------------------------------------------------------------
# weighted_composite_means — role-audition composite per the philology
# weighting supplement
# -----------------------------------------------------------------------------


def test_weighted_composite_means_default_weights_sum_to_one():
    """The default ROLE_AUDITION_WEIGHTS must sum to 1.0 within float tolerance.

    This is asserted by weighted_composite_means at call time; the test ensures
    the constant is correctly maintained.
    """
    total = sum(cluster_bootstrap.ROLE_AUDITION_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9, f"weights sum to {total}, expected 1.0"


def test_weighted_composite_means_golden_case():
    """Hand-derivable golden case for the role-audition weighted composite.

    Setup: a single model 'm' with hand-picked per-domain means designed to make
    the composite arithmetic transparent.

      D1 = 4.0   weight 0.08    contribution 0.32
      D2 = 3.0   weight 0.17    contribution 0.51
      D3 = 2.0   weight 0.175   contribution 0.35
      D4 = 1.0   weight 0.07    contribution 0.07
      D5 = 5.0   weight 0.14    contribution 0.70
      D6 = 2.0   weight 0.145   contribution 0.29
      D7 = 3.0   weight 0.22    contribution 0.66

      composite = 0.32 + 0.51 + 0.35 + 0.07 + 0.70 + 0.29 + 0.66 = 2.90
    """
    domain_model_means = {
        "D1": {"m": 4.0},
        "D2": {"m": 3.0},
        "D3": {"m": 2.0},
        "D4": {"m": 1.0},
        "D5": {"m": 5.0},
        "D6": {"m": 2.0},
        "D7": {"m": 3.0},
    }
    composites = cluster_bootstrap.weighted_composite_means(domain_model_means)
    assert "m" in composites
    assert abs(composites["m"] - 2.90) < 1e-9, f"got {composites['m']}, expected 2.90"


def test_weighted_composite_means_excludes_models_missing_a_domain():
    """A model missing from any weighted domain is excluded from the output (loud,
    not silent zero)."""
    domain_model_means = {
        "D1": {"m1": 1.0, "m2": 1.0},
        "D2": {"m1": 1.0, "m2": 1.0},
        "D3": {"m1": 1.0, "m2": 1.0},
        "D4": {"m1": 1.0, "m2": 1.0},
        "D5": {"m1": 1.0, "m2": 1.0},
        "D6": {"m1": 1.0, "m2": 1.0},
        "D7": {"m1": 1.0},  # m2 absent from D7
    }
    composites = cluster_bootstrap.weighted_composite_means(domain_model_means)
    assert "m1" in composites
    assert "m2" not in composites
    # m1 composite = 1.0 * (sum of all weights) = 1.0 * 1.0 = 1.0
    assert abs(composites["m1"] - 1.0) < 1e-9


def test_weighted_composite_means_rejects_non_unit_weights():
    """Weights that do not sum to 1.0 must raise (loud failure on miscalibration)."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        cluster_bootstrap.weighted_composite_means(
            {"D1": {"m": 1.0}}, weights={"D1": 0.5}
        )


def test_weighted_composite_means_marsbench_gemma_4_31b_it_sanity():
    """Sanity-check against the round-2 audit's hand-computation of Gemma 4 31B-IT.

    Per audit U-1 round-2 finding (decisions.md line 19): with the per-domain
    averaged Codex+Claude paired-overlap means
        D1=4.3029, D2=2.825, D3=2.1086, D4=2.5054,
        D5=2.8991, D6=2.0, D7=2.1632
    the weighted composite should equal 2.5406 (rounded to 4 places).

    This is a direct production-data sanity check that the implementation
    matches the manual computation the audit relied on.
    """
    domain_model_means = {
        "D1": {"gemma": 4.3029},
        "D2": {"gemma": 2.825},
        "D3": {"gemma": 2.1086},
        "D4": {"gemma": 2.5054},
        "D5": {"gemma": 2.8991},
        "D6": {"gemma": 2.0},
        "D7": {"gemma": 2.1632},
    }
    composites = cluster_bootstrap.weighted_composite_means(domain_model_means)
    # Expected: 0.08*4.3029 + 0.17*2.825 + 0.175*2.1086 + 0.07*2.5054
    #         + 0.14*2.8991 + 0.145*2.0 + 0.22*2.1632
    #         = 0.344232 + 0.480250 + 0.369005 + 0.175378
    #         + 0.405874 + 0.290000 + 0.475904
    #         = 2.540643
    assert abs(composites["gemma"] - 2.540643) < 1e-4, f"got {composites['gemma']}"


# ---------------------------------------------------------------------------
# paired_composite_differences — paired-difference primitive for the §6 H-F
# bridge's distinguishability claim. Replaces the necessary-but-not-sufficient
# marginal-CI-overlap inference (Schenker and Gentleman 2001) with a properly
# paired test.
# ---------------------------------------------------------------------------


def test_paired_composite_differences_default_pairs_known():
    """The default PAIRED_DIFFERENCE_PAIRS includes the §6 H-F bridge pair."""
    assert ("llama-3.3-70b-instruct", "gemma-4-31b-it") in cluster_bootstrap.PAIRED_DIFFERENCE_PAIRS


def test_paired_composite_differences_simple():
    """For the (llama, gemma) pair with composites llama=2.627, gemma=2.541,
    the default-pairs call should return one entry with diff = 0.086."""
    composites = {
        "llama-3.3-70b-instruct": 2.627,
        "gemma-4-31b-it": 2.541,
    }
    diffs = cluster_bootstrap.paired_composite_differences(composites)
    assert len(diffs) == 1
    pair_key = "llama-3.3-70b-instruct__minus__gemma-4-31b-it"
    assert pair_key in diffs
    assert abs(diffs[pair_key] - 0.086) < 1e-9


def test_paired_composite_differences_custom_pairs():
    """Caller-supplied pairs should produce a difference per pair, both signs."""
    composites = {"a": 3.0, "b": 1.5, "c": 2.0}
    diffs = cluster_bootstrap.paired_composite_differences(
        composites, pairs=[("a", "b"), ("b", "c"), ("c", "a")]
    )
    assert diffs == {
        "a__minus__b": 1.5,
        "b__minus__c": -0.5,
        "c__minus__a": -1.0,
    }


def test_paired_composite_differences_skips_missing_models():
    """If either model in a pair is missing from `composites`, the pair is
    silently skipped — this happens during a bootstrap resample where one
    model doesn't have per-domain means for every weighted domain."""
    composites = {"a": 1.0}  # b missing
    diffs = cluster_bootstrap.paired_composite_differences(
        composites, pairs=[("a", "b"), ("b", "a")]
    )
    assert diffs == {}  # both pairs skipped


def test_paired_composite_differences_zero_diff():
    """Identical composites yield 0.0 difference, not None or missing entry."""
    composites = {"a": 2.5, "b": 2.5}
    diffs = cluster_bootstrap.paired_composite_differences(
        composites, pairs=[("a", "b")]
    )
    assert diffs == {"a__minus__b": 0.0}
