"""
test_two_phase.py
-----------------
Tests _run_two_phase with all three algorithms using real CSV data.

Expected input files
--------------------
similarity_scores.csv
    Row index    : mentee IDs  (e.g. mentee_001, mentee_002, ...)
    Column headers: mentor IDs (e.g. mentor_001, mentor_002, ...)
    Values       : float similarity scores

personality_similarity_score.csv
    Same shape / index / columns as similarity_scores.csv

mentor_metadata.csv
    Columns: mentor_id, capacity
    Example:
        mentor_id,capacity
        mentor_001,2
        mentor_002,3

mentee_metadata.csv
    Columns: mentee_id, is_also_mentor
    Example:
        mentee_id,is_also_mentor
        mentee_001,True
        mentee_002,False

All four files must be in the same directory as this script, or update
the paths in the CONFIG block below.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from matching_algorithms import MatchingSystem, MatchingResults


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ── update paths here if your files live elsewhere
# ─────────────────────────────────────────────────────────────────────────────
SIMILARITY_CSV      = "similarity_scores.csv"
PERSONALITY_CSV     = "personality_similarity_score.csv"
MENTOR_METADATA_CSV = "mentor_metadata.csv"          # columns: mentor_id, capacity
MENTEE_METADATA_CSV = "mentee_metadata.csv"          # columns: mentee_id, is_also_mentor


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & validate all input files
# ─────────────────────────────────────────────────────────────────────────────
def load_inputs():
    """
    Loads all four CSVs, validates consistency, and returns a dict of
    ready-to-use values for MatchingSystem.

    Returns
    -------
    dict with keys:
        mentee_ids        : list of str
        mentor_ids        : list of str
        mentor_capacities : list of int  (aligned to mentor_ids order)
        mentee_flags      : list of bool (aligned to mentee_ids order)
        similarity_csv    : str  (path, passed through to MatchingSystem)
        personality_csv   : str  (path, passed through to MatchingSystem)
    """

    # ── similarity matrix (drives the canonical ID order) ────────────────────
    print("Loading similarity_scores.csv ...")
    sim_df = pd.read_csv(SIMILARITY_CSV, index_col=0)
    mentee_ids = list(sim_df.index.astype(str))
    mentor_ids = list(sim_df.columns.astype(str))
    print(f"  → {len(mentee_ids)} mentees × {len(mentor_ids)} mentors")

    # ── personality matrix ────────────────────────────────────────────────────
    print("Loading personality_similarity_score.csv ...")
    pers_df = pd.read_csv(PERSONALITY_CSV, index_col=0)
    pers_df.index   = pers_df.index.astype(str)
    pers_df.columns = pers_df.columns.astype(str)

    if list(pers_df.index) != mentee_ids or list(pers_df.columns) != mentor_ids:
        raise ValueError(
            "personality_similarity_score.csv does not match the shape / ID order "
            "of similarity_scores.csv.\n"
            "Both files must have identical row and column labels in the same order."
        )
    print(f"  → shape matches ✓")

    # ── mentor capacities ─────────────────────────────────────────────────────
    print("Loading mentor_metadata.csv ...")
    mentor_meta = pd.read_csv(MENTOR_METADATA_CSV, dtype={"mentor_id": str, "capacity": int})

    if "mentor_id" not in mentor_meta.columns or "capacity" not in mentor_meta.columns:
        raise ValueError("mentor_metadata.csv must have columns: mentor_id, capacity")

    mentor_meta = mentor_meta.set_index("mentor_id")

    # Check every mentor in similarity matrix has a capacity entry
    missing_mentors = [mid for mid in mentor_ids if mid not in mentor_meta.index]
    if missing_mentors:
        raise ValueError(
            f"mentor_metadata.csv is missing capacity entries for: {missing_mentors}"
        )

    # Align capacities to the column order of similarity_scores.csv
    mentor_capacities = [int(mentor_meta.loc[mid, "capacity"]) for mid in mentor_ids]
    print(f"  → capacities loaded  (total slots = {sum(mentor_capacities)})")

    # ── mentee flags ──────────────────────────────────────────────────────────
    print("Loading mentee_metadata.csv ...")
    mentee_meta = pd.read_csv(MENTEE_METADATA_CSV, dtype={"mentee_id": str})

    if "mentee_id" not in mentee_meta.columns or "is_also_mentor" not in mentee_meta.columns:
        raise ValueError("mentee_metadata.csv must have columns: mentee_id, is_also_mentor")

    mentee_meta = mentee_meta.set_index("mentee_id")

    # Normalise the boolean column (handles "True"/"False" strings and 0/1)
    mentee_meta["is_also_mentor"] = mentee_meta["is_also_mentor"].map(
        lambda v: str(v).strip().lower() in ("true", "1", "yes")
    )

    # Check every mentee in similarity matrix has a flag entry
    missing_mentees = [mid for mid in mentee_ids if mid not in mentee_meta.index]
    if missing_mentees:
        raise ValueError(
            f"mentee_metadata.csv is missing flag entries for: {missing_mentees}"
        )

    # Align flags to the row order of similarity_scores.csv
    mentee_flags = [bool(mentee_meta.loc[mid, "is_also_mentor"]) for mid in mentee_ids]
    n_flagged = sum(mentee_flags)
    print(f"  → flags loaded  ({n_flagged} mentees flagged as phase-1 priority)")

    # ── capacity sanity check ─────────────────────────────────────────────────
    n_mentees       = len(mentee_ids)
    total_capacity  = sum(mentor_capacities)
    if total_capacity < n_mentees:
        print(
            f"\n  ⚠  WARNING: total mentor capacity ({total_capacity}) < "
            f"number of mentees ({n_mentees}). "
            f"{n_mentees - total_capacity} mentee(s) will be unmatched.\n"
        )

    return dict(
        mentee_ids        = mentee_ids,
        mentor_ids        = mentor_ids,
        mentor_capacities = mentor_capacities,
        mentee_flags      = mentee_flags,
        similarity_csv    = SIMILARITY_CSV,
        personality_csv   = PERSONALITY_CSV,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Helper: print + validate results
# ─────────────────────────────────────────────────────────────────────────────
def run_and_report(label: str, results: MatchingResults, system: MatchingSystem):
    print("\n" + "=" * 65)
    print(f"RESULTS: {label}")
    print("=" * 65)
    print(f"  Algorithm        : {results.algorithm}")
    print(f"  Matches made     : {results.num_matches}")
    print(f"  Unmatched mentees: {len(results.unmatched_mentees)}")
    print(f"  Unmatched mentors: {len(results.unmatched_mentors)}")
    print(f"  Average score    : {results.average_score:.4f}")
    print(f"  Total score      : {results.total_score:.4f}")
    print(f"  Time (ms)        : {results.execution_time_ms:.2f}")
    print()

    df = system.results_to_df(results)
    # For large datasets, show head + tail instead of the full table
    if len(df) > 30:
        print(df.head(15).to_string(index=False))
        print(f"  ... ({len(df) - 30} rows hidden) ...")
        print(df.tail(15).to_string(index=False))
    else:
        print(df.to_string(index=False))

    # Validate
    try:
        system.validate_results(results)
    except ValueError as e:
        print(f"\n  [VALIDATION FAILED] {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build a fresh MatchingSystem from shared config dict
# ─────────────────────────────────────────────────────────────────────────────
def make_system(cfg: dict, verbose: bool = True) -> MatchingSystem:
    return MatchingSystem(
        similarity_csv               = cfg["similarity_csv"],
        personality_similarity_csv   = cfg["personality_csv"],
        mentor_capacities            = cfg["mentor_capacities"],
        mentee_ids                   = cfg["mentee_ids"],
        mentor_ids                   = cfg["mentor_ids"],
        mentee_flags                 = cfg["mentee_flags"],
        verbose                      = verbose,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "=" * 65)
    print("LOADING INPUT DATA")
    print("=" * 65)
    cfg = load_inputs()

    n_mentees = len(cfg["mentee_ids"])
    n_mentors = len(cfg["mentor_ids"])
    print(
        f"\nReady: {n_mentees} mentees "
        f"({sum(cfg['mentee_flags'])} phase-1 priority) "
        f"× {n_mentors} mentors "
        f"(total capacity = {sum(cfg['mentor_capacities'])})"
    )

    # ── TEST 1: Two-Phase + LAPJV ─────────────────────────────────────────────
    print("\n\n>>> TEST 1: _run_two_phase with match_lapjv")
    system_lapjv = make_system(cfg, verbose=True)
    results_lapjv = system_lapjv._run_two_phase(lambda s: s.match_lapjv())
    run_and_report("Two-Phase + LAPJV", results_lapjv, system_lapjv)

    # ── TEST 2: Two-Phase + Stable Marriage ───────────────────────────────────
    print("\n\n>>> TEST 2: _run_two_phase with match_stable")
    system_stable = make_system(cfg, verbose=True)
    results_stable = system_stable._run_two_phase(lambda s: s.match_stable())
    run_and_report("Two-Phase + Stable Marriage", results_stable, system_stable)

    # ── TEST 3: Two-Phase + Min-Cost Max-Flow ─────────────────────────────────
    print("\n\n>>> TEST 3: _run_two_phase with match_min_cost_flow")
    system_mcf = make_system(cfg, verbose=True)
    results_mcf = system_mcf._run_two_phase(lambda s: s.match_min_cost_flow())
    run_and_report("Two-Phase + Min-Cost Max-Flow", results_mcf, system_mcf)

    # ── SIDE-BY-SIDE COMPARISON ───────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 65)
    print(f"{'Metric':<25} {'LAPJV':>12} {'Stable':>12} {'MinCostFlow':>12}")
    print("-" * 65)
    print(f"{'Matches made':<25} {results_lapjv.num_matches:>12} {results_stable.num_matches:>12} {results_mcf.num_matches:>12}")
    print(f"{'Total score':<25} {results_lapjv.total_score:>12.4f} {results_stable.total_score:>12.4f} {results_mcf.total_score:>12.4f}")
    print(f"{'Average score':<25} {results_lapjv.average_score:>12.4f} {results_stable.average_score:>12.4f} {results_mcf.average_score:>12.4f}")
    print(f"{'Time (ms)':<25} {results_lapjv.execution_time_ms:>12.2f} {results_stable.execution_time_ms:>12.2f} {results_mcf.execution_time_ms:>12.2f}")
    print(f"{'Unmatched mentees':<25} {len(results_lapjv.unmatched_mentees):>12} {len(results_stable.unmatched_mentees):>12} {len(results_mcf.unmatched_mentees):>12}")

    # ── TEST 4: MLflow experiment ─────────────────────────────────────────────
    print("\n\n>>> TEST 4: MLflow experiment tracking")
    try:
        from matching_mlflow import MatchingTracker

        tracker = MatchingTracker(experiment_name="mentorship_matching_real")

        all_results = tracker.compare_algorithms(
            system=make_system(cfg, verbose=False),
            algorithms=["lapjv", "stable", "min_cost_flow"],
            two_phase=True,
            run_prefix="real_data_",
            extra_params={"n_mentees": n_mentees, "n_mentors": n_mentors},
        )

        print("\n[MLflow] All runs saved. To view the UI, run:")
        print("    mlflow ui")
        print("    # then open http://localhost:5000")
        print("    # Navigate to 'mentorship_matching_real' experiment")

    except ImportError as e:
        print(f"[MLflow] Skipped — install mlflow first: pip install mlflow\n  ({e})")