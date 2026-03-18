"""
run_experiments.py
------------------
Benchmarking harness for the three matching algorithms:
    lapjv | stable | min_cost_flow

Loads real CSV data (same format as test_two_phase.py), runs N_TRIALS
per algorithm with different random seeds, logs every run to MLflow,
and prints a statistical summary table at the end.

Usage
-----
    python run_experiments.py

Expected input files (same directory, or update CONFIG below)
-------------------------------------------------------------
    similarity_scores.csv
    personality_similarity_score.csv
    mentor_metadata.csv
    mentee_metadata.csv
"""

import copy
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from matching_algorithms import MatchingSystem
from matching_mlflow import MatchingTracker


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ── adjust paths, trial count, and experiment label here
# ─────────────────────────────────────────────────────────────────────────────
SIMILARITY_CSV      = "similarity_scores.csv"
PERSONALITY_CSV     = "personality_similarity_score.csv"
MENTOR_METADATA_CSV = "mentor_metadata.csv"
MENTEE_METADATA_CSV = "mentee_metadata.csv"

ALGORITHMS          = ["lapjv", "stable", "min_cost_flow"]
N_TRIALS            = 10         # number of repeated runs per algorithm
BASE_SEED           = 42         # seeds will be BASE_SEED + trial index
EXPERIMENT_NAME     = "algo_comparison_300_rows"
DATASET_TAG         = "cohort_2025_Q1"
TWO_PHASE           = True       # use priority-flagging two-phase wrapper


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load input data  (identical logic to test_two_phase.py)
# ─────────────────────────────────────────────────────────────────────────────
def load_inputs() -> dict:
    """
    Loads all four CSVs, validates consistency, and returns a config dict
    ready to pass into make_system().
    """
    print("=" * 65)
    print("LOADING INPUT DATA")
    print("=" * 65)

    # Similarity matrix — canonical ID order
    print(f"  Loading {SIMILARITY_CSV} ...")
    sim_df     = pd.read_csv(SIMILARITY_CSV, index_col=0)
    mentee_ids = list(sim_df.index.astype(str))
    mentor_ids = list(sim_df.columns.astype(str))
    print(f"  → {len(mentee_ids)} mentees × {len(mentor_ids)} mentors")

    # Personality matrix — must match shape/order
    print(f"  Loading {PERSONALITY_CSV} ...")
    pers_df         = pd.read_csv(PERSONALITY_CSV, index_col=0)
    pers_df.index   = pers_df.index.astype(str)
    pers_df.columns = pers_df.columns.astype(str)
    if list(pers_df.index) != mentee_ids or list(pers_df.columns) != mentor_ids:
        raise ValueError(
            f"{PERSONALITY_CSV} shape/ID order does not match {SIMILARITY_CSV}."
        )
    print("  → shape matches ✓")

    # Mentor capacities
    print(f"  Loading {MENTOR_METADATA_CSV} ...")
    mentor_meta = pd.read_csv(MENTOR_METADATA_CSV, dtype={"mentor_id": str, "capacity": int})
    if "mentor_id" not in mentor_meta.columns or "capacity" not in mentor_meta.columns:
        raise ValueError(f"{MENTOR_METADATA_CSV} must have columns: mentor_id, capacity")
    mentor_meta = mentor_meta.set_index("mentor_id")
    missing = [m for m in mentor_ids if m not in mentor_meta.index]
    if missing:
        raise ValueError(f"{MENTOR_METADATA_CSV} missing capacity for: {missing}")
    mentor_capacities = [int(mentor_meta.loc[m, "capacity"]) for m in mentor_ids]
    print(f"  → capacities loaded (total slots = {sum(mentor_capacities)})")

    # Mentee flags
    print(f"  Loading {MENTEE_METADATA_CSV} ...")
    mentee_meta = pd.read_csv(MENTEE_METADATA_CSV, dtype={"mentee_id": str})
    if "mentee_id" not in mentee_meta.columns or "is_also_mentor" not in mentee_meta.columns:
        raise ValueError(f"{MENTEE_METADATA_CSV} must have columns: mentee_id, is_also_mentor")
    mentee_meta = mentee_meta.set_index("mentee_id")
    mentee_meta["is_also_mentor"] = mentee_meta["is_also_mentor"].map(
        lambda v: str(v).strip().lower() in ("true", "1", "yes")
    )
    missing = [m for m in mentee_ids if m not in mentee_meta.index]
    if missing:
        raise ValueError(f"{MENTEE_METADATA_CSV} missing flag for: {missing}")
    mentee_flags = [bool(mentee_meta.loc[m, "is_also_mentor"]) for m in mentee_ids]
    print(f"  → flags loaded ({sum(mentee_flags)} mentees flagged as phase-1 priority)")

    # Capacity sanity
    n_mentees      = len(mentee_ids)
    total_capacity = sum(mentor_capacities)
    if total_capacity < n_mentees:
        print(
            f"\n  ⚠  WARNING: total capacity ({total_capacity}) < "
            f"mentees ({n_mentees}). "
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
# 2. Factory — fresh MatchingSystem from config dict
# ─────────────────────────────────────────────────────────────────────────────
def make_system(cfg: dict, verbose: bool = False) -> MatchingSystem:
    return MatchingSystem(
        similarity_csv             = cfg["similarity_csv"],
        personality_similarity_csv = cfg["personality_csv"],
        mentor_capacities          = cfg["mentor_capacities"],
        mentee_ids                 = cfg["mentee_ids"],
        mentor_ids                 = cfg["mentor_ids"],
        mentee_flags               = cfg["mentee_flags"],
        verbose                    = verbose,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_stats(values: list) -> dict:
    """Returns a dict of descriptive statistics for a list of floats."""
    arr = np.array(values, dtype=float)
    return {
        "mean":   float(np.mean(arr)),
        "std":    float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
        "median": float(np.median(arr)),
        "cv_%":   float(np.std(arr, ddof=1) / np.mean(arr) * 100)
                  if np.mean(arr) != 0 and len(arr) > 1 else 0.0,
    }


def pairwise_ttest(scores: dict) -> list:
    """
    Runs Welch's t-test on every pair of algorithms.
    Returns list of result dicts.
    """
    algos   = list(scores.keys())
    results = []
    for i in range(len(algos)):
        for j in range(i + 1, len(algos)):
            a, b   = algos[i], algos[j]
            t, p   = stats.ttest_ind(scores[a], scores[b], equal_var=False)
            results.append({"pair": f"{a} vs {b}", "t_stat": t, "p_value": p})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Print helpers
# ─────────────────────────────────────────────────────────────────────────────
DIVIDER = "=" * 75

def print_trial_header(trial: int, seed: int):
    print(f"\n{'─' * 75}")
    print(f"  TRIAL {trial + 1}/{N_TRIALS}   seed={seed}")
    print(f"{'─' * 75}")


def print_trial_row(algo: str, r):
    print(
        f"  {algo:<18}  score={r.total_score:>10.4f}  "
        f"matches={r.num_matches:>4}  "
        f"unmatched={len(r.unmatched_mentees):>3}  "
        f"time={r.execution_time_ms:>7.2f}ms"
    )


def print_summary(trial_data: dict):
    """
    trial_data: {algo: {"total_score": [...], "num_matches": [...],
                         "unmatched": [...], "time_ms": [...]}}
    """
    metrics = ["total_score", "num_matches", "unmatched", "time_ms"]
    labels  = {
        "total_score": "Total Score",
        "num_matches": "Num Matches",
        "unmatched":   "Unmatched",
        "time_ms":     "Time (ms)",
    }

    print(f"\n{DIVIDER}")
    print("STATISTICAL SUMMARY")
    print(DIVIDER)

    for metric in metrics:
        print(f"\n  {labels[metric]}")
        print(f"  {'Algorithm':<20} {'Mean':>10} {'Std':>10} {'Min':>10} "
              f"{'Max':>10} {'Median':>10} {'CV%':>8}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
        for algo in ALGORITHMS:
            s = compute_stats(trial_data[algo][metric])
            print(
                f"  {algo:<20} {s['mean']:>10.4f} {s['std']:>10.4f} "
                f"{s['min']:>10.4f} {s['max']:>10.4f} "
                f"{s['median']:>10.4f} {s['cv_%']:>7.2f}%"
            )

    # Pairwise significance on total_score
    print(f"\n  {'─'*65}")
    print("  Pairwise Welch's t-test on Total Score")
    print(f"  {'Pair':<35} {'t-stat':>10} {'p-value':>12} {'sig (p<0.05)':>14}")
    print(f"  {'─'*35} {'─'*10} {'─'*12} {'─'*14}")
    score_arrays = {a: trial_data[a]["total_score"] for a in ALGORITHMS}
    for row in pairwise_ttest(score_arrays):
        sig = "✓" if row["p_value"] < 0.05 else ""
        print(
            f"  {row['pair']:<35} {row['t_stat']:>10.4f} "
            f"{row['p_value']:>12.6f} {sig:>14}"
        )

    # Best algorithm by mean total_score
    best = max(ALGORITHMS, key=lambda a: np.mean(trial_data[a]["total_score"]))
    print(f"\n  ★  Best mean total score: {best.upper()}")
    print(DIVIDER)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    wall_start = time.time()

    # ── Load data ─────────────────────────────────────────────────────────────
    cfg = load_inputs()
    n_mentees = len(cfg["mentee_ids"])
    n_mentors = len(cfg["mentor_ids"])
    n_flagged = sum(cfg["mentee_flags"])

    print(
        f"\nDataset: {n_mentees} mentees "
        f"({n_flagged} phase-1 priority) × {n_mentors} mentors "
        f"(total capacity = {sum(cfg['mentor_capacities'])})"
    )

    # ── Set up MLflow tracker ─────────────────────────────────────────────────
    tracker = MatchingTracker(experiment_name=EXPERIMENT_NAME)

    # ── Storage for aggregation ───────────────────────────────────────────────
    trial_data = {
        algo: {"total_score": [], "num_matches": [], "unmatched": [], "time_ms": []}
        for algo in ALGORITHMS
    }

    # ── Trial loop ─────────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print(f"RUNNING {N_TRIALS} TRIALS × {len(ALGORITHMS)} ALGORITHMS")
    print(DIVIDER)

    for trial in range(N_TRIALS):
        seed = BASE_SEED + trial
        np.random.seed(seed)
        print_trial_header(trial, seed)

        results = tracker.compare_algorithms(
            system       = make_system(cfg, verbose=False),
            algorithms   = ALGORITHMS,
            two_phase    = TWO_PHASE,
            run_prefix   = f"trial_{trial:02d}_",
            extra_params = {
                "dataset":    DATASET_TAG,
                "trial":      trial,
                "seed":       seed,
                "n_mentees":  n_mentees,
                "n_mentors":  n_mentors,
                "two_phase":  TWO_PHASE,
                "notes":      f"baseline comparison — {len(ALGORITHMS)} algos, 300-row dataset",
            },
        )

        for algo, r in results.items():
            print_trial_row(algo, r)
            trial_data[algo]["total_score"].append(r.total_score)
            trial_data[algo]["num_matches"].append(r.num_matches)
            trial_data[algo]["unmatched"].append(len(r.unmatched_mentees))
            trial_data[algo]["time_ms"].append(r.execution_time_ms)

    # ── Summary ────────────────────────────────────────────────────────────────
    print_summary(trial_data)

    wall_elapsed = time.time() - wall_start
    total_runs   = N_TRIALS * len(ALGORITHMS)
    print(
        f"\nCompleted {total_runs} MLflow runs in {wall_elapsed:.1f}s  "
        f"(avg {wall_elapsed / total_runs:.2f}s/run)"
    )
    print(f"\nTo view results in the MLflow UI, run:")
    print(f"    mlflow ui")
    print(f"    # then open http://localhost:5000")
    print(f"    # Navigate to experiment: '{EXPERIMENT_NAME}'\n")


if __name__ == "__main__":
    main()