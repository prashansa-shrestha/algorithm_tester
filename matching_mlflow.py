"""
matching_mlflow.py
------------------
MLflow integration for the MatchingSystem.

Tracks every algorithm run as an MLflow experiment with:
  - Parameters  : algorithm name, n_mentees, n_mentors, capacities, flags
  - Metrics     : total_score, avg_score, num_matches, unmatched counts, time
  - Artifacts   : matches CSV, mentor utilisation CSV

Usage:
    from matching_mlflow import MatchingTracker

    tracker = MatchingTracker(experiment_name="mentorship_matching")

    # wrap any _run_two_phase or direct algorithm call:
    results = tracker.run(
        system=system,
        algorithm="lapjv",          # "lapjv" | "stable" | "min_cost_flow"
        two_phase=True,             # True uses _run_two_phase wrapper
        run_name="pilot_cohort_v1", # optional label
        extra_params={"cohort": "2025-Q3"}  # any extra info you want logged
    )
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import tempfile
import os
import copy
from typing import Optional, Dict, Any

from matching_algorithms import MatchingSystem, MatchingResults


class MatchingTracker:
    """
    Thin MLflow wrapper around MatchingSystem.

    Each call to .run() is one MLflow run logged under the given experiment.
    """

    ALGORITHM_MAP = {
        "lapjv":          lambda s: s.match_lapjv(),
        "stable":         lambda s: s.match_stable(),
        "min_cost_flow":  lambda s: s.match_min_cost_flow(),
    }

    def __init__(
        self,
        experiment_name: str = "mentorship_matching",
        tracking_uri: Optional[str] = None,
    ):
        """
        Args:
            experiment_name : MLflow experiment to log runs under.
            tracking_uri    : Optional remote URI e.g. "http://localhost:5000".
                              Defaults to local ./mlruns directory.
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        print(f"[MLflow] Experiment: '{experiment_name}'")
        print(f"[MLflow] Tracking URI: {mlflow.get_tracking_uri()}")

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def run(
        self,
        system: MatchingSystem,
        algorithm: str,
        two_phase: bool = True,
        run_name: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> MatchingResults:
        """
        Execute a matching algorithm and log everything to MLflow.

        Args:
            system       : Configured MatchingSystem instance.
            algorithm    : One of "lapjv", "stable", "min_cost_flow".
            two_phase    : If True, wraps with _run_two_phase (priority flagging).
            run_name     : Human-readable label for this MLflow run.
            extra_params : Any additional key-value pairs to log as params.

        Returns:
            MatchingResults from the algorithm.
        """
        if algorithm not in self.ALGORITHM_MAP:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Choose from: {list(self.ALGORITHM_MAP.keys())}"
            )

        algo_fn = self.ALGORITHM_MAP[algorithm]
        label = run_name or f"{algorithm}_{'two_phase' if two_phase else 'direct'}"

        with mlflow.start_run(run_name=label):

            # ── 1. Log parameters ─────────────────────────────────────────
            params = {
                "algorithm":        algorithm,
                "two_phase":        two_phase,
                "n_mentees":        system.n_mentees,
                "n_mentors":        system.n_mentors,
                "mentor_capacities": str(system.mentor_capacities),
                "total_capacity":   sum(system.mentor_capacities),
                "n_flagged_mentees": int(np.sum(system.mentee_flags)),
            }
            if extra_params:
                params.update(extra_params)

            mlflow.log_params(params)

            # ── 2. Run algorithm ──────────────────────────────────────────
            if two_phase:
                results = system._run_two_phase(algo_fn)
            else:
                results = algo_fn(system)

            # ── 3. Log metrics ────────────────────────────────────────────
            mlflow.log_metrics({
                "total_score":        results.total_score,
                "average_score":      results.average_score,
                "num_matches":        results.num_matches,
                "unmatched_mentees":  len(results.unmatched_mentees),
                "unmatched_mentors":  len(results.unmatched_mentors),
                "execution_time_ms":  results.execution_time_ms,
                "match_rate":         results.num_matches / system.n_mentees,
            })

            # ── 4. Log artifacts ──────────────────────────────────────────
            with tempfile.TemporaryDirectory() as tmp:
                # matches CSV
                matches_df = system.results_to_df(results)
                matches_path = os.path.join(tmp, "matches.csv")
                matches_df.to_csv(matches_path, index=False)
                mlflow.log_artifact(matches_path, artifact_path="results")

                # mentor utilisation CSV
                util_df = pd.DataFrame([
                    {
                        "mentor_id":  system.mentor_ids[j],
                        "assigned":   results.mentor_utilization[j],
                        "capacity":   system.mentor_capacities[j],
                        "utilisation": (
                            results.mentor_utilization[j] / system.mentor_capacities[j]
                            if system.mentor_capacities[j] > 0 else 0
                        ),
                    }
                    for j in range(system.n_mentors)
                ])
                util_path = os.path.join(tmp, "mentor_utilisation.csv")
                util_df.to_csv(util_path, index=False)
                mlflow.log_artifact(util_path, artifact_path="results")

                # summary txt
                summary_path = os.path.join(tmp, "summary.txt")
                with open(summary_path, "w") as f:
                    f.write(self._build_summary(results, system))
                mlflow.log_artifact(summary_path, artifact_path="results")

            run_id = mlflow.active_run().info.run_id
            print(f"[MLflow] Run logged → run_id: {run_id}")

        return results


    def compare_algorithms(self, system, algorithms=None, two_phase=True,
                        run_prefix="", extra_params=None):
        algorithms = algorithms or list(self.ALGORITHM_MAP.keys())
        all_results = {}

        for algo in algorithms:
            system_copy = copy.deepcopy(system)   # ← isolate each run
            name = f"{run_prefix}{algo}" if run_prefix else algo
            all_results[algo] = self.run(
                system=system_copy,
                algorithm=algo,
                two_phase=two_phase,
                run_name=name,
                extra_params=extra_params,
            )
        return all_results

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_summary(results: MatchingResults, system: MatchingSystem) -> str:
        lines = [
            f"Algorithm       : {results.algorithm}",
            f"Matches         : {results.num_matches}",
            f"Total score     : {results.total_score:.6f}",
            f"Average score   : {results.average_score:.6f}",
            f"Unmatched mentees: {results.unmatched_mentees}",
            f"Unmatched mentors: {results.unmatched_mentors}",
            f"Execution time  : {results.execution_time_ms:.2f}ms",
            "",
            "Mentor Utilisation:",
        ]
        for j in range(system.n_mentors):
            used = results.mentor_utilization[j]
            cap  = system.mentor_capacities[j]
            lines.append(f"  {system.mentor_ids[j]}: {used}/{cap}")
        return "\n".join(lines)