

# Mentorship Matching Pipeline: Integration & Usage Guide

This pipeline utilizes three distinct algorithms, a two-phase priority matching system, and MLflow experiment tracking to match mentees with mentors efficiently.

---

## 1. Overview

The pipeline matches mentees to mentors using the following core algorithms:
### Core Algorithms

* **LAPJV (Modified Hungarian):** Global optimization. Uses a blended cost matrix ($Score_{Primary} + \epsilon \cdot Score_{Personality}$) to maximize the total compatibility of the entire group.
* **Stable Marriage (Gale-Shapley):** Stability focus. Implements a **Mentor-Proposing** version of the Hospital-Resident algorithm where personality scores strictly break ties in preference lists.
* **Min-Cost Max-Flow:** Network flow optimization. Models the problem as a graph to guarantee maximum possible matches (cardinality) while minimizing "cost" (maximizing score).


### Two-Phase Priority System

Each algorithm supports a priority matching phase to handle specific user roles:

* **Phase 1:** Mentees who are also registered mentors are matched first against the full mentor capacity.
* **Phase 2:** All remaining mentees are matched after Phase 1 is complete.

---

### Core Files (Updated)

* `matching_algorithms.py`: Contains the `MatchingSystem` class with all three algorithms.
* `matching_mlflow.py`: An MLflow wrapper for experiment tracking and comparison.
* `test_two_phase.py`: A standard test runner for validating the pipeline on a single run.
* **`run_experiments.py`**: A statistical benchmarking harness that runs multiple trials with different seeds to compare algorithm performance and stability.

---

## 2. Repository Layout

```text
project/
├── matching_algorithms.py       # Core MatchingSystem class
├── matching_mlflow.py           # MLflow tracking wrapper
├── test_two_phase.py            # Single-run entry point
├── run_experiments.py           # Multi-trial benchmarking harness
│
├── similarity_scores.csv        # (User provided)
├── personality_similarity_score.csv  # (User provided)
├── mentor_metadata.csv          # (User provided)
└── mentee_metadata.csv          # (User provided)

```

---

> **Note:** All four CSV files must be in the same directory as the scripts, or the `CONFIG` paths in `test_two_phase.py` must be updated.

---

## 3. Dependencies

### 3.1 Install Python Packages

```bash
pip install numpy pandas scipy
pip install matching          # Gale-Shapley (stable marriage)
pip install networkx          # Min-Cost Max-Flow
pip install mlflow            # Optional — experiment tracking

```

### 3.2 Python Version

Python **3.8 or higher** is required.

---

## 4. Input File Specifications

The pipeline requires four CSV files with specific formats.

### 4.1 similarity_scores.csv

Defines pairwise similarity scores and the canonical ID ordering.

| Column / Row | Description |
| --- | --- |
| **Row index (col 0)** | `mentee_id` values (e.g., mentee_001) |
| **Column headers** | `mentor_id` values (e.g., mentor_001) |
| **Cell values** | float in **[0, 1]** (higher = better match) |

### 4.2 personality_similarity_score.csv

Used as a tie-breaker for Stable Marriage. It **must** have identical row/column labels and order as `similarity_scores.csv`.

* **Warning:** If indices/columns do not match exactly, a `ValueError` will be raised at startup.

### 4.3 mentor_metadata.csv

Defines mentor capacity.

| Column | Type / Values |
| --- | --- |
| **mentor_id** | str (must match `similarity_scores.csv` headers) |
| **capacity** | int $\ge$ 1 (number of mentees a mentor can accept) |

> **Note:** Every mentor ID in the similarity matrix must have an entry here. If total capacity is less than the number of mentees, some will remain unmatched.

### 4.4 mentee_metadata.csv

Identifies mentees for Phase 1 priority.

| Column | Type / Values |
| --- | --- |
| **mentee_id** | str (must match `similarity_scores.csv` index) |
| **is_also_mentor** | bool (True/False, 1/0, or yes/no) |

---

## 5. Running the Pipeline

### 5.1 Quick Start

Run the following command to load inputs, run all algorithms, and log to MLflow:

```bash
python test_two_phase.py

```

### 5.2 Direct Usage

You can use `MatchingSystem` directly in your code:

```python
from matching_algorithms import MatchingSystem

system = MatchingSystem(
    similarity_csv             = "similarity_scores.csv",
    personality_similarity_csv = "personality_similarity_score.csv",
    mentor_capacities          = [2, 3, 1],
    mentee_flags               = [True, False, True, False],
)

# Run with two-phase priority
results = system._run_two_phase(lambda s: s.match_lapjv())
df = system.results_to_df(results)

```


## 6. Benchmarking with `run_experiments.py`

While `test_two_phase.py` is used for a single execution, `run_experiments.py` is designed for **scientific evaluation** of the algorithms.

### 6.1 What it does

* **Multiple Trials:** Runs each algorithm $N$ times (default: 10) using different random seeds.
* **Statistical Analysis:** Performs a **Welch’s t-test** to determine if the difference in scores between algorithms is statistically significant ($p < 0.05$).
* **Aggregated Metrics:** Calculates Mean, Standard Deviation, and Coefficient of Variation (CV%) for execution time, match quality, and unmatched counts.
* **MLflow Integration:** Logs every individual trial to MLflow for deep-dive visualization.

### 6.2 How to run

Execute the benchmarking suite from your terminal:

```bash
python run_experiments.py

```

### 6.3 Understanding the Output

The script prints a summary table. A **High CV%** in execution time suggests an algorithm is inconsistent, while a **significant p-value (✓)** indicates one algorithm is objectively better than another for your specific dataset.

---

### Regarding Offsets (How they work)

Since you asked "how do offsets work" in the prompt context of this code:

In the `min_cost_flow` logic, the "Network" is just a giant list of nodes. To keep things organized, we use **offsets** to map different entities to specific ranges of ID numbers:

1. **Source Node:** Always ID `0`.
2. **Mentee Nodes:** Start at ID `1`. If you have 100 mentees, they occupy IDs `1` to `100`.
3. **Mentor Nodes:** Start at ID `101` (`1 + n_mentees`). This is the **offset**. It ensures mentor IDs don't overlap with mentees.
4. **Sink Node:** The very last ID (`1 + n_mentees + n_mentors`).

**Example:**
If you have 5 mentees and 3 mentors:

* Mentees: `1, 2, 3, 4, 5`
* Mentors: `6, 7, 8` (Offset of $1 + 5 = 6$)
* Sink: `9`

This allows the computer to treat the "Mentorship Program" as a simple mathematical graph where every participant has a unique "seat number."

---

## 7. MLflow Experiment Tracking

When installed, the following are logged automatically:

* **Parameters:** Algorithm name, mentee/mentor counts, capacities, two-phase flag.
* **Metrics:** Total/average score, number of matches, unmatched counts, execution time.
* **Artifacts:** `matches.csv`, `mentor_utilisation.csv`, `summary.txt`.

To view results, run `mlflow ui` and navigate to **http://localhost:5000**.


---

## 8. Algorithm Comparison

| Algorithm | Optimizes For | Complexity | Best Used When |
| --- | --- | --- | --- |
| **LAPJV** | Maximum total score | $O(n^3)$ | Highest overall match quality is the goal. |
| **Stable Marriage** | Stability (no blocking pairs) | $O(n^2)$ | Preventing post-match "swaps" matters. |
| **Min-Cost Max-Flow** | Max cardinality then score | $O(n^2m)$ | Minimizing unmatched mentees is the priority. |

---

## 9. Validation & Errors

Call `system.validate_results(results)` to ensure:

1. No mentor exceeds capacity.
2. No mentee has multiple matches.
3. Match count + unmatched count equals total mentee count.

**Common Fixes:**

* **Shape/ID Error:** Ensure both similarity CSVs have identical labels/order.
* **Missing Entries:** Add rows for every ID present in the similarity matrix to the metadata files.