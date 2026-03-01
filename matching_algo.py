import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import heapq

@dataclass
class MatchingResults:
    """
    Container for matching results
    """
    algorithm: str
    matches: List[Tuple[int,int,float]]
    unmatched_mentees: List[int]
    unmatched_mentors: List[int]
    mentor_utilization: Dict[int,int]
    total_score: float
    average_score: float
    num_matches: int
    execution_time_ms: float=0.0

class MatchingSystem:
    """
    Implements multiple matching algorithms that work directly with pre-computed
    similarity/compatibility scores from all CSV files.

    Algorithms Implemented:
    1. Hungarian Algorithm (optimal for total score)
    2. Stable Marriage (Gale-Shapley/stability)
    3. Min-Cost Max-flow (network flow approach)

    Input Format:
    a. CSV with similarity scores between mentees and mentors
    b. Mentor Capacity Constraints
    """

    def __init__(
            self,
            similarity_matrix: Optional[np.ndarray]=None,
            similarity_csv: Optional[str]=None,
            mentor_capacities: Optional[List[int]]=None,
            mentee_ids: Optional[List]=None,
            mentor_ids: Optional[List]=None,
            verbose: bool=True
    )
        """
        Initializes matching system.

        Args:
            similarity_matrix: (n_mentees x n_mentors) score matrix
            similarity_csv: Path to CSV file with scores
            mentor_capacities: List of capacities of all mentors
            mentee_ids: List of mentee identifiers
            mentor_ids: List of mentor identifiers
            verbose: Flag to print progress logs in terminal
        """
        pass

    def _load_csv(self, filepath: str)->np.ndarray:
        """
        Loads similarity matrix from csv
        """
        df=pd.read_csv(filepath, index_col=0)
        return df.values
    
    def _log(self, message: str):
        if self.verbose:
            print(message)