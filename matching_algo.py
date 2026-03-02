import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment #hungarian algorithm
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import heapq
import time

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
    ):
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
        # Load similarity matrix
        if similarity_matrix is not None:
            self.similarity=similarity_matrix
        elif similarity_csv is not None:
            self.similarity=self._load_csv(similarity_csv)
        else:
            raise ValueError("must provide either similarity_csv or similarity_matrix containing the similarity scores")
        
        # set number of mentors and mentees
        self.n_mentees,self.n_mentors=self.similarity.shape

        # set the list describing the mentors capacities
        if mentor_capacities is None:
            self.mentor_capacities=[1]*self.n_mentors
        else:
            self.mentor_capacities=mentor_capacities

        # set mentee ids
        if mentee_ids:
            self.mentee_ids=mentee_ids
        else:
            self.mentee_ids=list(range(self.n_mentees))
        
        # set mentor ids
        if mentor_ids:
            self.mentor_ids=mentor_ids
        else:
            self.mentor_ids=list(range(self.n_mentors))

        # set verbose flag
        self.verbose=verbose

    def _load_csv(self, filepath: str)->np.ndarray:
        """
        Loads similarity matrix from csv
        """
        df=pd.read_csv(filepath, index_col=0)
        return df.values
    
    def _log(self, message: str):
        """
        Prints logs if verbose is enabled
        """
        if self.verbose:
            print(message)

    def _expand_for_capacity(self, scores: np.ndarray)-> Tuple[np.ndarray,List[int]]:
        """
        Expand score matrix to handle >1 capcity of mentors

        Each mentor with capacity k gets duplicated k times

        Returns:
            (expanded_matrix, mentor_mapping)

            expanded_matrix = (n_mentees, mentor_entity)
            mentor_mapping = list of mentor indices that represent the columns in mentor_entity
        """
        expanded_cols=[]
        mentor_mapping=[]

        for mentor_idx in range(self.n_mentors):
            capacity=self.mentor_capacities[mentor_idx]
            for _ in range(capacity):
                expanded_cols.append(scores[:,mentor_idx])
                mentor_mapping.append(mentor_idx)
        
        expanded_matrix=np.column_stack(expanded_cols)
    
        return expanded_matrix, mentor_mapping
    
    def _pad_matrix(self, matrix:np.ndarray)->np.ndarray:
        """
        Pad matrix with dummy columns and rows if needed
        """
        n_rows=matrix.shape[0]
        n_cols=matrix.shape[1]
        
        if n_rows==n_cols:
            return matrix
        
        elif n_rows>n_cols:
            print(f"Adding {n_rows-n_cols} ghost columns")
            diff=n_rows-n_cols
            padding=np.full((n_rows,diff),-1e9)
            return np.hstack([matrix,padding])
        else:
            print(f"Adding {n_cols-n_rows} ghost rows")
            diff=n_cols-n_rows
            padding=np.full((diff,n_cols),-1e9)
            return np.vstack([matrix,padding])
        
    
    def match_hungarian(self)-> MatchingResults:
        """
        Hungarian Algorithm for maximum weight bipartite matching

        Guarantees optimal solution (maximum total compatibility score)
        Capacity constraints are handled by matrix expansion

        Time complexity: O(n^3), where n=max(no of mentees, no of mentors)

        Returns:
            MatchingResults with optimal matches
        """

        start=time.time()

        self._log("\n"+"="*70)
        self._log("HUNGARIAN ALGORITHM (OPTIMAL)")
        self._log("="*70)

        # expand for capacity
        expanded, mentor_mapping=self._expand_for_capacity(self.similarity)
        
        # pad the matrix
        expanded=self._pad_matrix(expanded)

        # calculate the similarity matrix from the cost matrix
        cost_matrix=-expanded

        # result of hungarian algorithm
        mentee_indices, col_indices=linear_sum_assignment(cost_matrix)

        # list to store results
        matches=[]

        # track the mentor utilization
        mentor_util={i:0 for i in range(self.n_mentors)}

        for mentee_idx,col_idx in zip(mentee_indices, col_indices):
            
            # === ERROR HANDLING ===
            # if a mentee is matched with a dummy column
            if col_idx>=len(mentor_mapping):
                self._log("Avoiding matches with padded columns")
                continue

            # if a mentor is matched with a dummy row
            if mentee_idx>=self.n_mentees:
                self._log("Avoiding matches with padded rows")
                continue
            
            # get mentor index based on the column values
            mentor_idx=mentor_mapping[col_idx]

            # === COMPUTATION ===
            score=self.similarity[mentee_idx,mentor_idx]

            if score>-1e8:
                # add the match to the list
                matches.append((mentor_idx,mentee_idx,score))

                # keep track of mentor_utilization  
                mentor_util[mentor_idx]+=1

            # track of mentees who are matched
            matched_mentees=set([m[0] for m in matches])

            # track of unmatched mentees
            unmatched_mentees=[i for i in range(self.n_mentees) if i not in matched_mentees]

            # track of matched mentors
            matched_mentors=set([m[1] for m in matches])

            #track of unmatched mentors
            unmatched_mentors=[i for i in range(self.n_mentors) if i not in matched_mentors]

            total_score=sum(m[2] for m in matches)
            avg_score=total_score/len(matches) if matches else 0

            elapsed=(time.time()-start)*1000
            
            self._log(f"Generated {len(matches)} matches")
            self._log(f"{len(matched_mentees)} mentees assigned")
            self._log(f"{len(matched_mentors)} mentors assigned")
            self._log(f"Average score: {avg_score:.6f}")
            self._log(f"Execution time: {elapsed:.2f}ms")

            return MatchingResults(
                algorithm="Hungarian",
                matches=matches,
                unmatched_mentees=unmatched_mentees,
                unmatched_mentors=unmatched_mentors,
                mentor_utilization=mentor_util,
                total_score=total_score,
                average_score=avg_score,
                num_matches=len(matches),
                execution_time_ms=elapsed
            )