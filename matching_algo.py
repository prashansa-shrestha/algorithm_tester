import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment #LAPVJ (modified hungarian) algorithm
from matching.games import HospitalResident
from typing import List, Dict, Tuple, Optional, Callable
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
    1. LAPVJ (modified hungarian) Algorithm (optimal for total score)
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
            personality_similarity_matrix: Optional[np.ndarray]=None,
            personality_similarity_csv: Optional[str]=None,
            mentor_capacities: Optional[List[int]]=None,
            mentee_ids: Optional[List]=None,
            mentor_ids: Optional[List]=None,
            mentee_flags: Optional[List[bool]]=None,
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
        
        # Load personality similarity matrix
        if personality_similarity_matrix is not None:
            self.personality_similarity=personality_similarity_matrix
        elif personality_similarity_csv is not None:
            self.personality_similarity=self._load_csv(personality_similarity_csv)
        else:
            raise ValueError("must provide either personality_similarity_csv or personality_similarity_matrix containing the similarity scores")
        
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

        # store mentee flags (default to False if not provided)
        if mentee_flags is None:
            self.mentee_flags=np.array([False]*self.n_mentees)
        else:
            self.mentee_flags=np.array(mentee_flags)

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
            self._log(f"Adding {n_rows-n_cols} ghost columns")
            diff=n_rows-n_cols
            padding=np.full((n_rows,diff),-1e9)
            return np.hstack([matrix,padding])
        else:
            self._log(f"Adding {n_cols-n_rows} ghost rows")
            diff=n_cols-n_rows
            padding=np.full((diff,n_cols),-1e9)
            return np.vstack([matrix,padding])
        
    def _run_two_phase(self, algorithm_fn:Callable[["MatchingSystem"],MatchingResults])->MatchingResults:
        """
        Runs algorithm_fn in two phases:
            Phase 1: mentees with registered_as_mentor=True
            Phase 2: remaining mentees against remaining capacity of mentors

        algorithm_fn should accept MatchinSystem instance, and return MatchingResults
        """
        start=time.time()

        # partition the data into 2 mentee groups:
        #   registered as mentors (priority_indices)
        #   not registered as mentors (normal_indices)
        priority_indices=[i for i,f in enumerate(self.mentee_flags) if f]
        normal_indices=[i for i,f in enumerate(self.mentee_flags) if not f]

        # to store final results
        all_matches=[]
        remaining_capacities=list(self.mentor_capacities)

        for phase,indices in enumerate([priority_indices,normal_indices],start=1):
            # either of the indices are empty
            if not indices:
                continue

            self._log(f"\n--- Phase {phase}: {len(indices)} mentees ---")

            # similarity matrix of the specific phase
            sub_similarity=self.similarity[np.ix_(indices,range(self.n_mentors))]

            # create the subsystem
            sub_system=MatchingSystem(
                similarity_matrix=sub_similarity,
                mentor_capacities=remaining_capacities,
                mentee_ids=[self.mentee_ids[i] for i in indices],
                mentor_ids=self.mentor_ids,
                verbose=self.verbose,
            )

            # personality matrix of the specific phase if it exists
            # check existence
            if hasattr(self,'personality_similarity'):
                sub_system.personality_similarity=self.personality_similarity[
                    np.ix_(indices,range(self.n_mentors))
                ]

            # runs the matchmaking algorithm
            result=algorithm_fn(sub_system)

            for (sub_mentee_idx,mentor_idx,score) in result.matches:
                global_mentee_idx=indices[sub_mentee_idx]
                all_matches.append((global_mentee_idx,mentor_idx,score))
                remaining_capacities[mentor_idx]-=1

            # track of mentees who are matched
        matched_mentees=set([m[0] for m in all_matches])

        # track of unmatched mentees
        unmatched_mentees=[i for i in range(self.n_mentees) if i not in matched_mentees]

        # track of matched mentors
        matched_mentors=set([m[1] for m in all_matches])

        # track of unmatched mentors
        unmatched_mentors=[i for i in range(self.n_mentors) if i not in matched_mentors]

        # track utilization of mentors
        mentor_util={j:0 for j in range(self.n_mentors)}

        # update utilization
        for _,mentor_idx,_ in all_matches:
            mentor_util[mentor_idx]+=1

        total_score=sum(m[2] for m in all_matches)
        avg_score=total_score/len(all_matches) if all_matches else 0

        elapsed=(time.time()-start)*1000
        
        self._log(f"Generated {len(all_matches)} matches")
        self._log(f"{len(matched_mentees)} mentees assigned")
        self._log(f"{len(matched_mentors)} mentors assigned")
        self._log(f"Average score: {avg_score:.6f}")
        self._log(f"Execution time: {elapsed:.2f}ms")

        return MatchingResults(
            algorithm=f"{result.algorithm} (Two-Phase Priority)",
            matches=all_matches,
            unmatched_mentees=unmatched_mentees,
            unmatched_mentors=unmatched_mentors,
            mentor_utilization=mentor_util,
            total_score=total_score,
            average_score=avg_score,
            num_matches=len(all_matches),
            execution_time_ms=elapsed
        )

    def match_lapjv(self)-> MatchingResults:
        """
        Jonker-Volegnant(Modified Hungarian Algorithm) for maximum weight bipartite matching

        Guarantees optimal solution (maximum total compatibility score)
        Capacity constraints are handled by matrix expansion

        Time complexity: O(n^3), where n=max(no of mentees, no of mentors)

        Returns:
            MatchingResults with optimal matches
        """

        start=time.time()

        self._log("\n"+"="*70)
        self._log("LAPJV (MODIFIED HUNGARIAN) ALGORITHM (OPTIMAL)")
        self._log("="*70)

        # expand for capacity
        expanded, mentor_mapping=self._expand_for_capacity(self.similarity)
        
        # pad the matrix
        expanded=self._pad_matrix(expanded)

        # calculate the similarity matrix from the cost matrix
        cost_matrix=-expanded

        # result of lapjv (modified hungarian) algorithm
        mentee_indices, col_indices=linear_sum_assignment(cost_matrix)

        # list to store results
        matches=[]

        # track the mentor utilization
        mentor_util={i:0 for i in range(self.n_mentors)}

        for mentee_idx,col_idx in zip(mentee_indices, col_indices):
            
            # === ERROR HANDLING ===
            # if a mentee is matched with a dummy column
            i=0
            if col_idx>=len(mentor_mapping):
                i+=1
                if i<1:
                    self._log("Avoiding matches with padded columns")
                continue

            # if a mentor is matched with a dummy row
            j=0
            if mentee_idx>=self.n_mentees:
                j+=1
                if j<=1:
                    self._log("Avoiding matches with padded rows")
                continue
            
            # get mentor index based on the column values
            mentor_idx=mentor_mapping[col_idx]

            # === COMPUTATION ===
            score=self.similarity[mentee_idx,mentor_idx]

            if score>-1e8:
                # add the match to the list
                matches.append((mentee_idx,mentor_idx,score))

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
            algorithm="LAPVJ (modified hungarian)",
            matches=matches,
            unmatched_mentees=unmatched_mentees,
            unmatched_mentors=unmatched_mentors,
            mentor_utilization=mentor_util,
            total_score=total_score,
            average_score=avg_score,
            num_matches=len(matches),
            execution_time_ms=elapsed
        )
    
    def match_stable(self)->MatchingResults:
        """
        Stable Marriage (Gale Shapley) algorithm with capacity

        Produces stable matching: no mentee-mentor pair both prefer each
        other over their assigned matches. 
        Mentor-proposing version

        Uses personality similarity matrix for tie-breaking.

        Time Complexity: O(n^2) where n = number of mentees

        Returns:
            Matching Results with stable matches
        """
        start=time.time()

        self._log("\n" + "="*70)
        self._log("STABLE MARRIAGE (Gale-Shapley) - Mentor Proposing")
        self._log("="*70)

        # --- CONSTRUCT MENTEE'S PREFERENCE LIST ---
        mentee_prefs_dict={}

        # iterating through each mentee row in the similarity score matrix
        for i in range(self.n_mentees):
            scores=self.similarity[i,:]
        
            score_groups={}
            # iterating through each mentor for the mentee
            for j in range(self.n_mentors):
                score=scores[j]
                # if this is the first time mentee is seeing that score intialize empty list
                if score not in score_groups:
                    score_groups[score]=[]
                # add the mentor id which has obtained that score
                score_groups[score].append(j)
            
            # sort the scores
            sorted_scores = sorted(score_groups.keys(), reverse=True)
            
            # list for each mentee to store mentors sorted by rank
            ranked = []
            for score in sorted_scores:
                # list containing mentors sorted by rank
                mentors_with_score = score_groups[score]

                # if there are ties, use personality to break it
                if len(mentors_with_score) > 1:
                    mentors_with_score.sort(
                        key=lambda j: self.personality_similarity[i, j],
                        reverse=True
                    )

                # add it to the mentee value in ranked
                ranked.extend(mentors_with_score)

            mentee_prefs_dict[f"mentee_{i}"] = [f"mentor_{j}" for j in ranked]

            # --- CONSTRUCT MENTOR'S PREFERENCE LIST ---

        mentor_prefs_dict = {}
        mentor_capacity_dict = {}

        # iterating through each mentor row in the similarity matrix
        for j in range(self.n_mentors):
            scores = self.similarity[:, j]
            
            score_groups = {}
            for i in range(self.n_mentees):
                score = scores[i]
                if score not in score_groups:
                    score_groups[score] = []
                score_groups[score].append(i)
            
            sorted_scores = sorted(score_groups.keys(), reverse=True)
            ranked = []
            for score in sorted_scores:
                mentees_with_score = score_groups[score]
                if len(mentees_with_score) > 1:
                    mentees_with_score.sort(
                        key=lambda i: self.personality_similarity[i, j],
                        reverse=True
                    )
                ranked.extend(mentees_with_score)
            
            mentor_prefs_dict[f"mentor_{j}"] = [f"mentee_{i}" for i in ranked]
            mentor_capacity_dict[f"mentor_{j}"] = self.capacities[j]
       
        # call the algorithm
        game=HospitalResident.create_from_dictionaries(
            resident_prefs=mentee_prefs_dict,
            hospital_prefs=mentor_prefs_dict,
            hospital_capacities=mentor_capacity_dict
        )

        matching_result=game.solve(optimal="hospital")
         # Convert results back to our format
        matches = []
        mentee_matches = {}
        mentor_matches = {j: [] for j in range(self.n_mentors)}
        

        for mentor_name, mentees_list in matching_result.items():
            mentor_idx = int(mentor_name.split('_')[1])
            for mentee_obj in mentees_list:
                mentee_idx = int(mentee_obj.name.split('_')[1])
                score = self.similarity[mentee_idx, mentor_idx]
                matches.append((mentee_idx, mentor_idx, score))
                mentee_matches[mentee_idx] = mentor_idx
                mentor_matches[mentor_idx].append(mentee_idx)
        
        matched_mentees = [i for i in range(self.n_mentees) if i in mentee_matches]
        
        unmatched_mentees = [i for i in range(self.n_mentees) if i not in mentee_matches]
        
        # track of matched mentors
        matched_mentors=set([m[1] for m in matches])

        #track of unmatched mentors
        unmatched_mentors=[i for i in range(self.n_mentors) if i not in matched_mentors]

        mentor_util = {j: len(mentor_matches[j]) for j in range(self.n_mentors)}
        total_score = sum([m[2] for m in matches])
        avg_score = total_score / len(matches) if matches else 0
        
        elapsed = (time.time() - start) * 1000
        
        self._log(f"Generated {len(matches)} matches")
        self._log(f"{len(matched_mentees)} mentees assigned")
        self._log(f"{len(matched_mentors)} mentors assigned")
        self._log(f"Average score: {avg_score:.6f}")
        self._log(f"Execution time: {elapsed:.2f}ms")

        return MatchingResults(
            algorithm="Stable Marriage (Gale-Shapley) - Mentor Proposing",
            matches=matches,
            unmatched_mentees=unmatched_mentees,
            unmatched_mentors=unmatched_mentors,
            mentor_utilization=mentor_util,
            total_score=total_score,
            average_score=avg_score,
            num_matches=len(matches),
            execution_time_ms=elapsed
        )

    def results_to_df(self, results: MatchingResults) -> pd.DataFrame:
        """
        Converts MatchingResults into a clean, readable Pandas DataFrame.
        """
        # Create the base DataFrame
        df = pd.DataFrame(results.matches, columns=['mentee_idx', 'mentor_idx', 'score'])
        
        # Map the indices to the actual IDs you provided in __init__
        df['Mentee ID'] = df['mentee_idx'].apply(lambda x: self.mentee_ids[x])
        df['Mentor ID'] = df['mentor_idx'].apply(lambda x: self.mentor_ids[x])
        
        # Reorder and return only the readable columns
        return df[['Mentee ID', 'Mentor ID', 'score']]
    
    def validate_results(self, results: MatchingResults):
        """
        Hard-check on the matching output. 
        Will throw an assertion error if the math doesn't add up.
        """
        self._log("\n>>> AUDITING MATCH DATA")
        
        # 1. Capacity Check: Using a counter to be sure
        actual_counts = defaultdict(int)
        for _, mentor_idx, _ in results.matches:
            actual_counts[mentor_idx] += 1
            
        for m_idx, count in actual_counts.items():
            limit = self.mentor_capacities[m_idx]
            if count > limit:
                raise ValueError(f"CRITICAL: {self.mentor_ids[m_idx]} over-assigned! ({count}/{limit})")

        # 2. Uniqueness: One mentee, one slot.
        mentees_seen = [m[0] for m in results.matches]
        if len(mentees_seen) != len(set(mentees_seen)):
            raise ValueError("CRITICAL: Duplicate mentee detected in match list!")

        # 3. Score Truth: Cross-ref against the raw similarity matrix
        for m_idx, mentor_idx, score in results.matches:
            raw_val = self.similarity[m_idx, mentor_idx]
            if not np.isclose(raw_val, score):
                raise ValueError(f"DATA MISMATCH: Score for M{m_idx}->Mentor{mentor_idx} is tampered.")

        # 4. Ghost/Padding logic check
        # Matches + Unmatched Mentees must = total mentees
        if (len(results.matches) + len(results.unmatched_mentees)) != self.n_mentees:
            raise ValueError("LEAK DETECTED: Total mentee count doesn't balance with match/unmatch lists.")

        self._log(">>> [OK] Constraints verified. No leaks.")

    @staticmethod
    def test_algo():
        actual_n_mentees, actual_n_mentors = pd.read_csv('similarity_scores.csv', index_col=0).values.shape

        mentee_ids = [f"mentee_{i:03d}" for i in range(actual_n_mentees)]
        mentor_ids = [f"mentor_{i:03d}" for i in range(actual_n_mentors)]

        system=MatchingSystem(
            similarity_csv='similarity_scores.csv',
            mentor_capacities=np.random.randint(1,4,len(mentor_ids)).tolist(),
            mentee_ids=mentee_ids,
            mentor_ids=mentor_ids
        )

        results = system.match_lapjv()

        try:
            system.validate_results(results)
        except ValueError as e:
            print(f"\n[!] MATCHING FAILED VALIDATION: {e}")
            return

        # If we got here, it's safe to look at the data
        print("\nTop 5 Matches:")
        df = system.results_to_df(results)
        print(df.head(5).to_string(index=False))

        df_results = system.results_to_df(results)
        
        print("\n" + "="*30)
        print("DETAILED MATCHING BREAKDOWN")
        print("="*30)
        # to_string(index=False) hides the 0, 1, 2... row numbers
        print(df_results.to_string(index=False))

if __name__=="__main__":
    MatchingSystem.test_algo()