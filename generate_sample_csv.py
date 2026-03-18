"""
Generate Sample Similarity Score CSV
=====================================

Creates a sample CSV file with realistic similarity scores for testing.
"""

import numpy as np
import pandas as pd

def generate_sample_csv(
    n_mentees: int = 100,
    n_mentors: int = 25,
    output_file: str = 'sample_scores.csv',
    score_range: tuple = (0.3, 0.95)
):
    """
    Generate sample similarity score CSV.

    Args:
        n_mentees: Number of mentees
        n_mentors: Number of mentors
        output_file: Output CSV filename
        score_range: (min, max) range for scores
    """
    # Generate scores
    scores = np.random.uniform(score_range[0], score_range[1], (n_mentees, n_mentors))

    # Add some structure (clusters of higher similarity)
    for i in range(0, n_mentees, 10):
        mentor_idx = np.random.randint(0, n_mentors)
        scores[i:i+10, mentor_idx] += 0.1
        scores[i:i+10, mentor_idx] = np.clip(scores[i:i+10, mentor_idx], 0, 1)

    # Create IDs
    mentee_ids = [f"mentee_{i:04d}" for i in range(n_mentees)]
    mentor_ids = [f"mentor_{i:03d}" for i in range(n_mentors)]

    # Create DataFrame
    df = pd.DataFrame(scores, index=mentee_ids, columns=mentor_ids)

    # Save
    df.to_csv(output_file)

    print(f"✓ Generated {output_file}")
    print(f"  Shape: {n_mentees} mentees × {n_mentors} mentors")
    print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"  Average score: {scores.mean():.3f}")

    return df

if __name__ == "__main__":
    generate_sample_csv()
