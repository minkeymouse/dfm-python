from typing import Dict, List
import numpy as np

from ..utils.aggregation import FREQUENCY_HIERARCHY


def group_series_by_frequency(
    idx_i: np.ndarray,
    frequencies: np.ndarray,
    clock: str
) -> Dict[str, np.ndarray]:
    """Group series indices by their actual frequency.
    
    Groups series by their actual frequency values, allowing each frequency
    to be processed independently. Faster frequencies than clock are rejected.
    
    Parameters
    ----------
    idx_i : np.ndarray
        Array of series indices to group (1D integer array)
    frequencies : np.ndarray
        Array of frequency strings for each series (e.g., 'm', 'q', 'sa', 'a')
        Length should match total number of series
    clock : str
        Clock frequency ('m', 'q', 'sa', 'a') - all factors evolve at this frequency
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping frequency strings to arrays of series indices.
        Keys are frequency strings (e.g., 'm', 'q'), values are numpy arrays
        of integer indices for series with that frequency.
        
    Raises
    ------
    ValueError
        If any series has a frequency faster than the clock frequency
        (e.g., daily/weekly when clock is monthly)
        
    Notes
    -----
    - Faster frequencies (lower hierarchy number) are not supported and raise ValueError
    - If frequencies is None or empty, all series are grouped under clock frequency
    - Used in init_conditions and em_step for mixed-frequency handling
    """
    if frequencies is None or len(frequencies) == 0:
        # Fallback: assume all are same as clock if frequencies not provided
        return {clock: idx_i.copy()}
    
    clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)  # Default to monthly
    
    freq_groups: Dict[str, List[int]] = {}
    faster_indices = []
    
    for idx in idx_i:
        if idx >= len(frequencies):
            # Index out of bounds - skip
            continue
        
        freq = frequencies[idx]
        freq_hierarchy = FREQUENCY_HIERARCHY.get(freq, 3)  # Default to monthly
        
        if freq_hierarchy < clock_hierarchy:
            # Faster frequency (lower hierarchy number) - NOT SUPPORTED
            faster_indices.append(idx)
        else:
            # Group by actual frequency
            if freq not in freq_groups:
                freq_groups[freq] = []
            freq_groups[freq].append(idx)
    
    # Validate: faster frequencies are not supported
    if len(faster_indices) > 0:
        raise ValueError(
            f"Higher frequencies (daily, weekly) are not supported. "
            f"Found {len(faster_indices)} series with frequency faster than clock '{clock}'. "
            f"Please use monthly, quarterly, semi-annual, or annual frequencies only."
        )
    
    # Convert lists to numpy arrays
    return {freq: np.array(indices, dtype=int) for freq, indices in freq_groups.items()}


