"""DFM mixed-frequency parameter setup.

This module contains functions for setting up mixed-frequency parameters
from configuration, including tent kernel weights and constraint matrices.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from ...config.constants import (
    DEFAULT_HIERARCHY_VALUE,
    FREQUENCY_HIERARCHY,
    DEFAULT_IDENTITY_SCALE,
    DEFAULT_REGULARIZATION,
    MAX_TENT_SIZE,
)
from ...numeric.stability import create_scaled_identity
from ...utils.errors import ConfigurationError, DataValidationError
from .tent import get_agg_structure


def find_slower_frequency(
    clock: str,
    tent_weights_dict: Optional[Dict[str, np.ndarray]] = None
) -> Optional[str]:
    """Find slower frequency from tent_weights_dict or hierarchy.
    
    Parameters
    ----------
    clock : str
        Clock frequency
    tent_weights_dict : Optional[Dict[str, np.ndarray]]
        Dictionary of tent weights by frequency
        
    Returns
    -------
    Optional[str]
        Slower frequency if found, None otherwise
    """
    # Try tent_weights_dict first
    if tent_weights_dict:
        slower_freq = next((freq for freq in tent_weights_dict.keys() if freq != clock), None)
        if slower_freq is not None:
            return slower_freq
    
    # Try slower frequencies from hierarchy (sorted by hierarchy, ascending)
    # Note: Without lookup table, we rely on tent_weights_dict which must be provided
    # from config. This function now requires tent_weights_dict to be populated.
    clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, DEFAULT_HIERARCHY_VALUE)
    slower_freqs = sorted(
        [freq for freq in FREQUENCY_HIERARCHY if FREQUENCY_HIERARCHY[freq] > clock_hierarchy],
        key=lambda f: FREQUENCY_HIERARCHY[f]
    )
    # Return first slower frequency if tent_weights_dict is empty (fallback)
    # This is a best-effort guess; proper tent weights should be in tent_weights_dict
    if slower_freqs:
        return slower_freqs[0]
    
    return None


def setup_mixed_frequency_params(
    config: Any,
    clock: str,
    columns: Optional[List[str]],
    N: int
) -> Dict[str, Any]:
    """Extract and compute mixed-frequency parameters from config.
    
    Parameters
    ----------
    config : Any
        DFMConfig instance with get_frequencies() method
    clock : str
        Clock frequency
    columns : Optional[List[str]]
        Column names from dataset
    N : int
        Number of series (columns)
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with keys: R_mat, q, n_slower_freq, n_clock_freq, tent_weights_dict,
        frequencies_np, idio_indicator, mixed_freq, tent_kernel_size
    """
    # Get frequencies for each series from config
    if columns is not None:
        frequencies_list = config.get_frequencies(columns=columns)
    else:
        frequencies_list = []
    
    # Compute mixed-frequency parameters from config
    agg_structure = get_agg_structure(config, clock=clock)
    structures = agg_structure['structures']
    tent_weights_dict = agg_structure['tent_weights']
    
    # Determine if mixed-frequency and extract R_mat, q
    if structures:
        # Mixed-frequency: use first structure (typically only one slower frequency)
        (R_mat, q) = next(iter(structures.values()))
        mixed_freq = True
    else:
        # Single-frequency: no tent kernel constraints
        R_mat = None
        q = None
        mixed_freq = False
    
    # Count slower-frequency series
    if frequencies_list:
        clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, DEFAULT_HIERARCHY_VALUE)
        
        # Validate clock frequency is in hierarchy
        if clock not in FREQUENCY_HIERARCHY:
            raise ConfigurationError(
                f"Clock frequency '{clock}' is not in FREQUENCY_HIERARCHY. "
                f"Valid frequencies: {list(FREQUENCY_HIERARCHY.keys())}",
                details=f"Clock frequency: {clock}"
            )
        
        # Identify slower frequencies and validate
        slower_freqs = [
            freq for freq in frequencies_list 
            if FREQUENCY_HIERARCHY.get(freq, DEFAULT_HIERARCHY_VALUE) > clock_hierarchy
        ]
        
        # Validation: Allow only one clock frequency and one additional slower frequency
        try:
            unique_slower_freqs = list(set(slower_freqs))
            if len(unique_slower_freqs) > 1:
                raise ConfigurationError(
                    f"Only one slower frequency is allowed, but found {len(unique_slower_freqs)}: {unique_slower_freqs}. "
                    f"DFM currently supports one clock frequency and one additional slower frequency only.",
                    details=f"Slower frequencies found: {unique_slower_freqs}, clock: {clock}"
                )
            
            # Validate each slower frequency is in hierarchy
            for freq in unique_slower_freqs:
                if freq not in FREQUENCY_HIERARCHY:
                    raise ConfigurationError(
                        f"Slower frequency '{freq}' is not in FREQUENCY_HIERARCHY. "
                        f"Valid frequencies: {list(FREQUENCY_HIERARCHY.keys())}",
                        details=f"Invalid slower frequency: {freq}, clock: {clock}"
                    )
            
            n_slower_freq = len(slower_freqs)
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(
                f"Error validating slower frequencies: {e}",
                details=f"Frequencies: {frequencies_list}, clock: {clock}"
            ) from e
    else:
        n_slower_freq = 0
    
    n_clock_freq = N - n_slower_freq
    
    # Convert frequencies to numpy array if available
    frequencies_np = np.array(frequencies_list) if frequencies_list else None
    
    # Idiosyncratic indicator: all series have idiosyncratic components by default
    idio_indicator = np.ones(N, dtype=np.int32)
    
    # Determine tent kernel size and validate
    try:
        if R_mat is not None:
            tent_kernel_size = R_mat.shape[1]
        elif tent_weights_dict:
            first_weights = next(iter(tent_weights_dict.values()))
            tent_kernel_size = len(first_weights)
        else:
            tent_kernel_size = 1  # No tent kernel for single-frequency data
        
        # Validate tent kernel size
        if tent_kernel_size < 1:
            raise DataValidationError(
                f"Tent kernel size must be >= 1, but got {tent_kernel_size}",
                details=f"Tent kernel size: {tent_kernel_size}"
            )
        
        if tent_kernel_size > MAX_TENT_SIZE:
            raise DataValidationError(
                f"Tent kernel size {tent_kernel_size} exceeds maximum allowed size {MAX_TENT_SIZE}. "
                f"For frequency gaps larger than {MAX_TENT_SIZE}, use the missing data approach instead.",
                details=f"Tent kernel size: {tent_kernel_size}, MAX_TENT_SIZE: {MAX_TENT_SIZE}"
            )
    except (DataValidationError, ConfigurationError):
        raise
    except Exception as e:
        raise DataValidationError(
            f"Error determining or validating tent kernel size: {e}",
            details=f"R_mat shape: {R_mat.shape if R_mat is not None else None}, "
                   f"tent_weights_dict: {list(tent_weights_dict.keys()) if tent_weights_dict else None}"
        ) from e
    
    return {
        'R_mat': R_mat,
        'q': q,
        'n_slower_freq': n_slower_freq,
        'n_clock_freq': n_clock_freq,
        'tent_weights_dict': tent_weights_dict,
        'frequencies_np': frequencies_np,
        'idio_indicator': idio_indicator,
        'mixed_freq': mixed_freq,
        'tent_kernel_size': tent_kernel_size
    }


def build_slower_freq_idiosyncratic_chain(
    n_slower_freq: int,
    chain_size: int,
    rho0: float,
    sig_e: np.ndarray,
    dtype: type = np.float32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build slower-frequency idiosyncratic chain transition matrices and covariance.
    
    Parameters
    ----------
    n_slower_freq : int
        Number of slower-frequency series
    chain_size : int
        Chain size (tent kernel size, e.g., 5 for quarterly-to-monthly)
    rho0 : float
        AR(1) coefficient for slower-frequency series
    sig_e : np.ndarray
        Observation noise variances for slower-frequency series (n_slower_freq,)
    dtype : type, default np.float32
        Data type for output matrices
        
    Returns
    -------
    BQ : np.ndarray
        Transition matrix for slower-frequency chains (chain_size * n_slower_freq x chain_size * n_slower_freq)
    SQ : np.ndarray
        Process noise covariance (chain_size * n_slower_freq x chain_size * n_slower_freq)
    initViQ : np.ndarray
        Initial covariance (chain_size * n_slower_freq x chain_size * n_slower_freq)
    """
    if n_slower_freq == 0:
        return (
            np.zeros((0, 0), dtype=dtype),
            np.zeros((0, 0), dtype=dtype),
            np.zeros((0, 0), dtype=dtype)
        )
    
    # Build block structure
    temp = np.zeros((chain_size, chain_size), dtype=dtype)
    temp[0, 0] = 1.0
    SQ = np.kron(np.diag((1 - rho0 ** 2) * sig_e), temp)
    
    BQ_block = np.zeros((chain_size, chain_size), dtype=dtype)
    BQ_block[0, 0] = rho0
    BQ_block[1:, :chain_size-1] = create_scaled_identity(chain_size-1, DEFAULT_IDENTITY_SCALE, dtype=dtype)
    BQ = np.kron(create_scaled_identity(n_slower_freq, DEFAULT_IDENTITY_SCALE, dtype=dtype), BQ_block)
    
    # Compute initial covariance: solve (I - BQ ⊗ BQ) vec(V_0) = vec(SQ)
    from ...numeric.estimator import compute_initial_covariance_from_transition
    initViQ = compute_initial_covariance_from_transition(BQ, SQ, regularization=DEFAULT_REGULARIZATION, dtype=dtype)
    
    return BQ, SQ, initViQ
