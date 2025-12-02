"""Configuration utility functions.

This module provides validation functions and constants for DFM configuration:
- validate_frequency: Validate frequency codes
- validate_transformation: Validate transformation codes
- Constants for valid codes and transformation mappings
"""

import warnings

# Valid frequency codes
_VALID_FREQUENCIES = {'d', 'w', 'm', 'q', 'sa', 'a'}

# Valid transformation codes
_VALID_TRANSFORMATIONS = {
    'lin', 'chg', 'ch1', 'pch', 'pc1', 'pca', 
    'cch', 'cca', 'log'
}

# Transformation to readable units mapping
TRANSFORM_UNITS_MAP = {
    'lin': 'Levels (No Transformation)',
    'chg': 'Change (Difference)',
    'ch1': 'Year over Year Change (Difference)',
    'pch': 'Percent Change',
    'pc1': 'Year over Year Percent Change',
    'pca': 'Percent Change (Annual Rate)',
    'cch': 'Continuously Compounded Rate of Change',
    'cca': 'Continuously Compounded Annual Rate of Change',
    'log': 'Natural Log'
}


def validate_frequency(frequency: str) -> str:
    """Validate frequency code.
    
    Parameters
    ----------
    frequency : str
        Frequency code to validate
        
    Returns
    -------
    str
        Validated frequency code (same as input if valid)
        
    Raises
    ------
    ValueError
        If frequency is not in the set of valid frequencies
        
    Examples
    --------
    >>> validate_frequency('m')
    'm'
    >>> validate_frequency('invalid')
    ValueError: Invalid frequency: invalid. Must be one of {'d', 'w', 'm', 'q', 'sa', 'a'}
    """
    if frequency not in _VALID_FREQUENCIES:
        raise ValueError(f"Invalid frequency: {frequency}. Must be one of {_VALID_FREQUENCIES}")
    return frequency


def validate_transformation(transformation: str) -> str:
    """Validate transformation code.
    
    Parameters
    ----------
    transformation : str
        Transformation code to validate
        
    Returns
    -------
    str
        Validated transformation code (same as input, even if unknown)
        
    Notes
    -----
    Unknown transformation codes trigger a warning but are not rejected,
    allowing for extensibility. The code will be used as-is, and the
    transformation logic should handle unknown codes appropriately.
    
    Examples
    --------
    >>> validate_transformation('lin')
    'lin'
    >>> validate_transformation('unknown')  # Issues warning but returns value
    'unknown'
    """
    if transformation not in _VALID_TRANSFORMATIONS:
        warnings.warn(f"Unknown transformation code: {transformation}. Will use untransformed data.")
    return transformation

