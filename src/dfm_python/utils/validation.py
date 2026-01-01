"""Validation utilities for common checks.

This module provides generic validation helpers that can be used across
the codebase to reduce duplication and improve consistency.
"""

from typing import Any, Optional, Type


def check_condition(
    condition: bool,
    error_class: Type[Exception],
    message: str,
    details: Optional[str] = None
) -> None:
    """Check a condition and raise an error if it fails.
    
    Parameters
    ----------
    condition : bool
        Condition to check (raises if False, returns None if True)
    error_class : Type[Exception]
        Exception class to raise
    message : str
        Error message
    details : str, optional
        Additional error details
        
    Returns
    -------
    None
        Returns None if condition is True (validation passes)
        
    Raises
    ------
    error_class
        If condition is False
    """
    if not condition:
        if details:
            raise error_class(f"{message}\nDetails: {details}")
        else:
            raise error_class(message)


def check_not_none(
    value: Any,
    name: str,
    error_class: Type[Exception] = ValueError
) -> None:
    """Check that a value is not None.
    
    Parameters
    ----------
    value : Any
        Value to check
    name : str
        Name of the value (for error message)
    error_class : Type[Exception], default ValueError
        Exception class to raise
        
    Raises
    ------
    error_class
        If value is None
    """
    if value is None:
        raise error_class(f"{name} must not be None")


def check_has_attr(
    obj: Any,
    attr: str,
    name: str,
    error_class: Type[Exception] = AttributeError
) -> None:
    """Check that an object has an attribute.
    
    Parameters
    ----------
    obj : Any
        Object to check
    attr : str
        Attribute name
    name : str
        Name of the object (for error message)
    error_class : Type[Exception], default AttributeError
        Exception class to raise
        
    Raises
    ------
    error_class
        If object does not have the attribute
    """
    if not hasattr(obj, attr):
        raise error_class(f"{name} must have attribute '{attr}'")


def has_shape_with_min_dims(
    obj: Any,
    min_dims: int = 2
) -> bool:
    """Check if an object has a shape attribute with at least min_dims dimensions.
    
    This is a non-raising helper for conditional checks (unlike validation helpers
    that raise exceptions). Used to consolidate duplicate shape checking patterns.
    
    Parameters
    ----------
    obj : Any
        Object to check
    min_dims : int, default=2
        Minimum number of dimensions required
        
    Returns
    -------
    bool
        True if object has shape attribute with at least min_dims dimensions, False otherwise
    """
    return obj is not None and hasattr(obj, 'shape') and len(obj.shape) >= min_dims

