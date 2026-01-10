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
    """Check a condition and raise an error if it fails."""
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
    """Check that a value is not None."""
    if value is None:
        raise error_class(f"{name} must not be None")


def check_has_attr(
    obj: Any,
    attr: str,
    name: str,
    error_class: Type[Exception] = AttributeError
) -> None:
    """Check that an object has an attribute."""
    if not hasattr(obj, attr):
        raise error_class(f"{name} must have attribute '{attr}'")


def has_shape_with_min_dims(
    obj: Any,
    min_dims: int = 2
) -> bool:
    """Check if an object has a shape attribute with at least min_dims dimensions.
    
    Non-raising helper for conditional checks (unlike validation helpers that raise exceptions).
    """
    return obj is not None and hasattr(obj, 'shape') and len(obj.shape) >= min_dims

