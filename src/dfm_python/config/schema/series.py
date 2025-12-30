"""Series configuration for Dynamic Factor Models.

This module contains SeriesConfig, which defines the configuration
for individual time series in a factor model.
"""

from dataclasses import dataclass
from typing import Optional

# Define validate_frequency locally to avoid circular import with utils.config
# This breaks the circular dependency: config.schema.series -> utils.config -> utils.__init__ -> models -> config
_VALID_FREQUENCIES = {'d', 'w', 'm', 'q', 'sa', 'a'}

def _validate_frequency_local(frequency: str) -> str:
    """Validate frequency string (local implementation to avoid circular import)."""
    if frequency not in _VALID_FREQUENCIES:
        from ...utils.errors import ConfigurationError
        raise ConfigurationError(f"Invalid frequency '{frequency}'. Must be one of {_VALID_FREQUENCIES}")
    return frequency

# ============================================================================
# Series Configuration
# ============================================================================

@dataclass
class SeriesConfig:
    """Configuration for a single time series.
    
    This is a generic DFM configuration - no API or database-specific fields.
    For API/database integration, implement adapters in your application layer.
    
    Note: Transformation is handled by preprocessing pipeline, not in SeriesConfig.
    Note: Blocks are defined in DFMConfig, not in SeriesConfig.
    
    Attributes
    ----------
    frequency : str
        Series frequency: 'm' (monthly), 'q' (quarterly), 'sa' (semi-annual), 'a' (annual)
    series_id : str, optional
        Unique identifier (auto-generated if None)
    series_name : str, optional
        Human-readable name (defaults to series_id if None)
    units : str, optional
        Units of measurement (optional metadata for display purposes only).
        Used in news decomposition output for readability. Not used in model estimation.
    release_date : int, optional
        Release date information for pseudo real-time nowcasting.
        - Positive value (1-31): Day of month when data is released
        - Negative value: Days before end of previous month when data is released
        Example: 25 = released on 25th of each month, -5 = released 5 days before end of previous month
    """
    # Required fields (no defaults)
    frequency: str
    # Optional fields (with defaults - must come after required fields)
    series_id: Optional[str] = None  # Auto-generated if None: "series_0", "series_1", etc.
    series_name: Optional[str] = None  # Optional metadata for display
    units: Optional[str] = None  # Optional metadata for display only (used in news.py output)
    release_date: Optional[int] = None  # Release date for pseudo real-time nowcasting
    
    def __post_init__(self):
        """Validate fields after initialization."""
        self.frequency = _validate_frequency_local(self.frequency)
        # Auto-generate series_name if not provided
        if self.series_name is None and self.series_id:
            self.series_name = self.series_id

