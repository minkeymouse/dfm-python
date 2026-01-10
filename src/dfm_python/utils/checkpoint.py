"""Checkpoint loading utilities for factor models.

This module provides utilities for loading and parsing PyTorch model checkpoints,
inferring model parameters from state dictionaries, and extracting input dimensions.

The utilities are designed to work with DDFM models but can be parameterized
for other autoencoder-based models.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    _has_torch = True
except ImportError:
    _has_torch = False
    torch = None

from ..logger import get_logger
from ..utils.errors import DataError, DataValidationError, ConfigurationError
from ..utils.validation import has_shape_with_min_dims
from ..utils.misc import resolve_param
from ..config.constants import (
    MAX_WARNING_ITEMS,
    DEFAULT_ENCODER_LAYERS,
    DEFAULT_NUM_FACTORS,
    DEFAULT_ACTIVATION,
    DEFAULT_DECODER,
    DEFAULT_USE_BATCH_NORM,
)

_logger = get_logger(__name__)


def parse_checkpoint(checkpoint: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract state_dict and hyperparameters from checkpoint.
    
    Parameters
    ----------
    checkpoint : Any
        Checkpoint object (dict with 'state_dict' key or raw state_dict)
        
    Returns
    -------
    state_dict : Dict[str, Any]
        Model state dictionary
    hparams : Dict[str, Any]
        Hyperparameters dictionary (empty if not available)
    """
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint['state_dict'], checkpoint.get('hyper_parameters', {})
    return checkpoint, {}


def infer_input_dim_from_data(
    data: Union[pd.DataFrame, np.ndarray, "torch.Tensor"],
    date_id_col: str = "date_id"
) -> int:
    """Infer input dimension from data.
    
    Parameters
    ----------
    data : pd.DataFrame, np.ndarray, or torch.Tensor
        Input data
    date_id_col : str, default "date_id"
        Column name to exclude if data is DataFrame
        
    Returns
    -------
    int
        Input dimension (number of features)
        
    Raises
    ------
    DataError
        If data type is unsupported or shape is invalid
    """
    if isinstance(data, pd.DataFrame):
        # Exclude date column if present
        feature_cols = [c for c in data.columns if c != date_id_col]
        return len(feature_cols)
    elif isinstance(data, np.ndarray):
        if not has_shape_with_min_dims(data, min_dims=2):
            raise DataError(
                "Data must be at least 2D to infer input dimension",
                details=f"Got shape: {data.shape}"
            )
        return data.shape[1]
    elif _has_torch and isinstance(data, torch.Tensor):
        if not has_shape_with_min_dims(data, min_dims=2):
            raise DataError(
                "Data must be at least 2D to infer input dimension",
                details=f"Got shape: {data.shape}"
            )
        return data.shape[1]
    else:
        raise DataError(
            f"Unsupported data type: {type(data)}",
            details="Data must be pandas.DataFrame, numpy.ndarray, or torch.Tensor"
        )


def infer_input_dim_from_state_dict(
    state_dict: Dict[str, Any],
    encoder_prefix: str = "encoder.layers.0.weight",
    decoder_prefixes: Optional[List[str]] = None,
    output_dim_equals_input_dim: bool = True
) -> Optional[int]:
    """Infer input_dim from checkpoint state_dict (generalized for autoencoder models).
    
    This function can be used for DDFM or other autoencoder-based models by
    specifying the appropriate key prefixes.
    
    Parameters
    ----------
    state_dict : Dict[str, Any]
        Model state dictionary
    encoder_prefix : str, default "encoder.layers.0.weight"
        Key pattern for encoder first layer weight (shape: (hidden_dim, input_dim))
    decoder_prefixes : List[str], optional
        Key patterns for decoder layers. If None, uses default DDFM patterns.
        For DDFM: ["decoder.decoder.weight", "decoder.layers.0.weight"]
    output_dim_equals_input_dim : bool, default True
        If True, assumes decoder output_dim equals input_dim (DDFM convention).
        If False, decoder shape interpretation may differ.
        
    Returns
    -------
    Optional[int]
        Inferred input dimension, or None if cannot be determined
        
    Raises
    ------
    DataValidationError
        If state_dict is invalid
    """
    if not isinstance(state_dict, dict):
        raise DataValidationError(
            "Cannot infer input_dim: state_dict must be a dictionary",
            details=f"Received type: {type(state_dict).__name__}"
        )
    
    # Check encoder first layer: (hidden_dim, input_dim)
    first_layer_keys = [k for k in state_dict.keys() if encoder_prefix in k]
    if first_layer_keys:
        weight = state_dict[first_layer_keys[0]]
        if _has_torch and isinstance(weight, torch.Tensor):
            return weight.shape[1]  # input_dim is second dimension
    
    # Check decoder layers: (output_dim, num_factors) or (output_dim, input_dim)
    if decoder_prefixes is None:
        decoder_prefixes = ["decoder.decoder.weight", "decoder.layers.0.weight"]
    
    for prefix in decoder_prefixes:
        decoder_keys = [k for k in state_dict.keys() if prefix in k]
        for key in decoder_keys:
            weight = state_dict[key]
            if _has_torch and isinstance(weight, torch.Tensor):
                if output_dim_equals_input_dim:
                    return weight.shape[0]  # output_dim = input_dim (DDFM)
                else:
                    return weight.shape[1]  # input_dim is second dimension
    
    return None


def infer_ddfm_input_dim(state_dict: Dict[str, Any]) -> Optional[int]:
    """Infer input_dim from DDFM checkpoint state_dict.
    
    This is a convenience wrapper around infer_input_dim_from_state_dict
    with DDFM-specific defaults.
    
    Parameters
    ----------
    state_dict : Dict[str, Any]
        Model state dictionary
        
    Returns
    -------
    Optional[int]
        Inferred input dimension, or None if cannot be determined
        
    Raises
    ------
    DataValidationError
        If state_dict is invalid
    """
    return infer_input_dim_from_state_dict(
        state_dict,
        encoder_prefix="encoder.layers.0.weight",
        decoder_prefixes=["decoder.decoder.weight", "decoder.layers.0.weight"],
        output_dim_equals_input_dim=True
    )


def infer_ddfm_params_from_state_dict(
    state_dict: Dict[str, Any],
    hparams: Dict[str, Any],
    kwargs: Dict[str, Any],
    defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Infer DDFM model parameters from state_dict.
    
    Uses resolve_param utility for consistent parameter resolution and
    constants for default values instead of hardcoded literals.
    
    Parameters
    ----------
    state_dict : Dict[str, Any]
        Model state dictionary
    hparams : Dict[str, Any]
        Hyperparameters from checkpoint
    kwargs : Dict[str, Any]
        Additional keyword arguments
    defaults : Dict[str, Any], optional
        Default parameter values. If None, uses constants from config.
        
    Returns
    -------
    Dict[str, Any]
        Inferred model parameters
    """
    # Use constants for defaults instead of hardcoded values
    if defaults is None:
        defaults = {
            'encoder_layers': DEFAULT_ENCODER_LAYERS,
            'num_factors': DEFAULT_NUM_FACTORS,
            'activation': DEFAULT_ACTIVATION,
            'use_batch_norm': DEFAULT_USE_BATCH_NORM,
            'decoder': DEFAULT_DECODER,
            'decoder_layers': None,
        }
    
    if not isinstance(state_dict, dict):
        # Fallback: use resolve_param pattern for consistent resolution
        return {
            'encoder_layers': resolve_param(name='encoder_layers', kwargs=kwargs, config=None, defaults={**hparams, **defaults}),
            'num_factors': resolve_param(name='num_factors', kwargs=kwargs, config=None, defaults={**hparams, **defaults}),
            'activation': resolve_param(name='activation', kwargs=kwargs, config=None, defaults={**hparams, **defaults}),
            'use_batch_norm': resolve_param(name='use_batch_norm', kwargs=kwargs, config=None, defaults={**hparams, **defaults}),
            'decoder': resolve_param(name='decoder', kwargs=kwargs, config=None, defaults={**hparams, **defaults}),
            'decoder_layers': resolve_param(name='decoder_layers', kwargs=kwargs, config=None, defaults={**hparams, **defaults}),
        }
    
    # Infer encoder_layers from state_dict keys
    inferred_encoder_layers = None
    encoder_layer_keys = [k for k in sorted(state_dict.keys()) 
                         if 'encoder.layers' in k and 'weight' in k and 'output' not in k]
    if encoder_layer_keys:
        inferred_encoder_layers = []
        for key in encoder_layer_keys:
            weight = state_dict[key]
            if _has_torch and isinstance(weight, torch.Tensor):
                inferred_encoder_layers.append(weight.shape[0])
    
    # Infer num_factors from encoder output layer or decoder
    inferred_num_factors = None
    
    # Try encoder output layer first
    output_layer_keys = [k for k in state_dict.keys() if 'encoder.output_layer.weight' in k]
    if output_layer_keys:
        weight = state_dict[output_layer_keys[0]]
        if _has_torch and isinstance(weight, torch.Tensor):
            inferred_num_factors = weight.shape[0]
    
    # Fallback to decoder if encoder output not found
    if inferred_num_factors is None:
        decoder_keys = [k for k in state_dict.keys() if 'decoder' in k and 'weight' in k]
        for key in decoder_keys:
            if 'decoder.weight' in key or ('layers' in key and '0.weight' in key):
                weight = state_dict[key]
                if _has_torch and isinstance(weight, torch.Tensor):
                    inferred_num_factors = weight.shape[1]  # decoder input_dim = num_factors
                    break
    
    # Use resolve_param for consistent parameter resolution
    combined_defaults = {**defaults}
    if inferred_encoder_layers is not None:
        combined_defaults['encoder_layers'] = inferred_encoder_layers
    if inferred_num_factors is not None:
        combined_defaults['num_factors'] = inferred_num_factors
    
    return {
        'encoder_layers': resolve_param(name='encoder_layers', kwargs=kwargs, config=None, defaults={**hparams, **combined_defaults}),
        'num_factors': resolve_param(name='num_factors', kwargs=kwargs, config=None, defaults={**hparams, **combined_defaults}),
        'activation': resolve_param(name='activation', kwargs=kwargs, config=None, defaults={**hparams, **combined_defaults}),
        'use_batch_norm': resolve_param(name='use_batch_norm', kwargs=kwargs, config=None, defaults={**hparams, **combined_defaults}),
        'decoder': resolve_param(name='decoder', kwargs=kwargs, config=None, defaults={**hparams, **combined_defaults}),
        'decoder_layers': resolve_param(name='decoder_layers', kwargs=kwargs, config=None, defaults={**hparams, **combined_defaults}),
    }

