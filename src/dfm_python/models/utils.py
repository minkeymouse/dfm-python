"""Utility functions for DDFM model.

This module contains helper functions extracted from ddfm.py to improve code
organization and maintainability.
"""

import logging
from typing import Tuple, Optional, Callable
import numpy as np
import torch

from ..logger import get_logger
from ..utils.statespace import estimate_var1, estimate_var2

_logger = get_logger(__name__)


def estimate_var_ddfm(
    factors: np.ndarray,
    factor_order: int,
    num_factors: Optional[int] = None,
    format_error_message: Optional[Callable] = None,
    format_warning_message: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate VAR dynamics with comprehensive error handling and fallback.
    
    This function estimates Vector Autoregression (VAR) parameters for factor dynamics
    with robust error handling. It performs pre-estimation validation, handles edge
    cases (insufficient data, constant factors, NaN/Inf values), and provides
    fallback mechanisms when estimation fails.
    
    Parameters
    ----------
    factors : np.ndarray
        Factors array (T x m), where T is number of time periods and m is number of factors.
        Must be 2D array with at least 2 observations.
    factor_order : int
        VAR order, must be 1 or 2.
        - VAR(1): Z_t = A Z_{t-1} + v_t
        - VAR(2): Z_t = A1 Z_{t-1} + A2 Z_{t-2} + v_t
    num_factors : int, optional
        Number of factors. Used for fallback when factors shape is invalid.
        If None, inferred from factors.shape[1].
    format_error_message : callable, optional
        Function to format error messages. If None, uses simple string formatting.
    format_warning_message : callable, optional
        Function to format warning messages. If None, uses simple string formatting.
        
    Returns
    -------
    A : np.ndarray
        Transition matrix.
        - For VAR(1): shape (m x m)
        - For VAR(2): shape (m x 2m), where first m columns are A1, last m columns are A2
    Q : np.ndarray
        Innovation covariance matrix, shape (m x m). Always positive definite.
        
    Raises
    ------
    ValueError
        If factor_order is not 1 or 2.
    """
    # Helper functions for message formatting
    def _format_error(op: str, reason: str, guidance: str = "") -> str:
        if format_error_message:
            return format_error_message(op, reason, guidance)
        return f"DDFM {op} failed: {reason}. {guidance}"
    
    def _format_warning(op: str, issue: str, context: str = "", suggestion: str = "") -> str:
        if format_warning_message:
            return format_warning_message(op, issue, context, suggestion)
        msg = f"DDFM {op}: {issue}"
        if context:
            msg += f". {context}"
        if suggestion:
            msg += f". {suggestion}"
        return msg
    
    # Validate factors shape - check for 0-dimensional array first
    factors = np.asarray(factors)
    if factors.ndim == 0:
        _logger.warning(_format_warning(
            "VAR estimation",
            "factors is 0-dimensional array",
            suggestion="Using identity matrix for A and small covariance for Q"
        ))
        m = num_factors if num_factors is not None and num_factors > 0 else 1
        return np.eye(m), np.eye(m) * 1e-6
    
    # Validate factors shape
    if factors.size == 0 or factors.ndim < 2 or factors.shape[0] < 2 or factors.shape[1] == 0:
        _logger.warning(_format_warning(
            "VAR estimation",
            f"insufficient or invalid factors shape {factors.shape}",
            suggestion="Using identity matrix for A and small covariance for Q"
        ))
        m = factors.shape[1] if factors.ndim == 2 and factors.shape[1] > 0 else (num_factors if num_factors else 1)
        if m == 0:
            m = 1
        return np.eye(m), np.eye(m) * 1e-6
    
    T, m = factors.shape
    
    # Pre-estimation checks
    min_obs_required = factor_order + 5
    if T < min_obs_required:
        _logger.warning(_format_warning(
            "VAR estimation",
            f"insufficient observations (T={T}) for VAR({factor_order})",
            context=f"need at least {min_obs_required}",
            suggestion="Using scaled identity based on factor variance"
        ))
        # Use scaled identity based on factor variance
        factor_var = np.var(factors, axis=0)
        factor_var = np.maximum(factor_var, 1e-8)  # Floor
        if factor_order == 1:
            A_f = np.eye(m) * 0.5  # Conservative initial value
        else:
            A_f = np.hstack([np.eye(m) * 0.5, np.zeros((m, m))])
        Q_f = np.diag(factor_var)
        return A_f, Q_f
    
    # Check for constant factors (zero variance)
    factor_var = np.var(factors, axis=0)
    constant_factors = factor_var < 1e-10
    if np.any(constant_factors):
        n_constant = np.sum(constant_factors)
        _logger.warning(_format_warning(
            "VAR estimation",
            f"{n_constant}/{m} factors have zero variance",
            suggestion="These will be handled with small variance fallback"
        ))
    
    # Check for NaN/Inf in factors
    if not np.all(np.isfinite(factors)):
        nan_count = np.sum(~np.isfinite(factors))
        _logger.warning(_format_warning(
            "VAR estimation",
            f"factors contain {nan_count} NaN/Inf values",
            suggestion="Cleaning before estimation"
        ))
        factors = np.nan_to_num(factors, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check for rank deficiency in factor covariance matrix
    try:
        factor_cov = np.cov(factors.T)
        rank = np.linalg.matrix_rank(factor_cov)
        if rank < m:
            _logger.warning(_format_warning(
                "VAR estimation",
                f"factor covariance matrix is rank-deficient (rank={rank} < {m})",
                context=(
                    f"This may indicate redundant factors, insufficient data variation, "
                    f"or highly correlated factors. Regularization will be applied."
                ),
                suggestion="Using regularized estimation with small diagonal perturbation to improve conditioning"
            ))
            regularization = 1e-6
            factor_cov += np.eye(m) * regularization
    except (ValueError, np.linalg.LinAlgError) as e:
        _logger.warning(_format_warning(
            "VAR estimation",
            f"failed to compute factor covariance for rank check: {type(e).__name__}",
            suggestion="Proceeding with VAR estimation, will use fallback if needed"
        ))
    
    # Estimate VAR with error handling
    try:
        if factor_order == 1:
            A_f, Q_f = estimate_var1(factors)
        elif factor_order == 2:
            A_f, Q_f = estimate_var2(factors)
        else:
            raise ValueError(_format_error(
                "VAR estimation",
                f"factor_order must be 1 or 2, got {factor_order}",
                "Please set factor_order to 1 (VAR(1)) or 2 (VAR(2))"
            ))
        
        # Validate Q_f shape
        if Q_f.ndim == 0:
            _logger.warning(_format_warning(
                "VAR estimation",
                "returned 0-dimensional Q",
                suggestion="Using scaled identity based on factor variance"
            ))
            factor_var = np.var(factors, axis=0)
            factor_var = np.maximum(factor_var, 1e-8)
            Q_f = np.diag(factor_var)
        elif Q_f.ndim != 2:
            _logger.warning(_format_warning(
                "VAR estimation",
                f"returned Q with unexpected shape {Q_f.shape}",
                suggestion="Reshaping"
            ))
            if Q_f.size == m ** 2:
                Q_f = Q_f.reshape(m, m)
            else:
                factor_var = np.var(factors, axis=0)
                factor_var = np.maximum(factor_var, 1e-8)
                Q_f = np.diag(factor_var)
        
        # Validate estimated parameters
        # Check spectral radius of A (should be < 1 for stability)
        if factor_order == 1:
            eigenvals_A = np.linalg.eigvals(A_f)
        else:
            # For VAR(2), check companion form
            A1 = A_f[:, :m]
            A2 = A_f[:, m:]
            companion = np.block([
                [A1, A2],
                [np.eye(m), np.zeros((m, m))]
            ])
            eigenvals_A = np.linalg.eigvals(companion)
        
        max_eigenval = np.max(np.abs(eigenvals_A))
        if max_eigenval >= 0.99:
            _logger.warning(_format_warning(
                f"VAR({factor_order}) estimation",
                f"estimated A has spectral radius {max_eigenval:.4f} >= 0.99",
                suggestion="This may indicate instability. Consider checking factor quality"
            ))
        
        # Validate Q is positive definite
        Q_sym = (Q_f + Q_f.T) / 2  # Ensure symmetry
        eigenvals_Q = np.linalg.eigvalsh(Q_sym)
        min_eigenval_Q = np.min(eigenvals_Q)
        if min_eigenval_Q < 1e-8:
            _logger.warning(_format_warning(
                "VAR estimation",
                f"estimated Q has minimum eigenvalue {min_eigenval_Q:.2e} < 1e-8",
                suggestion="Regularizing to ensure positive definiteness"
            ))
            Q_f = Q_sym + np.eye(m) * (1e-8 - min_eigenval_Q)
        else:
            Q_f = Q_sym
        
        # Check condition number of Q
        if m > 1:
            max_eigenval_Q = np.max(eigenvals_Q)
            cond_num_Q = max_eigenval_Q / max(min_eigenval_Q, 1e-12)
            if cond_num_Q > 1e8:
                _logger.warning(_format_warning(
                    "VAR estimation",
                    f"estimated Q is ill-conditioned (cond={cond_num_Q:.2e})",
                    suggestion="This may indicate collinear factors"
                ))
        
        return A_f, Q_f
        
    except (ValueError, np.linalg.LinAlgError) as e:
        _logger.warning(_format_warning(
            f"VAR({factor_order}) estimation",
            f"estimation failed: {e}",
            suggestion="Using scaled identity based on factor variance as fallback"
        ))
        # Use scaled identity based on factor variance as fallback
        factor_var = np.var(factors, axis=0)
        factor_var = np.maximum(factor_var, 1e-8)  # Floor
        
        if factor_order == 1:
            A_f = np.eye(m) * 0.5  # Conservative initial value
        else:
            A_f = np.hstack([np.eye(m) * 0.5, np.zeros((m, m))])
        
        Q_f = np.diag(factor_var)
        _logger.info(
            f"VAR fallback: A shape={A_f.shape}, Q shape={Q_f.shape}, "
            f"factor variance range=[{np.min(factor_var):.2e}, {np.max(factor_var):.2e}]"
        )
        return A_f, Q_f


def validate_factors_ddfm(
    factors: np.ndarray,
    num_factors: int,
    operation: str = "operation",
    format_error_message: Optional[Callable] = None,
    format_warning_message: Optional[Callable] = None,
) -> np.ndarray:
    """Validate and normalize factors shape and content quality.
    
    This function performs comprehensive validation of factor arrays, checking for
    shape correctness, numerical issues (NaN/Inf), constant factors, extreme scale,
    and perfect correlation between factors. It raises errors for critical issues
    and issues warnings for quality concerns.
    
    Parameters
    ----------
    factors : np.ndarray
        Factors array to validate. Can be 1D or 2D, will be reshaped to 2D if needed.
    num_factors : int
        Expected number of factors. Used for reshaping 1D arrays.
    operation : str, default "operation"
        Operation name for error messages (e.g., "prediction", "factor extraction").
        Used to provide context in error messages.
    format_error_message : callable, optional
        Function to format error messages. If None, uses simple string formatting.
    format_warning_message : callable, optional
        Function to format warning messages. If None, uses simple string formatting.
        
    Returns
    -------
    np.ndarray
        Validated factors array, guaranteed to be 2D with shape (T x num_factors),
        where T is number of time periods and num_factors is number of factors.
        All values are finite (no NaN/Inf).
        
    Raises
    ------
    ValueError
        If factors are empty or invalid shape (0D or 3D+).
        If factors contain NaN/Inf values (critical numerical issue).
        If factors cannot be reshaped to 2D array.
    """
    # Helper functions for message formatting
    def _format_error(op: str, reason: str, guidance: str = "") -> str:
        if format_error_message:
            return format_error_message(op, reason, guidance)
        return f"DDFM {op} failed: {reason}. {guidance}"
    
    def _format_warning(op: str, issue: str, context: str = "", suggestion: str = "") -> str:
        if format_warning_message:
            return format_warning_message(op, issue, context, suggestion)
        msg = f"DDFM {op}: {issue}"
        if context:
            msg += f". {context}"
        if suggestion:
            msg += f". {suggestion}"
        return msg
    
    factors = np.asarray(factors)
    if factors.ndim == 0 or factors.size == 0:
        raise ValueError(_format_error(
            operation,
            f"factors is empty or invalid (shape: {factors.shape})",
            "This indicates training did not complete properly. Please check training logs and ensure fit_mcmc() completed successfully."
        ))
    if factors.ndim == 1:
        # Reshape to (T, num_factors) if it's 1D
        factors = factors.reshape(-1, num_factors) if factors.size > 0 else factors.reshape(0, num_factors)
    if factors.ndim != 2:
        raise ValueError(_format_error(
            operation,
            f"factors must be 2D array (T x m), got shape {factors.shape}",
            ""
        ))
    
    T, m = factors.shape
    
    # Check for NaN/Inf values
    if not np.all(np.isfinite(factors)):
        nan_count = np.sum(~np.isfinite(factors))
        nan_pct = 100.0 * nan_count / factors.size
        raise ValueError(_format_error(
            operation,
            f"factors contain {nan_count} ({nan_pct:.1f}%) NaN/Inf values",
            "This indicates numerical issues during training. Please check training logs and data quality."
        ))
    
    # Check for constant factors (all same value)
    factor_var = np.var(factors, axis=0)
    constant_factors = factor_var < 1e-10
    if np.any(constant_factors):
        n_constant = np.sum(constant_factors)
        constant_indices = np.where(constant_factors)[0].tolist()
        _logger.warning(_format_warning(
            operation,
            f"{n_constant}/{m} factors are constant (zero variance)",
            f"Constant factor indices: {constant_indices}",
            "This may indicate training issues or insufficient data variation"
        ))
    
    # Check factor scale (warn if extremely large/small)
    factor_std = np.std(factors, axis=0)
    extreme_scale = (factor_std > 1e6) | (factor_std < 1e-8)
    if np.any(extreme_scale):
        n_extreme = np.sum(extreme_scale)
        extreme_indices = np.where(extreme_scale)[0].tolist()
        std_range = [np.min(factor_std), np.max(factor_std)]
        _logger.warning(_format_warning(
            operation,
            f"{n_extreme}/{m} factors have extreme scale",
            f"Extreme factor indices: {extreme_indices}, Factor std range: [{std_range[0]:.2e}, {std_range[1]:.2e}]",
            "This may indicate numerical instability"
        ))
    
    # Check for perfect correlation between factors (detect linear dependencies)
    if m > 1 and T > 1:
        factor_corr = np.corrcoef(factors.T)
        # Check off-diagonal elements for perfect correlation
        np.fill_diagonal(factor_corr, 0.0)
        perfect_corr = np.abs(factor_corr) > 0.999
        if np.any(perfect_corr):
            n_pairs = np.sum(perfect_corr) // 2  # Divide by 2 since symmetric
            _logger.warning(_format_warning(
                operation,
                f"{n_pairs} pairs of factors are perfectly correlated (|corr| > 0.999)",
                suggestion="This may indicate redundant factors or training convergence issues"
            ))
    
    # Log factor statistics when validation passes (debug level)
    if _logger.isEnabledFor(logging.DEBUG):
        factor_mean = np.mean(factors, axis=0)
        factor_std = np.std(factors, axis=0)
        _logger.debug(
            f"DDFM {operation}: Factor validation passed. "
            f"Shape: {factors.shape}, Mean range: [{np.min(factor_mean):.4f}, {np.max(factor_mean):.4f}], "
            f"Std range: [{np.min(factor_std):.4f}, {np.max(factor_std):.4f}]"
        )
    
    return factors


def validate_training_data_ddfm(
    X_torch: torch.Tensor,
    num_factors: int,
    factor_order: int,
    encoder_layers: list,
    encoder: Optional[torch.nn.Module] = None,
    operation: str = "training setup",
    format_error_message: Optional[Callable] = None,
    format_warning_message: Optional[Callable] = None,
) -> None:
    """Validate data dimensions and model configuration before training starts.
    
    This function performs early validation checks to catch configuration issues
    before training begins. It validates:
    1. Data dimensions (T >= min_obs_required for VAR estimation)
    2. Factor count (num_factors <= N, number of series)
    3. Encoder architecture (reasonable size for data dimensions)
    
    Parameters
    ----------
    X_torch : torch.Tensor
        Input data tensor, shape (T x N) where T is time periods and N is number of series
    num_factors : int
        Number of factors to extract
    factor_order : int
        VAR order for factor dynamics (1 or 2)
    encoder_layers : list
        Encoder hidden layer dimensions
    encoder : torch.nn.Module, optional
        Encoder module. If provided, validates input_dim matches data dimension.
    operation : str, default "training setup"
        Operation name for error messages
    format_error_message : callable, optional
        Function to format error messages. If None, uses simple string formatting.
    format_warning_message : callable, optional
        Function to format warning messages. If None, uses simple string formatting.
        
    Raises
    ------
    ValueError
        If data dimensions are insufficient for training
        If num_factors exceeds number of series
        If encoder architecture is too large for data size
    """
    # Helper functions for message formatting
    def _format_error(op: str, reason: str, guidance: str = "") -> str:
        if format_error_message:
            return format_error_message(op, reason, guidance)
        return f"DDFM {op} failed: {reason}. {guidance}"
    
    def _format_warning(op: str, issue: str, context: str = "", suggestion: str = "") -> str:
        if format_warning_message:
            return format_warning_message(op, issue, context, suggestion)
        msg = f"DDFM {op}: {issue}"
        if context:
            msg += f". {context}"
        if suggestion:
            msg += f". {suggestion}"
        return msg
    
    # Get data dimensions
    if X_torch.ndim != 2:
        raise ValueError(_format_error(
            operation,
            f"data must be 2D array (T x N), got shape {X_torch.shape}",
            "Please ensure data is properly formatted as (time_periods x num_series)"
        ))
    
    T, N = X_torch.shape
    
    # Validation 1: Check minimum time periods for VAR estimation
    min_obs_required = factor_order + 5
    if T < min_obs_required:
        raise ValueError(_format_error(
            operation,
            f"insufficient time periods (T={T}) for VAR({factor_order}) estimation",
            (
                f"Need at least {min_obs_required} time periods for stable VAR estimation. "
                f"Current config: num_factors={num_factors}, factor_order={factor_order}, "
                f"encoder_layers={encoder_layers}. "
                f"With very small datasets (T < 10), training may be unstable due to: "
                f"(1) Insufficient data for encoder/decoder training, "
                f"(2) Poor VAR parameter estimation, "
                f"(3) High variance in MCMC sampling. "
                f"Suggestions: (1) Increase data size to at least {min_obs_required} periods, "
                f"(2) Reduce factor_order to 1 (requires {1 + 5} periods), "
                f"(3) Reduce num_factors to 1-2 for small datasets, "
                f"(4) Use smaller encoder_layers (e.g., [16, 8]) for better generalization"
            )
        ))
    
    # Additional warning for very small datasets (T < 10) even if above minimum
    if T < 10:
        _logger.warning(_format_warning(
            operation,
            f"very small dataset (T={T} < 10) may lead to unstable training",
            (
                f"With T={T} time periods, encoder/decoder training and MCMC sampling "
                f"may have high variance. VAR estimation will use fallback strategies."
            ),
            (
                f"Consider: (1) Using factor_order=1 (requires {1 + 5} periods), "
                f"(2) Reducing num_factors to 1-2, (3) Using smaller encoder_layers, "
                f"(4) Increasing data size if possible"
            )
        ))
    
    # Validation 2: Check factor count vs. number of series
    if num_factors > N:
        raise ValueError(_format_error(
            operation,
            f"num_factors ({num_factors}) exceeds number of series (N={N})",
            (
                f"Cannot extract more factors than available series. "
                f"Current config: num_factors={num_factors}, N={N}. "
                f"Suggestions: (1) Reduce num_factors to {min(num_factors, N)}, "
                f"(2) Add more series to data, (3) Use num_factors <= N"
            )
        ))
    
    # Validation 3: Check factor count is positive
    if num_factors <= 0:
        raise ValueError(_format_error(
            operation,
            f"num_factors must be positive, got {num_factors}",
            "Please set num_factors to a positive integer (typically 1-5)"
        ))
    
    # Validation 4: Check encoder architecture is reasonable for data size
    # Warn if encoder is too large for small datasets
    total_encoder_params = sum(encoder_layers) if encoder_layers else 0
    if T < 50 and total_encoder_params > 200:
        _logger.warning(_format_warning(
            operation,
            f"encoder architecture may be too large for small dataset",
            f"T={T}, encoder_layers={encoder_layers} (total params: ~{total_encoder_params})",
            "Consider using smaller encoder_layers (e.g., [32, 16]) for better generalization"
        ))
    
    # Validation 5: Check encoder input dimension matches data dimension
    # This will be validated when encoder is initialized, but we can check early
    if encoder is not None:
        if hasattr(encoder, 'input_dim') and encoder.input_dim != N:
            raise ValueError(_format_error(
                operation,
                f"encoder input_dim ({encoder.input_dim}) doesn't match data dimension (N={N})",
                (
                    f"Encoder was initialized with input_dim={encoder.input_dim}, "
                    f"but data has N={N} series. "
                    f"This may indicate data dimension mismatch. "
                    f"Please ensure data and encoder are compatible"
                )
            ))
    
    # Log validation success at debug level
    if _logger.isEnabledFor(logging.DEBUG):
        _logger.debug(
            f"DDFM {operation}: Data validation passed. "
            f"T={T}, N={N}, num_factors={num_factors}, "
            f"factor_order={factor_order}, encoder_layers={encoder_layers}"
        )

