"""iVDFM-specific logging utilities.

This module provides iVDFM-specific logging classes and functions.
"""

from .train_logger import BaseTrainLogger
from .inference_logger import BaseInferenceLogger


class iVDFMTrainLogger(BaseTrainLogger):
    """Training logger for iVDFM models (variational inference)."""
    
    def __init__(self, verbose: bool = True):
        """Initialize iVDFM training logger."""
        super().__init__(model_name="iVDFM", verbose=verbose)
    
    def log_epoch(
        self,
        epoch: int,
        elbo: float = None,
        recon_loss: float = None,
        kl_loss: float = None,
        learning_rate: float = None,
        **kwargs
    ) -> None:
        """Log training epoch information for iVDFM.
        
        Parameters
        ----------
        epoch : int
            Current epoch number
        elbo : float, optional
            ELBO value (negative, to minimize)
        recon_loss : float, optional
            Reconstruction loss component
        kl_loss : float, optional
            KL divergence component
        learning_rate : float, optional
            Current learning rate
        **kwargs
            Additional metrics to log
        """
        self.epochs = epoch
        
        # Store metrics in history
        metrics = {
            "epoch": epoch,
            "elbo": elbo,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "learning_rate": learning_rate
        }
        metrics.update(kwargs)
        self.metrics_history.append(metrics)
        
        if self.verbose:
            msg = f"Epoch {epoch:4d}"
            if elbo is not None:
                msg += f" | ELBO: {elbo:12.6f}"
            if recon_loss is not None:
                msg += f" | Recon: {recon_loss:12.6f}"
            if kl_loss is not None:
                msg += f" | KL: {kl_loss:12.6f}"
            if learning_rate is not None:
                msg += f" | LR: {learning_rate:.6e}"
            
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    msg += f" | {key}: {value:.6f}"
                else:
                    msg += f" | {key}: {value}"
            
            from .logger import get_logger
            _logger = get_logger(__name__)
            _logger.info(msg)
        else:
            # Less verbose: log every 10 epochs
            if epoch % 10 == 0 or epoch == 1:
                from .logger import get_logger
                _logger = get_logger(__name__)
                msg = f"Epoch {epoch:4d}"
                if elbo is not None:
                    msg += f" | ELBO: {elbo:12.6f}"
                _logger.info(msg)


class iVDFMInferenceLogger(BaseInferenceLogger):
    """Inference logger for iVDFM models."""
    
    def __init__(self, verbose: bool = True):
        """Initialize iVDFM inference logger."""
        super().__init__(model_name="iVDFM", verbose=verbose)
