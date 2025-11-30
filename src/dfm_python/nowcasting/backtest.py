"""Backtest result classes for nowcasting evaluation."""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from .nowcast import NowcastResult
    from .news import NewsDecompResult


@dataclass
class BacktestResult:
    """Result from backtest evaluation of nowcasting model.
    
    This dataclass contains all information from a pseudo real-time backtest,
    including nowcasts at different view dates, news decomposition between steps,
    and evaluation metrics.
    """
    target_series: str
    target_date: datetime
    backward_steps: int
    higher_freq: bool
    backward_freq: str
    view_list: List  # List[DataView] - avoiding circular import
    nowcast_results: List["NowcastResult"]
    news_results: List[Optional["NewsDecompResult"]]
    actual_values: np.ndarray
    errors: np.ndarray
    mae_per_step: np.ndarray
    mse_per_step: np.ndarray
    rmse_per_step: np.ndarray
    overall_mae: Optional[float]
    overall_rmse: Optional[float]
    overall_mse: Optional[float]
    failed_steps: List[int]
    
    def plot(self, save_path: Optional[str] = None, show: bool = True):
        """Plot backtest results."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot 1: Nowcast values vs actual
            ax1 = axes[0]
            view_dates = [r.view_date for r in self.nowcast_results]
            nowcast_values = [r.nowcast_value for r in self.nowcast_results]
            
            ax1.plot(view_dates, nowcast_values, 'o-', label='Nowcast', color='blue')
            if not np.all(np.isnan(self.actual_values)):
                ax1.axhline(y=self.actual_values[0], color='red', linestyle='--', label='Actual')
            ax1.set_xlabel('View Date')
            ax1.set_ylabel('Value')
            ax1.set_title(f'Backtest Results: {self.target_series} at {self.target_date}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Plot 2: Error metrics
            ax2 = axes[1]
            steps = range(self.backward_steps)
            ax2.plot(steps, self.rmse_per_step, 'o-', label='RMSE', color='green')
            ax2.set_xlabel('Backward Step')
            ax2.set_ylabel('Error')
            ax2.set_title('Error Metrics per Step')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close()
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    def plot_trajectory(self, save_path: Optional[str] = None, show: bool = True):
        """Plot nowcast trajectory over backward steps."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            view_dates = [r.view_date for r in self.nowcast_results]
            nowcast_values = [r.nowcast_value for r in self.nowcast_results]
            
            ax.plot(view_dates, nowcast_values, 'o-', label='Nowcast Trajectory', color='blue', linewidth=2, markersize=8)
            
            if not np.all(np.isnan(self.actual_values)):
                ax.axhline(y=self.actual_values[0], color='red', linestyle='--', linewidth=2, label='Actual')
            
            ax.set_xlabel('View Date', fontsize=12)
            ax.set_ylabel('Nowcast Value', fontsize=12)
            ax.set_title(f'Nowcast Trajectory: {self.target_series} at {self.target_date}', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close()
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
