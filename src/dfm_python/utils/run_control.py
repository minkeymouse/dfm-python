"""Run control of one training run: the ``max_time`` budget and the ``fit(callback=...)`` hook.

The estimators build a :class:`RunControl` in ``fit`` and pass it to their
engine's training loop, which calls :meth:`RunControl.step` after every
iteration (DFM: every E-step; DDFM: every MCMC iteration; iVDFM and AFM:
every epoch). ``step`` calls the callback with the iteration's info dict and
says whether training must stop at this boundary; the loop stops only where it
would otherwise continue, keeping the parameters a ``max_iter`` stop there
would keep.
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any, Optional

__all__ = ["RunControl"]


class RunControl:
    """The time budget and the per-iteration callback of one training run.

    The clock starts when the object is built.

    Parameters
    ----------
    model : str
        The estimator's class name (``info["model"]``).
    max_iter : int
        The iteration budget (``info["max_iter"]``: ``max_iter``, or
        ``max_epochs`` for iVDFM and AFM).
    objective_name : str
        Name of the objective the loop reports (``info["objective_name"]``).
    max_time : float, optional
        Training budget in seconds; None sets none.
    callback : callable, optional
        Called as ``callback(info)`` after every iteration; a true return value
        asks training to stop.
    """

    def __init__(
        self,
        model: str,
        max_iter: int,
        objective_name: str,
        max_time: Optional[float] = None,
        callback: Optional[Callable[[dict], Any]] = None,
    ) -> None:
        self.model = model
        self.max_iter = int(max_iter)
        self.objective_name = objective_name
        self.max_time = max_time
        self.callback = callback
        self._start = perf_counter()

    def step(self, iteration: int, objective: float, **extra: Any) -> Optional[str]:
        """Report iteration ``iteration`` (1-based); return ``"callback"`` or ``"max_time"`` to stop, else None.

        The callback receives ``{"model", "iteration", "max_iter", "elapsed",
        "objective", "objective_name"}`` plus ``extra``, with ``elapsed`` the
        seconds since the run started. ``"callback"``: the callback returned a
        true value. ``"max_time"``: the elapsed time, read after the callback
        returns, exceeds ``max_time``. An exception raised by the callback
        propagates.
        """
        if self.callback is not None:
            info = {
                "model": self.model,
                "iteration": int(iteration),
                "max_iter": self.max_iter,
                "elapsed": perf_counter() - self._start,
                "objective": float(objective),
                "objective_name": self.objective_name,
                **extra,
            }
            if self.callback(info):
                return "callback"
        if self.max_time is not None and perf_counter() - self._start > self.max_time:
            return "max_time"
        return None
