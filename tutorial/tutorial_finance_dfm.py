"""Tutorial: DFM on daily financial indicators with four blocks of factors.

Loads ``config/dfm_finance.yaml`` (a ``model:`` section with the DFM
constructor parameters, four blocks of two factors each, and a ``data:``
section with the ``PanelData.from_wide`` arguments), fits the model by EM on
the last 1000 days of ``data/finance.csv``, forecasts the 27 series 6 days
ahead in data units, forecasts the factors, and saves and reloads the run.

Run from the repository root (needs the ``yaml`` extra, or PyYAML)::

    python tutorial/tutorial_finance_dfm.py
    python tutorial/tutorial_finance_dfm.py --rows 120 --max-iter 1   # quick check

``--rows`` and ``--max-iter`` default to the environment variables
``DFM_TUTORIAL_ROWS`` and ``DFM_TUTORIAL_MAX_ITER``; unset, the last 1000 rows
and the YAML's ``max_iter`` are used.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # run from a clone without installing

import numpy as np
import pandas as pd

from dfm_python import DFM, PanelData
from dfm_python.config import DFMConfig, load_yaml

CONFIG = PROJECT_ROOT / "config" / "dfm_finance.yaml"
DATA = PROJECT_ROOT / "data" / "finance.csv"
RUN_DIR = PROJECT_ROOT / "models" / "dfm_finance"
DEFAULT_ROWS = 1000  # the most recent rows, for a run of reasonable length


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=CONFIG, help="YAML file with model: and data: sections")
    parser.add_argument("--rows", type=int, default=_env_int("DFM_TUTORIAL_ROWS") or DEFAULT_ROWS,
                        help=f"use only the last ROWS rows of the data (default {DEFAULT_ROWS})")
    parser.add_argument("--max-iter", type=int, default=_env_int("DFM_TUTORIAL_MAX_ITER"),
                        help="override the YAML's max_iter (EM iterations)")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR, help="directory the fitted run is saved to")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print("DFM tutorial: finance data")

    # 1. Config: model: holds the constructor parameters, data: the PanelData.from_wide arguments.
    sections = load_yaml(args.config)
    config = DFMConfig.from_dict(sections["model"])
    overrides = {}
    if args.max_iter is not None:
        overrides["max_iter"] = args.max_iter

    # 2. Data: finance.csv has no dates (only date_id), so daily dates from 1980-01-01 are added.
    df = pd.read_csv(DATA).iloc[-args.rows:].reset_index(drop=True)
    df["date"] = pd.date_range("1980-01-01", periods=len(df), freq="D")
    data = PanelData.from_wide(df, **sections["data"])
    print(f"   {len(data.series_names)} series (the blocks' series), {data.n_periods} days "
          f"{data.index[0].date()} .. {data.index[-1].date()}")

    # 3. Fit: initialization, then EM with the Kalman filter and smoother.
    model = DFM.from_config(config).set_params(**overrides)
    print(f"   {model!r}")
    model.fit(data)
    print(f"   EM ran {model.n_iter_} iteration(s), converged={model.converged_}, loglik={model.loglik_:.4f}")
    print(f"   {model.n_factors_} factors in blocks {list(model.blocks)}; factors_ {model.factors_.shape}, "
          f"states_ {model.states_.shape} (the factors, their lags and idiosyncratic states)")

    # 4. Forecast the series (data units, the data's column order) and the factors.
    forecast = model.predict(6)
    print("\n   6-step forecast (data units):")
    print(forecast.to_string(float_format=lambda v: f"{v:.4f}"))
    if not np.isfinite(forecast.to_numpy()).all():
        print("   WARNING: the forecast has non-finite values")
    factors = model.predict_factors(6)
    print(f"\n   factor forecast {factors.shape}, columns {list(factors.columns[:3])} ...")

    # 5. Save the run and load it back without the data.
    model.save(args.run_dir, overwrite=True)
    loaded = DFM.load(args.run_dir)
    assert loaded.predict(6).equals(forecast)
    print(f"\n   saved to {args.run_dir} and reloaded; the reloaded forecast is identical")


if __name__ == "__main__":
    main()
