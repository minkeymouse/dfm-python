"""Tutorial: DFM on the Korean macro panel (weekly clock, monthly series).

Loads ``config/dfm_macro.yaml`` (a ``model:`` section with the DFM constructor
parameters and a ``data:`` section with the ``PanelData.from_wide`` arguments,
including each series' frequency), fits the model by EM, forecasts the 14
series 6 weeks ahead in data units (monthly series included, at every week),
forecasts the factors, and saves and reloads the run.

Run from the repository root (needs the ``yaml`` extra, or PyYAML)::

    python tutorial/tutorial_macro_dfm.py
    python tutorial/tutorial_macro_dfm.py --rows 400 --max-iter 2   # quick check

``--rows`` and ``--max-iter`` default to the environment variables
``DFM_TUTORIAL_ROWS`` and ``DFM_TUTORIAL_MAX_ITER``; unset, the whole file and
the YAML's ``max_iter`` are used.
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

CONFIG = PROJECT_ROOT / "config" / "dfm_macro.yaml"
DATA = PROJECT_ROOT / "data" / "macro.csv"
RUN_DIR = PROJECT_ROOT / "models" / "dfm_macro"


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=CONFIG, help="YAML file with model: and data: sections")
    parser.add_argument("--rows", type=int, default=_env_int("DFM_TUTORIAL_ROWS"),
                        help="use only the last ROWS rows of the data (quick runs)")
    parser.add_argument("--max-iter", type=int, default=_env_int("DFM_TUTORIAL_MAX_ITER"),
                        help="override the YAML's max_iter (EM iterations)")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR, help="directory the fitted run is saved to")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print("DFM tutorial: macro data (weekly clock, monthly series)")

    # 1. Config: model: holds the constructor parameters, data: the PanelData.from_wide arguments.
    sections = load_yaml(args.config)
    config = DFMConfig.from_dict(sections["model"])
    overrides = {}
    if args.max_iter is not None:
        overrides["max_iter"] = args.max_iter

    # 2. Data: one row per week; a monthly series is NaN on the weeks it is not observed.
    df = pd.read_csv(DATA)
    if args.rows is not None:
        df = df.iloc[-args.rows:]
    data = PanelData.from_wide(df, **sections["data"])
    monthly = [name for name, code in data.freq.items() if code == "m"]
    print(f"   {len(data.series_names)} series ({len(monthly)} monthly), {data.n_timesteps} weeks "
          f"{data.index[0].date()} .. {data.index[-1].date()}")

    # 3. Fit: initialization, then EM with the Kalman filter and smoother.
    model = DFM.from_config(config).set_params(**overrides)
    print(f"   {model!r}")
    model.fit(data)
    print(f"   EM ran {model.n_iter_} iteration(s), converged={model.converged_}, loglik={model.loglik_:.4f}")
    print(f"   factors_ {model.factors_.shape} (the factors), states_ {model.states_.shape} (the factors, their lags "
          "and idiosyncratic states)")

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
