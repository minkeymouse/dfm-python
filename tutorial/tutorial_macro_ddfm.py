"""Tutorial: DDFM on the Korean macro panel.

Loads ``config/ddfm_macro.yaml`` (a ``model:`` section with the DDFM
constructor parameters and a ``data:`` section with the ``PanelData.from_wide``
arguments), fits the model, forecasts ``KOEQUIPTE`` 6 steps ahead in data
units, forecasts the factors, and saves and reloads the run. The other series
are the encoder's covariates (``exog``).

Run from the repository root (needs the ``yaml`` extra, or PyYAML)::

    python tutorial/tutorial_macro_ddfm.py
    python tutorial/tutorial_macro_ddfm.py --rows 120 --max-iter 1   # quick check

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

from dfm_python import DDFM, PanelData
from dfm_python.config import DDFMConfig, load_yaml

CONFIG = PROJECT_ROOT / "config" / "ddfm_macro.yaml"
DATA = PROJECT_ROOT / "data" / "macro.csv"
RUN_DIR = PROJECT_ROOT / "models" / "ddfm_macro"


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=CONFIG, help="YAML file with model: and data: sections")
    parser.add_argument("--rows", type=int, default=_env_int("DFM_TUTORIAL_ROWS"),
                        help="use only the last ROWS rows of the data (quick runs)")
    parser.add_argument("--max-iter", type=int, default=_env_int("DFM_TUTORIAL_MAX_ITER"),
                        help="override the YAML's max_iter (MCMC iterations)")
    parser.add_argument("--device", default=None, help="override the device ('auto', 'cpu', 'cuda')")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR, help="directory the fitted run is saved to")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print("DDFM tutorial: macro data")

    # 1. Config: model: holds the constructor parameters, data: the PanelData.from_wide arguments.
    sections = load_yaml(args.config)
    config = DDFMConfig.from_dict(sections["model"])
    overrides = {}
    if args.max_iter is not None:
        overrides["max_iter"] = args.max_iter
    if args.device is not None:
        overrides["device"] = args.device

    # 2. Data: weekly dates; the monthly series are sparse and are interpolated in fit.
    df = pd.read_csv(DATA)
    if args.rows is not None:
        df = df.iloc[-args.rows:]
    data = PanelData.from_wide(df, **sections["data"])
    print(f"   targets {list(data.series_names)}, {len(data.exog_names)} covariates, "
          f"{data.n_periods} dates {data.index[0].date()} .. {data.index[-1].date()}")

    # 3. Fit: pre-training, MCMC denoising and the linear state space on the factors.
    model = DDFM.from_config(config).set_params(**overrides)
    print(f"   {model!r}")
    model.fit(data)
    print(f"   trained {model.n_iter_} MCMC iteration(s), converged={model.converged_}")
    print(f"   factors_ {model.factors_.shape}, loadings result_.C {model.result_.C.shape} (series x factors)")

    # 4. Forecast the target (data units) and the factors.
    forecast = model.predict(6)
    print("\n   6-step forecast (data units):")
    print(forecast.to_string(float_format=lambda v: f"{v:.4f}"))
    if not np.isfinite(forecast.to_numpy()).all():
        print("   WARNING: the forecast has non-finite values")
    factors = model.predict_factors(6)
    print(f"\n   factor forecast {factors.shape}, columns {list(factors.columns)}")

    # 5. Save the run and load it back without the data.
    model.save(args.run_dir, overwrite=True)
    loaded = DDFM.load(args.run_dir)
    assert loaded.predict(6).equals(forecast)
    print(f"\n   saved to {args.run_dir} and reloaded; the reloaded forecast is identical")


if __name__ == "__main__":
    main()
