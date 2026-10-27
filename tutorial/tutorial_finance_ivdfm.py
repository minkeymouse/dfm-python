"""Tutorial: iVDFM on the finance panel.

Loads ``config/ivdfm_finance.yaml`` (a ``model:`` section with the iVDFM
constructor parameters and a ``data:`` section with the ``PanelData.from_wide``
arguments), fits the model, forecasts ``market_forward_excess_returns`` 6 steps
ahead in data units, forecasts the factors, and saves and reloads the run.

Run from the repository root (needs the ``yaml`` extra, or PyYAML)::

    python tutorial/tutorial_finance_ivdfm.py
    python tutorial/tutorial_finance_ivdfm.py --rows 400 --max-epochs 1   # quick check

``--rows`` and ``--max-epochs`` default to the environment variables
``DFM_TUTORIAL_ROWS`` and ``DFM_TUTORIAL_MAX_EPOCHS``; unset, the whole file
and the YAML's ``max_epochs`` are used.
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

from dfm_python import PanelData, iVDFM
from dfm_python.config import iVDFMConfig, load_yaml

CONFIG = PROJECT_ROOT / "config" / "ivdfm_finance.yaml"
DATA = PROJECT_ROOT / "data" / "finance.csv"
RUN_DIR = PROJECT_ROOT / "models" / "ivdfm_finance"


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=CONFIG, help="YAML file with model: and data: sections")
    parser.add_argument("--rows", type=int, default=_env_int("DFM_TUTORIAL_ROWS"),
                        help="use only the last ROWS rows of the data (quick runs)")
    parser.add_argument("--max-epochs", type=int, default=_env_int("DFM_TUTORIAL_MAX_EPOCHS"),
                        help="override the YAML's max_epochs")
    parser.add_argument("--device", default=None, help="override the device ('auto', 'cpu', 'cuda')")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR, help="directory the fitted run is saved to")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print("iVDFM tutorial: finance data")

    # 1. Config: model: holds the constructor parameters, data: the PanelData.from_wide arguments.
    sections = load_yaml(args.config)
    config = iVDFMConfig.from_dict(sections["model"])
    overrides = {}
    if args.max_epochs is not None:
        overrides["max_epochs"] = args.max_epochs
    if args.device is not None:
        overrides["device"] = args.device

    # 2. Data: finance.csv has no dates (only date_id), so give the rows daily dates from 1980-01-01.
    df = pd.read_csv(DATA)
    df["date"] = pd.date_range("1980-01-01", periods=len(df), freq="D")
    if args.rows is not None:
        df = df.iloc[-args.rows:]
    data = PanelData.from_wide(df, **sections["data"])
    print(f"   targets {list(data.series_names)}, {data.n_periods} dates "
          f"{data.index[0].date()} .. {data.index[-1].date()}")

    # 3. Fit.
    model = iVDFM.from_config(config).set_params(**overrides)
    print(f"   {model!r}")
    model.fit(data)
    print(f"   trained {model.n_iter_} epoch(s), converged={model.converged_}, "
          f"ELBO={model.result_.training_elbo:.4f}")
    print(f"   window={model.window_}, stride={model.stride_}, factors_ {model.factors_.shape}")

    # 4. Forecast the targets (data units) and the factors.
    forecast = model.predict(6)
    print("\n   6-step forecast (data units):")
    print(forecast.to_string(float_format=lambda v: f"{v:.4f}"))
    if not np.isfinite(forecast.to_numpy()).all():
        print("   WARNING: the forecast has non-finite values")
    factors = model.predict_factors(6)
    print(f"\n   factor forecast {factors.shape}, columns {list(factors.columns)}")

    # 5. Save the run and load it back without the data.
    model.save(args.run_dir, overwrite=True)
    loaded = iVDFM.load(args.run_dir)
    assert loaded.predict(6).equals(forecast)
    print(f"\n   saved to {args.run_dir} and reloaded; the reloaded forecast is identical")


if __name__ == "__main__":
    main()
