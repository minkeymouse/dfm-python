# Development

## Setup

The `dev` dependency group holds pytest, ruff, the build tools and the
optional packages the tests import.

```bash
git clone https://github.com/minkeymouse/dfm-python
cd dfm-python
uv sync --group dev --all-extras    # or: pip install -e ".[all]" --group dev; --all-extras brings torch
uv run pytest                       # test suite (src/test)
uv run ruff check                   # lint; also run by pre-commit and CI
uvx pre-commit install              # optional: ruff and the file checks on every commit
```

CI (`.github/workflows/ci.yml`) runs the test suite on Python 3.10 to 3.13,
the DFM tests and `src/test/test_optional_torch.py` in an environment
without torch (`uv sync --group dev`, no extras), every `tutorial/` script on
truncated data, `ruff check` and a package build with `twine check`.

DFM must keep working without torch: the core package imports torch only
inside the torch models' `fit` and `load`, the `dfm_python.utils.optional`
helpers (`require_torch`, the lazy `torch` stand-in, `lazy_torch_exports`)
defer it, and `test_optional_torch.py` checks it in an interpreter where
torch and einops cannot be imported.

## Tests

| Location | Contents |
|----------|----------|
| `src/test/contract/` | `dfm_python.testing.check_estimator(model)` on every entry of `dfm_python.registry.all_estimators()`: tags, pure construction, `get_params`/`set_params`/`clone`, config round-trips, unfitted errors, deterministic `fit`, output shapes, purity of `predict`, capability errors, device, silence, `update`, save/load round-trip (a run saved on CUDA loads on CPU) |
| `src/test/parity/` | golden tests (below) |
| `src/test/ssm/test_estep_oracle.py`, `src/test/functional/test_em_oracle.py`, `src/test/functional/test_mstep_qfunction.py` | DFM E-step, EM monotonicity and M-step against a reference Kalman filter that does not import the package (`src/test/ssm/_reference_kf.py`), and against statsmodels' `KalmanSmoother` when installed |
| `src/test/test_readme.py` | runs every Python code block of `README.md` and `docs/*.md` (except `docs/design/`) |
| `src/test/reference/` | DFM against statsmodels' `DynamicFactorMQ` on `data/macro.csv` (about 50 s; measurements in its README) |
| `src/test/test_optional_torch.py` | the DFM workflow and the torch models' `ImportError` without torch |
| `src/test/test_robustness.py`, `src/test/test_evaluation.py` | degenerate and infinite inputs; the walk-forward `backtest` |
| the rest of `src/test/` | unit tests by subpackage |

`test_readme.py` runs each `##` or `###` section's Python blocks in order in
one fresh namespace, from a temporary directory that links the repository's
`config/` and `data/`. Keep a page's examples self-contained per section and
fast.

## Golden tests

The fixtures in `src/test/parity/fixtures/` pin the numbers each engine
produces for a given dataset and hyperparameters (`<case>.npz` holds the
outputs, `<case>.json` the git commit, platform fingerprint, data recipe and
hyperparameters). Every run happens with one BLAS thread, one torch thread
and deterministic torch algorithms.

- On the machine whose fingerprint (CPU model and SIMD level, Python minor
  version, numpy, torch and BLAS builds) the manifests record, arrays must be
  bitwise equal.
- On any other machine the comparison falls back to `assert_allclose`
  (float32 `rtol=1e-2, atol=1e-5`; float64 `rtol=5e-3, atol=1e-6`) and a
  `GoldenToleranceWarning` says so. CI runs on this path.
- Regenerate fixtures only when a numerical change is intended:
  `uv run python src/test/parity/make_golden.py` (all cases, or named ones).
  Say why in the pull request and in `CHANGELOG.md`.

## Repository layout

```
src/dfm_python/
  base.py      BaseFactorModel: the estimator base (tags, save/load template)
  models/      dfm/, ddfm/, ivdfm/, afm/: estimator.py (public) and _engine.py (private)
  dataset/     PanelData, and the private data adapters of the four models
  config/      constructor mirrors (DFMConfig, ..., AFMConfig), load_yaml, results
  io/          the run directory: layout, codecs, scaler files, dfm_python.load
  ssm/         state-space form and Kalman filtering
  functional/  EM algorithm
  numeric/, layer/, utils/   shared numerics, network layers, helpers, errors, seeding
  logger/      logger.py (set_verbosity) and progress.py (fit-scoped progress)
  tuning/      tune_hyperparameters (iVDFM, optuna)
  registry.py, testing.py    all_estimators() and check_estimator()
src/test/      pytest suite (contract/ and parity/ hold the contract and golden tests)
config/        example YAML configurations (model: and data: sections)
data/          example datasets used by the tutorials
tutorial/      end-to-end scripts per model and dataset
docs/          user guide, models, migration, this page, citing; docs/design/ holds the API design specification
```

## Contributing

Bug reports and pull requests go to
[GitHub](https://github.com/minkeymouse/dfm-python/issues); include a
minimal example for bugs. A pull request should pass `uv run pytest` and
`uv run ruff check`. The API design is specified in
[docs/design/api-redesign.md](https://github.com/minkeymouse/dfm-python/blob/main/docs/design/api-redesign.md).
Security problems go through
[SECURITY.md](https://github.com/minkeymouse/dfm-python/blob/main/SECURITY.md),
not a public issue.

## Releases

`.github/workflows/publish.yml` builds the sdist and wheel with `uv build`,
checks them with `twine check` and uploads them to PyPI with Trusted
Publishing (no API token is stored in the repository). Steps:

1. Set `version` in `pyproject.toml` and `CITATION.cff`, and add a
   `## <version>` section to `CHANGELOG.md` (the workflow copies it into the
   GitHub Release notes).
2. Start the workflow in one of three ways: push a `v<version>` tag, publish
   a GitHub Release with that tag, or run the workflow manually from `main`
   (Actions, Publish to PyPI, Run workflow). A manual run on another branch
   is skipped.
3. A tag that does not match `pyproject.toml` fails the build. A manual run
   creates the `v<version>` tag and the GitHub Release after the upload.
   If the tag push and the release both fire for one version, the second
   upload is skipped.

Every action in the workflows is pinned to a commit SHA and Dependabot keeps
the pins current.
