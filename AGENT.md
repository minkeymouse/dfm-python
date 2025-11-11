### dfm-python Agent Guide

Purpose: Guide the agent to iteratively transform the FRBNY Nowcasting MATLAB codebase into a clean, generalized, OOP Python package with robust tests, tutorials, and documentation — without getting stuck, timing out, or derailing the iteration loop.

### Success Criteria
- Functional package: DFM fit/nowcast/forecast/tutorials/tests pass in .venv.
- Robust numerics: no zero factors/loadings, stable Kalman filtering/smoothing, PSD covariances, real-symmetric matrices, stable EM convergence.
- Code quality: consistent naming, clear docstrings, consolidated utilities, reduced redundancies.
- Documentation: README/CHANGELOG current; tutorials are accurate and runnable quickly.
- Release hygiene: versioning in pyproject.toml and src/dfm_python/__init__.py aligned.

### Iteration Workflow (7 Steps)
Each iteration should complete all steps, then continue looping. Keep actions small, reversible, and aligned to the success criteria.

1) Package assessment (현 상황 파악 및 개선점 탐색)
- Scan src/dfm_python/core/{em.py, kalman.py, numeric.py}, config, tests, tutorials.
- Identify high-impact improvements (e.g., zero factor/loadings, covariance stability, circular imports).
- Compare logic with Nowcasting (MATLAB) where numerics differ.

2) Stage identification and planning (스테이지 식별 및 계획)
- Choose stage: implementation / test / refactor / docs.
- Draft a minimal plan with 2–5 concrete tasks for this iteration only.

3) Create actionable tasks (태스크 생성)
- Small, independent tasks; timeboxed; reversible.
- Examples:
  - Fix init_conditions Block_Global loadings scaling and A[0,0] floor.
  - Harden _compute_covariance_safe pairwise branch.
  - Centralize numeric stabilization (_ensure_real, _ensure_covariance_stable, _ensure_innovation_variance_minimum).
  - Add/adjust unit tests for numeric stability and tutorials.

4) Execute tasks (태스크 작업)
- Apply focused edits; keep diffs minimal.
- Prefer utility consolidation over copy-paste logic.
- After each set of edits, run only the most relevant tests to validate quickly.

5) Refactor & code hygiene (리팩토링/정리/일관성/효율화)
- Remove redundancies; unify naming; improve docstrings.
- Consolidate numerics into core/numeric.py where sensible.
- Keep interfaces stable; move dead code to trash/ rather than deleting.

6) Finalize & clean up (마무리)
- Ensure build/test passes; tutorials still run in a short window.
- Clean temporary artifacts.

7) Update TODO.md
- Mark completed items, add follow-ups, keep it concise and actionable.

### Operational Policies
- Always run commands with a timeout:
  - Use `timeout 30m <command>` (the workflow provides `timeout_cmd`).
  - Prefer smaller walltime by scoping tests (e.g., `-k "numeric or dfm"`).
- Non-interactive flags always:
  - pip: `--no-input`, `-y` where appropriate; pytest: non-interactive; CLI: default args or flags to avoid prompts.
- Keep iterations continuous:
  - Never block the loop on long tasks; split work into smaller chunks.
  - If a command risks running long, reduce data/time range or iteration counts (e.g., EM `max_iter`, tutorial sample windows).
- Resilience:
  - Retry cursor-agent calls on rate limits/transient errors (handled by workflow).
  - When a test intermittently fails due to numeric tolerance, tighten stabilization or relax assertion tolerance with justification.

### Coding Rules
- Scope: Only edit files inside the dfm-python project. Move removed files to `trash/` (don’t delete).
- Naming: descriptive, full words, consistent across modules; avoid abbreviations.
- Docstrings: explain parameters/returns, non-obvious rationale, key invariants; be concise.
- Control flow: early returns, guard clauses; avoid deep nesting; avoid unnecessary try/except.
- Comments: only for rationale, tricky invariants, or caveats; do not narrate edits.
- Formatting: respect existing style; avoid massive reformatting unrelated to the task.

### Numeric Stability & DFM-Specific Guidance
- Covariance:
  - Use `_compute_covariance_safe` with `pairwise_complete=True` for sparse/mixed-frequency; verify shape `(N, N)`.
  - Regularize negative eigenvalues minimally; keep covariance PSD.
- Real-symmetric matrices:
  - Use `_ensure_real` and `_ensure_real_and_symmetric` in Kalman steps to prevent ComplexWarning and asymmetry.
- Innovation variances:
  - Enforce `_ensure_innovation_variance_minimum(Q, min_variance=1e-8)` post-cleaning.
- Init conditions (em.py):
  - Global block: use pairwise covariance; relax data completeness; median-impute NaNs for covariance; ensure first eigenvalue floor; scale loadings if too small; floor A[0,0].
- Kalman filter/smoother:
  - Stabilize covariance updates with `_ensure_covariance_stable`.
  - Ensure log-likelihood is real and finite; fallback to `-inf` only if necessary.
- Circular imports:
  - Use lazy import helpers (e.g., `_get_numeric_utils()`) to break cycles.

### Testing Strategy
- Quick loops:
  - `pytest -k "numeric or dfm" -q`
  - Scope specific tests while iterating (e.g., `test_numeric.py::test_compute_covariance_safe_pairwise_complete`).
- Tutorials:
  - Run with short sample windows, lower `max_iter`, looser thresholds to keep runtime low.
- Before finalizing:
  - `pytest -q` within `.venv`; ensure all tests pass; note any skipped tests explicitly.

### Testing & Tutorials Policy (No Cheating)
- Mandatory: All tests and all tutorials must pass and produce theoretically sound results, consistent with the DFM framework (and, where applicable, the MATLAB mother code).
- Forbidden: temporal fixes, monkey patches, hard-coded hacks, and data-specific constants introduced solely to make tests pass or to force outputs.
- Do not weaken or bypass tests/tutorials to achieve “green”; fix the underlying implementation instead.
- Any relaxation of numerical tolerances must be minimal, justified, and documented (why, where, and the theoretical basis).
- Keep tutorials fast but faithful: shortening the time window or iterations is allowed for speed, but the math and logic must be identical to the library’s code path.
- Do not modify tests to align with a broken implementation. Add regression tests for fixed bugs to prevent recurrence.
- Prefer public APIs in tests and tutorials; avoid monkeypatching internal functions.

### Tutorials & Docs
- Keep `tutorial/basic_tutorial.py` and `tutorial/hydra_tutorial.py` runnable with short defaults.
- Ensure README shows accurate quick-starts, minimal viable examples, and known tips for speed.
- Update CHANGELOG with clear bullet points per version bump.

### Release Hygiene
- Bump version in `pyproject.toml` and `src/dfm_python/__init__.py`.
- Verify `pip install -e .` success within `.venv`.
- Sanity test: import, small fit, quick nowcast/forecast, essential plots.

### Anti-Patterns to Avoid
- Long-running experiments or heavy data runs inside iterations.
- Unscoped refactors that span many modules at once.
- Large new features without tests.
- Deleting files outright (move to `trash/`).

### Common Pitfalls and Fixes
- Zero factors/loadings:
  - Check Block_Global initialization; ensure eigenvalue/loadings floors; inspect Q and A floors.
- Covariance dimension mismatch:
  - Ensure pairwise computation returns `(N, N)`; validate shapes after `np.cov`.
- Complex warnings:
  - Use `_ensure_real_and_symmetric` immediately after numeric ops likely to introduce tiny imaginaries.
- Convergence stalls:
  - Floor Q diagonals; stabilize AR coefficients; clip AR if needed; consider damping if likelihood decreases.

### Execution Notes
- Always work in `.venv`:
  - `source .venv/bin/activate`
  - `pip install -e . --no-deps --force-reinstall` when necessary.
- Prefer small, frequent commits (if VCS is used) aligned with the 7-step loop.
- Keep iteration logs readable; summarize concise progress and next steps.


