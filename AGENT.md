DFM Python – Agent Operational Guide

Purpose
- This project is the Python successor to the Nowcasting MATLAB package. Your mission is to finalize a robust, class-oriented DFM implementation that is generic, reliable, and easy to use.

Protected assets
- Do not delete, move, or rename:
  - src/test (all test modules)
  - tutorial/basic_tutorial.py
  - Nowcasting/ reference materials
- Do not move working code to trash/.

Functional success criteria (all required, prioritized)
1) Class-oriented Python implementation works for both DictConfig (Hydra) and Spec + class configs.
2) Factors/results are plausible and numerically stable (no complex numbers; sensible convergence).
3) Generic Block DFM: runs with any properly formatted CSV.
4) Full pipeline: initialization, Kalman filter/smoother, EM, nowcasting, forecasting are correct.
5) APIs for visualization, results access, nowcasting, forecasting behave like Nowcast MATLAB where appropriate.
6) Generalized clock with mixed frequency; slower series use tent weights across manageable gaps.
7) Missing data handled robustly (aligned with Nowcast behavior).
8) Frequency guardrails: any series faster than the clock raises a clear error.
9) Generic naming/logic/patterns (no overengineering).
10) Organized structure ≤ 20 Python files under src/ (src/dfm_python + src/test), never reducing functionality or tests to meet this.

Iteration policy
- Accuracy over speed; prefer minimal, surgical changes.
- Every iteration must:
  - Run pytest on all test modules (src/test).
  - Run tutorial/basic_tutorial.py with a small max_iter.
  - Only proceed if both pass; otherwise fix or back out changes.
- Keep file count under src/ ≤ 20; do not increase; only consolidate if safe and beneficial.

Priorities
- Fix import/layout mismatches first (e.g., dfm_python.core/data vs current structure).
- Stabilize EM and KF numerics (min Q diagonal > 0, symmetry/PSD safeguards, convergence checks).
- Enforce frequency/clock rules; implement tent weights for slower series.
- Ensure missing data handling is robust and consistent.
- Maintain API parity with MATLAB where it improves usability.

Acceptance checks (per iteration)
- Unit tests pass; tutorial completes without error.
- No complex-valued parameters; Q diagonal ≥ 1e-8; shapes consistent; factors reasonable.
- No deletions/moves in src/test or tutorial/basic_tutorial.py.
- File count not increased; no new files unless merged into existing modules.

Notes
- Reference Nowcasting/ for structure and behavior patterns.
- Avoid overengineering; keep code clear, generic, and maintainable.
