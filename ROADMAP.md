# ROADMAP.md — Project Milestones

Track progress by tagging tasks: `[analog]`, `[ML]`, `[layout]`, `[infra]`.

---

## Milestone 0 — Repository Skeleton ✅
*Goal: Establish a working project structure that any engineer can clone and run.*

- [x] `[infra]` Create directory structure and placeholder files
- [x] `[infra]` Add `AGENTS.md`, `ROADMAP.md`, `CONTRIBUTING.md`
- [x] `[infra]` Add CI workflow (lint + tests)
- [x] `[infra]` Add `requirements.txt`
- [x] `[infra]` Add `config/tech_placeholder.yaml` (PDK placeholder with TODOs)
- [x] `[analog]` Add minimal bandgap SPICE netlist (`bandgap_simple.sp`)
- [x] `[analog]` Add ngspice runner (`bandgap/runner.py`)
- [x] `[ML]` Add dataset generation script (`data_gen/sweep_bandgap.py`)
- [x] `[ML]` Add surrogate model stub (`ml/surrogate.py`)
- [x] `[layout]` Add layout data stub and patch model skeleton

---

## UI Review + High-Value Activities (Mar 2026)
*Goal: Track the concrete product/UX/backend tasks identified in `ui-review.md` and `High-value-next-steps.md`.*

### Delivered in PR #13 (`ui-updates`)
- [x] `[infra]` Real backend API wired to frontend (implemented with Flask in `api/server.py`)
- [x] `[infra]` Live optimizer progress streaming via SSE (`/api/optimize/stream`)
- [x] `[infra]` Save Project + Export Netlist actions wired end-to-end
- [x] `[infra]` Runtime hardening for API JSON/stream failures (fallback base + retry)
- [x] `[layout]` Layout Viewer wired to synthetic patch + DRC preview API
- [x] `[ML]` Convergence diagnostics improved in UI (axis semantics + uncertainty band)
- [x] `[ML]` Candidate metrics wiring in UI (PSRR/Iq displayed in Selected panel)
- [x] `[ML]` Unified pass/fail criterion between optimizer summary and corner table
- [x] `[infra]` Functional project/top-bar controls (datasets/netlists listing, history, settings, branch)
- [x] `[ML]` Initial convergence tuning pass (wider search bounds, higher budget defaults, surrogate/kernel tuning)
- [x] `[analog]` Real ngspice-backed flow set as default; synthetic mode requires explicit opt-in

### In progress / needs validation depth
- [ ] `[ML]` Demonstrate consistent spec convergence against real ngspice runs (not only synthetic fallback)
- [ ] `[analog]` Add reproducible benchmark report for pass-rate and convergence quality across seeds

### Pending high-value items
- [ ] `[analog]` Integrate real SKY130 PDK setup (library includes + validated tech values)
- [ ] `[ML]` Implement multi-corner joint optimization objective (corner constraints in-loop, not only post-hoc)

---

## Milestone 1 — MVP Bandgap Flow (manual sweep + logging)
*Goal: Run a real sweep, collect data, and inspect results. No ML yet.*

- [ ] `[analog]` Validate SPICE netlist against hand-calculated operating point
- [ ] `[analog]` Add corner/temperature sweep to `data_gen/sweep_bandgap.py`
- [ ] `[analog]` Log Vref, TC, PSRR, startup pass/fail per design point
- [ ] `[analog]` Write `tests/test_ngspice_smoke.py` — checks simulator is found and netlist parses
- [ ] `[ML]` Produce first CSV dataset from at least 50 SPICE runs
- [ ] `[infra]` Add `results/` directory with example plots (gitignored for large files)
- [ ] `[infra]` Update README with full quick-start instructions

---

## Milestone 2 — Surrogate v1 + Basic Optimizer Loop
*Goal: Train first ML model, wrap in Bayesian optimizer, show simulation savings.*

- [ ] `[ML]` Train Gaussian Process surrogate on Milestone-1 dataset
- [ ] `[ML]` Add uncertainty calibration check (reliability diagram)
- [ ] `[ML]` Compare surrogate predictions vs. ngspice on held-out test set
- [ ] `[ML]` Implement Bayesian optimization loop (`ml/optimize.py`)
- [ ] `[ML]` Report: simulations required (BO vs. grid sweep), spec pass rate
- [ ] `[ML]` Add `tests/test_surrogate.py` covering fit/predict/uncertainty paths
- [ ] `[analog]` Verify optimizer respects all analog sanity checks (headroom, matching)
- [ ] `[infra]` Add optimizer run to CI smoke test (tiny 5-point budget)

---

## Milestone 3 — Layout Patch Model v1
*Goal: Self-supervised patch model that predicts contacts/vias in masked layout.*

- [ ] `[layout]` Collect or generate ≥ 500 training patches (synthetic or open-source)
- [ ] `[layout]` Implement UNet encoder-decoder in `layout/patch_model.py`
- [ ] `[layout]` Add self-supervised masking pre-training script
- [ ] `[layout]` Add fine-tuning script for contact/via prediction task
- [ ] `[layout]` Implement DRC rule checks for generated patterns
- [ ] `[layout]` Add pattern similarity metric (IoU, pixel accuracy)
- [ ] `[layout]` Add `tests/test_layout.py` covering data pipeline and model I/O

---

## Milestone 4 — Evaluation + Comparison vs. Baseline
*Goal: Quantify every claim. No demo without numbers.*

- [ ] `[ML]` End-to-end comparison: BO-assisted vs. grid sweep (simulation count, time, spec rate)
- [ ] `[ML]` Failure mode analysis notebook: where does the surrogate fail?
- [ ] `[layout]` DRC pass rate on AI-generated vs. human-designed patterns
- [ ] `[layout]` Data-efficiency study: performance vs. training set size
- [ ] `[analog]` Hand-tuned expert design vs. ML-suggested design comparison
- [ ] `[infra]` Generate automated HTML/PDF report from `results/`

---

## Milestone 5 — Documentation + Onboarding
*Goal: A new engineer can understand, run, and modify the flow in < 1 hour.*

- [ ] `[infra]` Complete `CONTRIBUTING.md` with worked examples
- [ ] `[infra]` Add `examples/` directory with Jupyter notebooks
- [ ] `[infra]` Add architecture diagram to README
- [ ] `[infra]` Add FAQ section covering common ngspice/PDK setup issues
- [ ] `[analog]` Add design notes explaining bandgap topology choices
- [ ] `[ML]` Add model card for each released surrogate checkpoint

---

## Future Ideas (not scheduled)
- Multi-corner joint optimization (slow/fast/nominal simultaneously)
- Mismatch/Monte Carlo integration in surrogate training
- Layout-aware sizing (parasitics feedback from layout model)
- Transfer learning across different bandgap topologies
- Integration with KLayout for DRC scripting
