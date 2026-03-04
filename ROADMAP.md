# ROADMAP.md — Project Milestones

Track progress by tagging tasks: `[analog]`, `[ML]`, `[layout]`, `[infra]`.

Status legend:
- ✅ Complete
- 🟡 In progress / partial
- ⬜ Not started

---

## Current Snapshot (Mar 2026)

- ✅ Milestone 0 complete (repo scaffold, CI, core modules)
- 🟡 Product/UI integration sprint delivered in PR #13 (`ui-updates`)
- 🟡 Milestone 1 partially complete (runner/sweep exist, real-PDK validation pending)
- 🟡 Milestone 2 partially complete (surrogate + BO loop implemented; benchmark quality work pending)
- ⬜ Milestones 3–5 mostly pending

---

## Product Integration Sprint — UI Review + High-Value Activities ✅
*Goal: Execute concrete UI/backend tasks from `ui-review.md` and `High-value-next-steps.md`.*

### Delivered
- [x] `[infra]` Frontend connected to live backend API (`api/server.py`)
- [x] `[infra]` SSE optimizer streaming (`GET /api/optimize/stream`)
- [x] `[infra]` Save Project + Export Netlist actions wired end-to-end
- [x] `[infra]` API robustness: JSON parsing hardening + fallback API base + stream retry
- [x] `[layout]` Layout tab backed by synthetic patch + DRC preview API
- [x] `[ML]` Convergence UI clarity: axis semantics + uncertainty band rendering
- [x] `[ML]` Selected candidate now shows PSRR and Iq from optimizer history
- [x] `[ML]` Unified pass/fail criterion across optimizer summary and corner panel
- [x] `[infra]` Project nav controls wired (datasets/netlists listing, top-bar actions)
- [x] `[ML]` Convergence tuning pass (bounds, budget defaults, GP tuning)
- [x] `[analog]` Real ngspice-backed flow is default; synthetic fallback is explicit opt-in

### Follow-up validation (remaining)
- [ ] `[ML]` Prove convergence consistency on real ngspice runs across multiple seeds
- [ ] `[analog]` Publish reproducible pass-rate benchmark from real runs

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
- [x] `[ML]` Add surrogate model module (`ml/surrogate.py`)
- [x] `[layout]` Add layout data stub and patch model skeleton

---

## Milestone 1 — Real-Data Bandgap Foundation 🟡
*Goal: Produce trusted real-simulation dataset and baseline checks.*

- [ ] `[analog]` Validate SPICE netlist against hand-calculated operating point
- [ ] `[analog]` Integrate real SKY130 (or selected open PDK) model includes in netlist flow
- [ ] `[analog]` Add/verify corner + temperature sweep in `data_gen/sweep_bandgap.py`
- [ ] `[analog]` Log Vref, TC, PSRR, Iq, startup pass/fail per design point
- [ ] `[analog]` Add `tests/test_ngspice_smoke.py` for simulator presence + netlist parse
- [ ] `[ML]` Produce and version first real dataset (≥ 50 valid SPICE points)
- [ ] `[infra]` Add dataset provenance metadata (PDK, commit hash, sweep config)
- [ ] `[infra]` Update README quick-start for real-simulation workflow

---

## Milestone 2 — Surrogate + Optimizer Quality Loop 🟡
*Goal: Show measurable simulation savings with calibrated uncertainty.*

- [x] `[ML]` Implement Bayesian optimization loop (`ml/optimize.py`)
- [x] `[ML]` Add surrogate fit/predict/uncertainty test coverage (`tests/test_surrogate.py`)
- [ ] `[ML]` Train GP surrogate on Milestone-1 real dataset
- [ ] `[ML]` Add uncertainty calibration check (reliability / coverage)
- [ ] `[ML]` Compare surrogate vs ngspice on held-out real test set
- [ ] `[ML]` Add BO-vs-grid benchmark report (sim count, time, spec-pass rate)
- [ ] `[analog]` Verify optimizer proposals satisfy analog sanity checks
- [ ] `[infra]` Add tiny optimizer smoke path in CI (short budget)

---

## Milestone 3 — Layout Patch Model v1 ⬜
*Goal: Build and evaluate a self-supervised patch completion model.*

- [ ] `[layout]` Collect/generate ≥ 500 training patches (open/synthetic only)
- [ ] `[layout]` Implement UNet encoder-decoder in `layout/patch_model.py`
- [ ] `[layout]` Add self-supervised masking pre-training script
- [ ] `[layout]` Add fine-tuning path for contact/via completion
- [ ] `[layout]` Implement DRC rule checks for generated patches
- [ ] `[layout]` Add similarity metrics (IoU, pixel accuracy)
- [ ] `[layout]` Add tests covering data pipeline and model I/O

---

## Milestone 4 — Quantitative Evaluation ⬜
*Goal: Quantify all claims with reproducible experiments.*

- [ ] `[ML]` End-to-end BO-assisted vs grid-sweep comparison
- [ ] `[ML]` Failure-mode analysis for surrogate errors
- [ ] `[layout]` DRC pass-rate comparison: generated vs reference layouts
- [ ] `[layout]` Data-efficiency sweep vs training-set size
- [ ] `[analog]` Expert hand-tuned vs ML-suggested design comparison
- [ ] `[infra]` Auto-generate consolidated HTML/PDF results report

---

## Milestone 5 — Documentation + Onboarding ⬜
*Goal: New contributor can run/modify flow in < 1 hour.*

- [ ] `[infra]` Expand `CONTRIBUTING.md` with worked examples
- [ ] `[infra]` Add `examples/` notebooks/scripts
- [ ] `[infra]` Add architecture diagram to README
- [ ] `[infra]` Add FAQ for ngspice + PDK setup
- [ ] `[analog]` Add design notes for bandgap topology choices
- [ ] `[ML]` Add model card template for released surrogate checkpoints

---

## Future Ideas (not scheduled)

- Multi-corner joint optimization (optimize with corner constraints in-loop)
- Mismatch / Monte Carlo integration in surrogate training
- Layout-aware sizing with parasitic feedback
- Transfer learning across bandgap topologies
- KLayout integration for programmable DRC scripting
