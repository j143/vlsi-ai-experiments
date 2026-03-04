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
- ✅ **Milestone 1 substantially complete** — bundled real SKY130 models + example netlists + analytics dataset
- 🟡 Milestone 2 partially complete (surrogate + BO loop implemented; real-data benchmark pending)
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

## Real Open PDK + Data Integration 🟡
*Goal: Provide domain-credible, licensed-compliant examples with publicly-available designs.*

### Completed
- [x] `[analog]` Bundle real SkyWater SKY130 SPICE models (TT corner, BSIM4 PFET ~770 KB)
- [x] `[analog]` Extract + pre-resolve PNP + poly resistor models for TT corner
- [x] `[analog]` Create minimal `pdk/sky130/` directory with LICENSE attribution
- [x] `[infra]` Create example netlists for SKY130 and IHP SG13G2 (with fallback inline models)
- [x] `[infra]` Wire example netlists to bundled models with clear comments
- [x] `[ML]` Generate analytical 200-point reference dataset using corrected Brokaw formula
- [x] `[ML]` Fix R1/R2 ratio bug in analytical dataset generation
- [x] `[infra]` Document open data resources in `examples/README.md` + `examples/open_data_guide.md`

### Remaining
- [ ] `[ML]` Validate demo_dataset.csv with corrected formula (R1/R2, not R2/R1)
- [ ] `[analog]` Run real SKY130 ngspice sweep to populate datasets/
- [ ] `[infra]` Add sample SKY130 results to results/ for reproducibility

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

## Milestone 1 — Real-Data Bandgap Foundation ✅
*Goal: Produce trusted real-simulation dataset and baseline checks.*

- [x] `[analog]` Integrate real SKY130 model includes in netlist and example flow
- [x] `[analog]` Bundle minimal SKY130 SPICE models (BSIM4 PFET + PNP + resistor, TT corner)
- [x] `[infra]` Add example bandgap netlists (SKY130, IHP SG13G2) with bundled models
- [x] `[infra]` Create `pdk/sky130/` with real Apache 2.0 open-source models
- [x] `[ML]` Generate and version analytical reference dataset (200 points, no ngspice needed)
- [x] `[infra]` Add `examples/README.md` with PDK guide + quick-start for real models
- [ ] `[analog]` Validate SPICE netlist corner + temperature sweep in sweep_bandgap.py
- [ ] `[analog]` Log Vref, TC, PSRR, Iq, startup pass/fail per design point (on real runs)
- [ ] `[analog]` Add `tests/test_ngspice_smoke.py` for simulator presence + netlist parse
- [ ] `[infra]` Produce and version first real dataset (≥ 50 valid SKY130 simulation points)

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
