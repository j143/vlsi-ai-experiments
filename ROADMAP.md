# ROADMAP.md — Project Milestones

Track progress by tagging tasks: `[analog]`, `[ML]`, `[layout]`, `[infra]`.

**Practical targets (from issue #roadmap-practical-aspects):**
1. Bandgap: given specs, suggest 2–3 candidate designs that pass SPICE with 3–5× fewer
   simulations than a brute-force sweep.
2. Layout: given a partial patch, propose dummy/via/fill completions so that ≥90% of patches
   pass DRC/LVS on synthetic/open data.

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

## Milestone 1 — MVP Bandgap Flow (manual sweep + logging)
*Goal: Run a real sweep, collect data, and inspect results. No ML yet.*

- [ ] `[analog]` Validate SPICE netlist against hand-calculated operating point
- [ ] `[analog]` Add corner/temperature sweep to `data_gen/sweep_bandgap.py`
- [ ] `[analog]` Log Vref, TC, PSRR, startup pass/fail per design point
- [ ] `[analog]` Write `tests/test_ngspice_smoke.py` — checks simulator is found and netlist parses
- [ ] `[ML]` Produce first CSV dataset from at least 50 SPICE runs
- [x] `[infra]` Add `results/` directory with example plots (gitignored for large files)
- [x] `[infra]` Update README with full quick-start instructions

---

## Milestone 2 — Surrogate v1 + Basic Optimizer Loop ✅
*Goal: Train first ML model, wrap in Bayesian optimizer, show simulation savings.
Delivers practical target #1: suggest 2–3 spec-passing candidates with 3–5× fewer sims.*

- [x] `[ML]` Gaussian Process surrogate implemented in `ml/surrogate.py` (fit/predict/uncertainty)
- [x] `[ML]` Uncertainty calibration reported via `evaluate_surrogate()` (within-1σ fraction)
- [x] `[ML]` Surrogate evaluation vs. held-out test set (`evaluate_surrogate`)
- [x] `[ML]` Bayesian optimization loop (`ml/optimize.py`) — EI acquisition, LHS init
- [x] `[ML]` Report: simulations (BO vs. grid sweep), spec pass rate, simulation reduction %
- [x] `[ML]` `OptimizationResult.top_candidates` — 2–3 spec-passing designs sorted by |Vref − target|
- [x] `[ML]` `tests/test_surrogate.py` covering fit/predict/uncertainty paths
- [x] `[ML]` `tests/test_optimize.py` covering BO loop, top_candidates, JSON output
- [ ] `[analog]` Verify optimizer respects all analog sanity checks (headroom, matching)
- [x] `[infra]` Optimizer smoke test in CI (5-point budget, synthetic runner)

---

## Milestone 3 — Layout Patch Model v1
*Goal: Self-supervised patch model that predicts contacts/vias in masked layout.
Delivers practical target #2: ≥90% DRC pass rate on synthetic/open data.*

- [x] `[layout]` Synthetic data generator produces ≥ 500 training patches (`layout/data_stub.py`)
- [x] `[layout]` UNet encoder-decoder implemented in `layout/patch_model.py`
- [x] `[layout]` Self-supervised masking utility (`mask_patches` in `layout/data_stub.py`)
- [ ] `[layout]` Add fine-tuning script for contact/via/dummy fill prediction task
- [x] `[layout]` Partial DRC rule checks for generated patterns (`layout/evaluate.py`)
- [x] `[layout]` Pattern similarity metrics: IoU and pixel accuracy per layer
- [x] `[layout]` `tests/test_layout.py` covering data pipeline, DRC, and model I/O
- [ ] `[layout]` Train model on ≥ 500 patches; report DRC pass rate (target ≥ 90%)

---

## Milestone 4 — Evaluation + Comparison vs. Baseline
*Goal: Quantify every claim. No demo without numbers.*

- [x] `[ML]` End-to-end comparison: BO-assisted vs. grid sweep (simulation count, spec rate)
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
