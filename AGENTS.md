# AGENTS.md — Agent Role Definitions

This document defines the roles, responsibilities, inputs, outputs, and guardrails for
each AI agent operating in this repository. All agents must read this file before acting.

---

## 1. Design Agent

**Purpose**: Work on analog circuit design — topology, sizing, netlists, and specs.

### Inputs
- `bandgap/netlists/*.sp` — SPICE netlists
- `bandgap/specs.yaml` — target specifications
- `config/tech_placeholder.yaml` — technology parameters (PDK placeholders)
- `data_gen/*.py` — sweep/simulation scripts

### Outputs
- New or updated SPICE netlists in `bandgap/netlists/`
- Updated `bandgap/specs.yaml` if specs change
- Updated `data_gen/sweep_bandgap.py` for new design variables
- Documentation in `bandgap/README.md`

### Guardrails
- **DO NOT** modify PDK model files or `config/tech_placeholder.yaml` with guessed values.
  Always add a `# TODO(human):` comment instead.
- **DO NOT** remove or silently relax spec constraints in `bandgap/specs.yaml`.
- **DO NOT** alter golden netlists without a clear commit message explaining the change.
- **DO NOT** hardcode process-specific parameters (Vth, Is, Beta) without citing source.
- Always keep netlists unit-consistent; add comments for every component value.
- Any new design variable must be documented with its physical meaning and range.

---

## 2. ML / Optimization Agent

**Purpose**: Build and improve surrogate models, optimization loops, and datasets.

### Inputs
- `data_gen/` — dataset generation scripts
- `ml/` — existing surrogate model and optimizer code
- `datasets/` — CSV/Parquet files produced by simulation sweeps (incl. `bandgap_sweep_real_sky130.csv`)
- `bandgap/specs.yaml` — spec definitions (for objective functions)

### Outputs
- Updated `ml/surrogate.py` and `ml/optimize.py`
- New model checkpoints in `ml/checkpoints/` (not committed if > 50 MB)
- Evaluation plots/reports in `results/`
- Updated tests in `tests/test_surrogate.py`, `tests/test_optimize.py`

### Guardrails
- **DO NOT** modify SPICE netlists or PDK configs.
- **DO NOT** silently skip spec constraints — all objectives must be logged.
- Always report uncertainty estimates with ML predictions.
- Always compare ML outputs against ngspice baseline in evaluation scripts.
- Never commit large dataset files (> 10 MB) — add them to `.gitignore`.
- Model hyperparameters must be configurable via arguments, not hardcoded.
- Every new model must have at least one unit test.
- **Accuracy must be evaluated on real data**: use `accuracy_confidence()` from `ml/surrogate.py`
  and run evaluation on `datasets/bandgap_sweep_real_sky130.csv` whenever available.
- **Preset weights** must be wired into the BO loss function; do not add presets that only
  change `budget`/`n_init` without also adjusting the multi-output scalar loss.

---

## 3. Layout Agent

**Purpose**: Work on layout automation tasks — patch models, DRC/LVS, data pipelines.

### Inputs
- `layout/` — existing layout data stubs and model code
- `config/tech_placeholder.yaml` — layer definitions and design rules
- Reference patches in `layout/data/` (synthetic or from open-source layouts)

### Outputs
- Updated `layout/patch_model.py`, `layout/data_stub.py`, `layout/evaluate.py`
- Evaluation reports in `results/layout/`
- Updated tests in `tests/test_layout.py`

### Guardrails
- **DO NOT** invent DRC rules — only use rules explicitly defined in
  `config/tech_placeholder.yaml` or cite the source PDK.
- **DO NOT** commit proprietary layout data.
- Always flag AI-generated layouts as unverified until DRC/LVS checks pass.
- Pattern similarity metrics must be reported alongside DRC results.
- Layout data must be kept in open, documented formats (GDS2, LEF/DEF, or structured NumPy arrays).

---

## 4. Docs / Infra Agent

**Purpose**: Maintain documentation, CI, examples, and repo health.

### Inputs
- `README.md`, `ROADMAP.md`, `CONTRIBUTING.md`, `AGENTS.md`
- `.github/workflows/` — CI configuration
- `PROMPTS/` — agent prompt files
- All other files (read-only for context)

### Outputs
- Updated Markdown documentation
- Updated `.github/workflows/ci.yml`
- New examples in `examples/`
- Updated `requirements.txt` or `setup.py`

### Guardrails
- **DO NOT** modify SPICE netlists, Python source files, or ML model code.
- **DO NOT** remove any existing CI checks.
- All new CI steps must be additive and non-breaking.
- Keep documentation concise — prefer examples over lengthy prose.
- Never hardcode credentials or secrets in any file.

---

## Inter-Agent Coordination

- Before modifying a shared file, check if another agent owns it (see above).
- Use `# TODO(agent-name):` comments to leave notes for other agents.
- Open a PR for every substantive change; do not commit directly to `main`.
- Milestone tracking lives in `ROADMAP.md` — update it when closing tasks.

## CI Checklist (all agents)

Every PR must pass all of the following before merging:
1. `flake8 . --max-line-length=100 --exclude=.venv,__pycache__,.git,node_modules` — 0 errors
2. `pytest tests/ -m "not requires_ngspice and not slow"` — 0 failures
3. Playwright E2E (`cd frontend && npx playwright test`) — 27/27 pass
4. `npm run build` in `frontend/` — 0 errors
