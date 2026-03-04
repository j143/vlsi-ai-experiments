# Contributing to vlsi-ai-experiments

Thank you for contributing! This document explains how to work in this repo safely
and consistently. Please read `AGENTS.md` first to understand which agent/role
owns which parts of the codebase.

---

## Table of Contents
1. [Quick Setup](#quick-setup)
2. [Code Style](#code-style)
3. [Adding a New Experiment](#adding-a-new-experiment)
4. [Adding a New Model](#adding-a-new-model)
5. [Data Storage Rules](#data-storage-rules)
6. [Testing Requirements](#testing-requirements)
7. [Pull Request Checklist](#pull-request-checklist)

---

## Quick Setup

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install ngspice
#    Ubuntu/Debian: sudo apt-get install ngspice
#    macOS:         brew install ngspice
#    Windows:       https://ngspice.sourceforge.io/download.html

# 4. Run tests
pytest tests/ -v

# 5. Run lint
flake8 . --max-line-length=100
```

---

## Code Style

- **Python 3.9+** required.
- Follow [PEP 8](https://peps.python.org/pep-0008/) with a max line length of **100 characters**.
- Use type hints for all public function signatures.
- Use `flake8` for linting (config in `setup.cfg`).
- Format with `black --line-length 100` (optional but preferred).
- Write docstrings for every public function and class (Google style).

### Naming Conventions
| Type | Convention | Example |
|------|-----------|---------|
| Files | `snake_case.py` | `sweep_bandgap.py` |
| Functions | `snake_case` | `run_ngspice()` |
| Classes | `PascalCase` | `BandgapSurrogate` |
| Constants | `UPPER_CASE` | `DEFAULT_VDD` |
| Config keys | `snake_case` | `vref_target` |

---

## Adding a New Experiment

1. Create a new directory under the appropriate module (e.g., `bandgap/`, `layout/`).
2. Add a `README.md` inside explaining:
   - What the experiment tests.
   - Required inputs and expected outputs.
   - How to run it.
3. Add a SPICE netlist or data file with **unit-consistent values** and source citations.
4. Add a sweep/run script following the pattern in `data_gen/sweep_bandgap.py`.
5. Log all outputs to CSV/Parquet with a timestamp column.
6. Add at least one smoke test in `tests/`.

### Technology Parameters
- **Never hardcode** PDK-specific values (Vth, Is, Beta, layer names) in Python code.
- All such values must live in `config/tech_placeholder.yaml`.
- Mark unknown values with `# TODO(human): replace with actual PDK value` comments.
- See `config/tech_placeholder.yaml` for the expected format.

---

## Adding a New Model

1. Add the model class to `ml/surrogate.py` (surrogate models) or `layout/patch_model.py`
   (layout models).
2. The class must implement:
   - `fit(X, y)` — train on data.
   - `predict(X)` — return predictions.
   - `predict_with_uncertainty(X)` — return `(mean, std)` if supported.
3. Add a corresponding entry in `ml/model_registry.py` (create if absent).
4. Document hyperparameters in the class docstring with default values and ranges.
5. Add unit tests in `tests/test_surrogate.py` or `tests/test_layout.py`.

### Model Artifacts
- Checkpoints must be saved to `ml/checkpoints/<model_name>/<timestamp>/`.
- Files > 50 MB must **not** be committed; add to `.gitignore`.
- Provide a script to regenerate any committed checkpoint from scratch.

---

## Data Storage Rules

| Data type | Format | Max committed size | Location |
|-----------|--------|--------------------|----------|
| Simulation sweep output | CSV or Parquet | 5 MB per file | `datasets/` |
| Raw ngspice output | `.raw` / `.log` | Not committed | Local only |
| Layout patches | NumPy `.npy` | 10 MB total | `layout/data/` |
| Model checkpoints | `.pt` / `.pkl` | 50 MB total | `ml/checkpoints/` |
| Result plots | `.png` / `.pdf` | 2 MB per file | `results/` |

Large files (> limits above) must be documented in `datasets/README.md` with
download or generation instructions.

---

## Testing Requirements

- Every new Python module must have at least **one test file** in `tests/`.
- Tests must not require ngspice to be installed (mock or skip if absent).
- Use `pytest.mark.skip` or `pytest.mark.skipif` for hardware/tool-dependent tests.
- Tests must be fast (< 30 seconds per file) — use small synthetic data.
- Aim for > 80% line coverage on new code.

### Running Tests
```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_surrogate.py -v

# Skip slow / tool-dependent tests
pytest tests/ -v -m "not slow and not requires_ngspice"
```

---

## Pull Request Checklist

Before opening a PR, confirm:

- [ ] `flake8` passes with no errors.
- [ ] All existing tests pass: `pytest tests/ -v`.
- [ ] New code has tests (see requirements above).
- [ ] No PDK parameters are hardcoded — they are in `config/tech_placeholder.yaml`.
- [ ] No large files (> limits above) are committed.
- [ ] `ROADMAP.md` is updated if a milestone task is completed.
- [ ] PR description explains *what* changed and *why*.
- [ ] If changing a SPICE netlist, describe the change's effect on the operating point.
