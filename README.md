# vlsi-ai-experiments

An ML-assisted analog design and layout flow, built around a Brokaw bandgap reference.

The goal is a **practical, trustworthy, and measurable** flow — not a demo.
Every claim is backed by simulation data and clear metrics.

---

## Repository Structure

```
vlsi-ai-experiments/
├── bandgap/                # Bandgap reference design
│   ├── netlists/           # SPICE netlists (ngspice-compatible)
│   │   └── bandgap_simple.sp
│   ├── runner.py           # ngspice runner: run sims, parse outputs, check specs
│   └── specs.yaml          # Target specifications (Vref, TC, PSRR, Iq, ...)
├── config/
│   └── tech_placeholder.yaml  # PDK/tech parameters — fill in for your process
├── data_gen/
│   └── sweep_bandgap.py    # Dataset generation: sweep design vars, log to CSV
├── ml/
│   ├── surrogate.py        # Surrogate models (GP, Random Forest)
│   └── optimize.py         # Bayesian optimization loop
├── layout/
│   ├── data_stub.py        # Synthetic layout patch generator
│   ├── patch_model.py      # UNet-style patch completion model (PyTorch)
│   └── evaluate.py         # IoU, pixel accuracy, and partial DRC check
├── tests/                  # Unit and smoke tests (pytest)
├── PROMPTS/                # Agent prompt files for AI co-developers
├── .github/workflows/      # CI: lint + tests
├── AGENTS.md               # Agent role definitions and guardrails
├── ROADMAP.md              # Project milestones
└── CONTRIBUTING.md         # Coding guidelines and data rules
```

---

## Quick Start

### 1. Set up environment
```bash
git clone https://github.com/j143/vlsi-ai-experiments.git
cd vlsi-ai-experiments
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure technology (required for real simulations)
Open `config/tech_placeholder.yaml` and fill in every `TODO(human):` entry
with values from your target PDK (e.g., SkyWater SKY130, GF180MCU).

### 3. Run tests (no ngspice required)
```bash
pytest tests/ -v -m "not requires_ngspice"
```
Expected output: all tests pass in ~10 seconds.

### 4. Generate a dataset (requires ngspice)
```bash
# Install ngspice first:  sudo apt-get install ngspice
python data_gen/sweep_bandgap.py --mode lhs --n-samples 50 --out datasets/
```
This writes `datasets/bandgap_sweep_<timestamp>.csv`.

### 5. Train a surrogate model
```python
import pandas as pd
from ml.surrogate import GaussianProcessSurrogate, evaluate_surrogate

df = pd.read_csv("datasets/bandgap_sweep_<timestamp>.csv")
valid = df["error"].isna() | (df["error"] == "")
X = df.loc[valid, ["N", "R1", "R2", "W_P", "L_P"]].values
y = df.loc[valid, "vref_V"].values

model = GaussianProcessSurrogate()
model.fit(X[:40], y[:40])
metrics = evaluate_surrogate(model, X[40:], y[40:])
print(metrics)
```

### 6. Run Bayesian optimization
```python
from bandgap.runner import BandgapRunner
from ml.optimize import BayesianOptimizer

runner = BandgapRunner()
opt = BayesianOptimizer(runner=runner, budget=30, n_init=10)
result = opt.run()
print(f"Best Vref: {result.best_vref_V:.4f} V")
print(f"Simulations used: {result.n_simulations}, Spec pass rate: {result.spec_pass_rate():.0%}")
```

---

## Key Design Choices

| Choice | Rationale |
|--------|-----------|
| Brokaw bandgap topology | Well-understood; easy to parameterize and compare |
| ngspice | Free, open-source, widely used; no license restrictions |
| Gaussian Process surrogate | Calibrated uncertainty; well-suited for small datasets (< 500 pts) |
| Bayesian optimization (EI) | Sample-efficient; transparent acquisition function |
| UNet for layout patches | Strong skip connections suit dense prediction tasks |
| tech_placeholder.yaml | All PDK values in one place; no guessing in code |

---

## Technology / PDK Setup

This repo ships with **placeholder** PDK values — it will not produce correct
simulation results until you replace them with values from a real PDK.

Steps:
1. Choose an open-source PDK (e.g., [SkyWater SKY130](https://github.com/google/skywater-pdk)).
2. Edit `config/tech_placeholder.yaml`: replace every `TODO(human):` line.
3. Edit `bandgap/netlists/bandgap_simple.sp`: replace `.model` lines with a
   `.lib` include pointing to your PDK model file.
4. Run `pytest tests/ -v` to confirm no regressions.
5. Run a single simulation to validate the netlist:
   ```bash
   python -c "
   from bandgap.runner import BandgapRunner
   print(BandgapRunner().run({'N':8,'R1':100e3,'R2':10e3,'W_P':4e-6,'L_P':1e-6}))
   "
   ```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards, data size limits,
and the PR checklist.

See [AGENTS.md](AGENTS.md) for agent role definitions and guardrails.

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for milestone tracking.
Current status: **Milestone 0 complete** — skeleton in place.
Next: Milestone 1 (MVP bandgap flow with real sweep data).

---

## License

[Apache 2.0](LICENSE) — see LICENSE file.