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

### 2. Run with bundled SKY130 models (recommended)
This repo includes **real open-source SKY130 SPICE models** in `pdk/sky130/`.
You can immediately run examples without installing a PDK:

```bash
# Instant: run analytics + ML pipeline
python examples/generate_reference_dataset.py
python examples/run_full_pipeline.py

# With ngspice installed: real simulation
sudo apt-get install ngspice
ngspice examples/sky130_bandgap.sp
```

See [examples/README.md](examples/README.md) for full details and other PDKs.

### 3. (Optional) Configure other PDKs
For other processes (GF180MCU, IHP SG13G2, custom), edit `config/tech_placeholder.yaml`
and replace every `TODO(human):` entry with values from your target PDK.

### 4. Run tests (no ngspice required)
```bash
pytest tests/ -v -m "not requires_ngspice"
```
Expected output: all tests pass in ~10 seconds.

### 5. Generate a dataset (requires ngspice + your own PDK)
```bash
# Install ngspice:  sudo apt-get install ngspice
python data_gen/sweep_bandgap.py --mode lhs --n-samples 50 --out datasets/
```
This writes `datasets/bandgap_sweep_<timestamp>.csv`.
Or use the pre-generated analytics dataset from `examples/demo_dataset.csv`.

### 6. Train a surrogate model
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

### 7. Run Bayesian optimization
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

## Real Open PDK Support

This repo bundles **real SPICE models** from open-source PDKs to support reproducible
simulation without requiring PDK installation.

### Bundled: SkyWater SKY130 (Apache 2.0)
- Location: `pdk/sky130/` — minimal TT-corner extraction (~770 KB BSIM4 PFET + PNP + resistor models)
- Example netlist: `examples/sky130_bandgap.sp` — ready to run with ngspice
- See [pdk/sky130/README.md](pdk/sky130/README.md) for model inventory

### Available: IHP SG13G2, GF180MCU, others
For other PDKs, see [examples/README.md](examples/README.md) for installation and usage.
Each includes a reference bandgap netlist and an `open_data_guide.md` for setup.

---

## Technology / PDK Configuration

**For SKY130 (recommended for quick start):**
- Models are bundled and ready to use.
- No additional configuration needed.

**For other PDKs or custom processes:**
1. Follow the installation guide in [examples/open_data_guide.md](examples/open_data_guide.md).
2. Edit `config/tech_placeholder.yaml` to add process parameters.
3. Update your netlist to `.include` the PDK model library.

For examples of pre-wired netlists, see `examples/`.

---

## Old-style: Tech Placeholder (deprecated but still supported)

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