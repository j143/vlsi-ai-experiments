# vlsi-ai-experiments

An ML-assisted analog design and layout flow, built around a Brokaw bandgap reference.

Every claim is backed by simulation data and clear metrics.

---

## Repository Structure

```
vlsi-ai-experiments/
├── bandgap/                # Bandgap reference design
│   ├── netlists/           # SPICE netlists (ngspice-compatible)
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
├── api/                    # Flask REST API (backend)
├── frontend/               # React/Vite UI (frontend)
├── docker/                 # Dockerfiles and docker-compose
├── tests/                  # Unit and smoke tests (pytest)
├── PROMPTS/                # Agent prompt files for AI co-developers
├── .github/workflows/      # CI: lint, build, test
├── AGENTS.md               # Agent role definitions and guardrails
├── ROADMAP.md              # Project milestones
└── CONTRIBUTING.md         # Coding guidelines and data rules
```

---

## Quick Start

### Option A — Docker (recommended, no local setup needed)

```bash
git clone https://github.com/j143/vlsi-ai-experiments.git
cd vlsi-ai-experiments

# Build and start the full app (UI + API) on http://localhost:5000
cd docker
docker compose up --build
```

To run just the engine CLI inside Docker:
```bash
# Show help
docker run --rm vlsi-ai-engine --help

# Run the built-in demo
docker run --rm -v $PWD/results:/app/results vlsi-ai-engine demo
```

### Option B — Local Python setup

```bash
git clone https://github.com/j143/vlsi-ai-experiments.git
cd vlsi-ai-experiments
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

#### Run the API server

```bash
python api/server.py              # default port 5000
```

Then open `http://localhost:5000` (requires the frontend to be built — see below).

#### Build and serve the frontend (UI)

```bash
cd frontend
npm ci
npm run build                     # writes static files to frontend/dist/

# Start the API server with the built UI
cd ..
STATIC_DIR=frontend/dist python api/server.py
# Open http://localhost:5000
```

During development, the Vite dev server can proxy API calls:
```bash
# Terminal 1 — backend
python api/server.py

# Terminal 2 — frontend (hot-reload)
cd frontend
npm run dev                       # http://localhost:5173
```

### Run tests (no ngspice required)

```bash
pytest tests/ -v -m "not requires_ngspice"
```

All tests pass in ~10 seconds.

---

## SKY130 / PDK Support

This repo bundles **real SPICE models** from open-source PDKs.

### Bundled: SkyWater SKY130 (Apache 2.0)
- Location: `pdk/sky130/` — minimal TT-corner extraction (~770 KB)
- Example netlist: `examples/sky130_bandgap.sp`

```bash
# With ngspice installed:
sudo apt-get install ngspice
ngspice examples/sky130_bandgap.sp
```

### Other PDKs (IHP SG13G2, GF180MCU, ...)
See [examples/README.md](examples/README.md) for installation and usage.

### Custom process
Edit `config/tech_placeholder.yaml` and replace every `TODO(human):` entry.

---

## Generate a Dataset

```bash
# Install ngspice first:  sudo apt-get install ngspice
python data_gen/sweep_bandgap.py --mode lhs --n-samples 50 --out datasets/
```

Or use the pre-generated reference dataset: `datasets/bandgap_sweep_real_sky130.csv`.

---

## Train a Surrogate Model

```python
import pandas as pd
from ml.surrogate import GaussianProcessSurrogate, evaluate_surrogate

df = pd.read_csv("datasets/bandgap_sweep_real_sky130.csv")
valid = df["error"].isna() | (df["error"] == "")
X = df.loc[valid, ["N", "R1", "R2", "W_P", "L_P"]].values
y = df.loc[valid, "vref_V"].values

model = GaussianProcessSurrogate()
model.fit(X[:40], y[:40])
print(evaluate_surrogate(model, X[40:], y[40:]))
```

---

## Run Bayesian Optimization

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards, data size limits,
and the PR checklist.

See [AGENTS.md](AGENTS.md) for agent role definitions and guardrails.

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for milestone tracking.

---

## License

[Apache 2.0](LICENSE) — see LICENSE file.