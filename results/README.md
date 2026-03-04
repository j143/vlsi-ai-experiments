# Results Directory

This directory contains outputs from various experiments and runs in the project.

## File Types

### Bayesian Optimization Runs
Files: `bo_run_<timestamp>.json`

Each file documents a single Bayesian Optimizer run with:
- **optimization_history**: List of evaluated points and their objective values
- **best_point**: Design parameters that achieved the best result
- **best_objective**: Vref (or other metric) at best point
- **n_iterations**: Number of BO iterations completed
- **n_simulations**: Total number of SPICE/surrogate evaluations
- **timestamp**: When the run was executed

Use these to:
- Benchmark optimizer performance (convergence speed, final fit)
- Compare different surrogate models or acquisition functions
- Reproduce results (raw data for papers/reports)

### Project State Snapshots
Files: `projects/ui_state_<timestamp>.json`

Snapshots of the UI/API project state, including:
- Design parameters
- Convergence history
- Selected candidates

Used by the frontend to persist and restore user sessions.

---

## Data Retention Policy

- **Simulation logs & raw data**: Not committed (leave in `.gitignore`).
- **BO run records**: Keep recent runs; archive older ones if > 50 runs exist.
- **Result plots & benchmarks**: Commit final summary reports; keep run logs for reproducibility.

---

## How to Generate Results

### Analytical Dataset
```bash
python examples/generate_reference_dataset.py
```
Output saved to: `examples/demo_dataset.csv`

### ML Pipeline with Analytical Data
```bash
python examples/run_full_pipeline.py
```
May produce plots or summary stats in `results/`.

### Real SPICE Sweep (requires ngspice + SKY130 models)
```bash
python data_gen/sweep_bandgap.py --mode lhs --n-samples 50 --out datasets/
```
Output saved to: `datasets/bandgap_sweep_<timestamp>.csv`

### BO Run with Real Data
```python
from bandgap.runner import BandgapRunner
from ml.optimize import BayesianOptimizer

runner = BandgapRunner()
opt = BayesianOptimizer(runner=runner, budget=30, n_init=5)
result = opt.run()
print(result.best_point)
```
Output written to: `results/bo_run_<timestamp>.json`

---

## Interpreting Results

### Convergence Metrics
- **Monotonic improvement**: Good. Each iteration finds a better candidate.
- **Plateauing**: Normal late in BO; surrogate has learned the landscape.
- **Noise / backtracking**: Expected with real simulations (SPICE variation).

### Spec Pass Rate
- Count: `(spec_vref_pass==True).sum()`
- Rate: `(spec_vref_pass==True).mean() * 100 %`
- Target: > 50% for a useful optimizer (depends on design space).

### Sample Efficiency
- **Surrogate-based BO**: Typically 20–40 simulations to converge (for small design spaces).
- **Grid search baseline**: 100–500 simulations for equivalent coverage.
- **Speedup**: Ratio of simulations saved vs grid search.

---

## Sharing & Reproducibility

To share results:
1. Include the relevant `bo_run_<timestamp>.json` file(s).
2. Include the dataset used (e.g., `datasets/bandgap_sweep_<timestamp>.csv`).
3. Document your environment: Python version, PDK, ngspice version.
4. Provide the commit hash of the code base: `git log -1 --oneline`.

Example:
```
Run: bo_run_20260304_170540.json
Dataset: datasets/bandgap_sweep_20260304.csv
Commit: a1b2c3d "Add real SKY130 models"
Python: 3.9.13
ngspice: ngspice-41
PDK: sky130_fd_pr (TT corner, bundled)
```
