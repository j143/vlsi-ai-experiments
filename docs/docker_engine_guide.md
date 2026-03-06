# Docker Engine Guide

This guide shows how to build and run the **vlsi-ai engine** Docker image — a
self-contained environment for bandgap design sweeps and Bayesian optimisation.
No local ngspice install or PDK setup is required on the host.

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) ≥ 20.10 installed and running.
- A clone of this repository.

---

## 1. Build the image

```bash
git clone https://github.com/j143/vlsi-ai-experiments.git
cd vlsi-ai-experiments

docker build -t vlsi-ai-engine -f docker/Dockerfile.engine .
```

The first build downloads system packages (ngspice, Python deps) and may take
a few minutes. Subsequent builds use the cached pip layer if `requirements.txt`
has not changed.

---

## 2. Run the end-to-end demo (recommended first step)

```bash
docker run --rm \
  -v "$PWD/results:/app/results" \
  vlsi-ai-engine demo
```

This single command:

1. Generates a synthetic bandgap dataset (analytical Brokaw model).
2. Trains a GP surrogate.
3. Runs Bayesian Optimisation (default budget: 30 simulations).
4. Compares BO against a brute-force grid search.
5. Writes `results/demo_report.md` and `results/demo_summary.json`.

Open `results/demo_report.md` to see the benchmark table.

---

## 3. Run a parameter sweep

```bash
docker run --rm \
  -v "$PWD/datasets:/app/datasets" \
  vlsi-ai-engine sweep \
    --netlist bandgap/netlists/bandgap_simple.sp \
    --out datasets/sky130_bandgap_real.csv \
    --n-samples 80
```

The CSV is written to `datasets/sky130_bandgap_real.csv` on your host via the
volume mount.

---

## 4. Run Bayesian Optimisation on an existing dataset

```bash
docker run --rm \
  -v "$PWD/datasets:/app/datasets" \
  -v "$PWD/results:/app/results" \
  vlsi-ai-engine optimize \
    --dataset datasets/sky130_bandgap_real.csv \
    --budget 30 \
    --out results/bo_run_$(date +%s)
```

---

## 5. CLI reference

```
vlsi-ai --help
vlsi-ai sweep --help
vlsi-ai optimize --help
vlsi-ai demo --help
```

---

## 6. Environment variables

| Variable | Default | Description |
|---|---|---|
| `VLSI_AI_DATASETS` | `/app/datasets` | Default dataset directory |
| `VLSI_AI_RESULTS` | `/app/results` | Default results directory |
| `NGSPICE_BIN` | (auto-detected) | Override ngspice binary path |

---

## Notes

- **ngspice** is pre-installed inside the image, so no host-side PDK configuration
  is needed.
- When ngspice is available the `sweep` command runs real SPICE simulations.
  The `optimize` and `demo` commands always use the analytic Brokaw fallback
  unless `--dataset` points to real SPICE data.
- Result files land in `/app/results` inside the container; mount a host
  directory with `-v $PWD/results:/app/results` to persist them.
