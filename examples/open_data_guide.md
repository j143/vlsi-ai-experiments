# Open Data Guide — Obtaining & Using Open-Source PDKs

This document provides step-by-step instructions for obtaining, installing,
and verifying open-source process design kits (PDKs) for use with the
VLSI-AI bandgap design flow.

---

## 1. SkyWater SKY130 (130nm CMOS)

**Best for:** All-CMOS bandgap designs, sub-bandgap references.
**Limitation:** No standalone NPN BJT; only parasitic PNP available.

### Install

```bash
# Option A: Clone directly (large — ~5 GB)
git clone --depth 1 https://github.com/google/skywater-pdk.git ~/skywater-pdk
git -C ~/skywater-pdk submodule update --init libraries/sky130_fd_pr/latest

# Option B: Use open_pdks (efabless) — lighter install
git clone --depth 1 https://github.com/RTimothyEdwards/open_pdks.git ~/open_pdks
cd ~/open_pdks && ./configure --enable-sky130-pdk && make && sudo make install
# Models end up at: /usr/share/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice
```

### Verify

```bash
# Check model file exists
ls ~/skywater-pdk/libraries/sky130_fd_pr/latest/models/sky130.lib.spice

# Run a quick ngspice test
echo "
.lib '~/skywater-pdk/libraries/sky130_fd_pr/latest/models/sky130.lib.spice' tt
M1 d g s b sky130_fd_pr__nfet_01v8 W=1u L=0.15u
Vgs g 0 0.9
Vds d 0 0.9
Vss s 0 0
Vbs b 0 0
.op
.end
" | ngspice -b
```

### Key Device Models

| Device | Model name | Notes |
|--------|-----------|-------|
| 1.8V NMOS | `sky130_fd_pr__nfet_01v8` | Core transistor |
| 1.8V PMOS | `sky130_fd_pr__pfet_01v8` | Core transistor |
| Parasitic PNP | `sky130_fd_pr__pnp_05v5_W3p40L3p40` | Substrate PNP |
| High-R poly | `sky130_fd_pr__res_high_po` | ~2 kΩ/□ |
| N-diff resistor | `sky130_fd_pr__res_generic_nd` | ~120 Ω/□ |

### Use in This Repo

```bash
# Update the netlist
sed -i 's|^* .lib.*sky130.*|.lib "~/skywater-pdk/libraries/sky130_fd_pr/latest/models/sky130.lib.spice" tt|' \
    examples/sky130_bandgap.sp

# Comment out fallback models
sed -i '/^\.model sky130/s/^/\* /' examples/sky130_bandgap.sp

# Run
ngspice -b examples/sky130_bandgap.sp
```

---

## 2. IHP SG13G2 (130nm SiGe BiCMOS)

**Best for:** Classic NPN-based Brokaw bandgap (closest to our topology).
**Advantage:** Real standalone NPN HBT (fT > 200 GHz).

### Install

```bash
git clone https://github.com/IHP-GmbH/IHP-Open-PDK.git ~/ihp-pdk

# Key model files:
# ~/ihp-pdk/ihp-sg13g2/libs.tech/ngspice/
#   - sg13g2_moslevel.lib
#   - npn13G2_moslevel.lib
```

### Key Device Models

| Device | Model name | Notes |
|--------|-----------|-------|
| LV NMOS | `sg13_lv_nmos` | 1.2V core |
| LV PMOS | `sg13_lv_pmos` | 1.2V core |
| NPN HBT | `npn13G2` | SiGe bipolar — ideal for Brokaw |
| High-R poly | `rhigh` | ~1.4 kΩ/□ |

### Use in This Repo

Update `examples/ihp_sg13g2_bandgap.sp` with the correct `.lib` path and
remove the fallback `.model` lines.

---

## 3. GlobalFoundries GF180MCU (180nm CMOS)

**Best for:** Higher-voltage designs (3.3V/5V), larger devices.
**Limitation:** No standalone BJT; parasitic devices only.

### Install

```bash
git clone --depth 1 https://github.com/google/gf180mcu-pdk.git ~/gf180mcu-pdk
```

### Key files
- Models at: `models/ngspice/`
- 3.3V NMOS: `nfet_03v3`, PMOS: `pfet_03v3`

---

## 4. Using Open Data with the ML Pipeline

### A. Generate Real SPICE Data

Once you have a PDK installed and ngspice working:

```bash
# Copy and adapt the netlist template for your PDK
cp examples/sky130_bandgap.sp bandgap/netlists/bandgap_sky130.sp
# Edit the .lib path, then:

python data_gen/sweep_bandgap.py \
    --mode lhs \
    --n-samples 100 \
    --out datasets/ \
    --netlist bandgap/netlists/bandgap_sky130.sp
```

### B. Train on Real Data

```python
import pandas as pd
from ml.surrogate import GaussianProcessSurrogate, evaluate_surrogate

df = pd.read_csv("datasets/bandgap_sweep_YYYYMMDD.csv")
valid = df["vref_V"].notna()
X = df.loc[valid, ["N", "R1", "R2", "W_P", "L_P"]].values
y = df.loc[valid, "vref_V"].values

model = GaussianProcessSurrogate()
model.fit(X[:80], y[:80])
metrics = evaluate_surrogate(model, X[80:], y[80:])
print(f"MAE: {metrics['mae']*1000:.2f} mV, R²: {metrics['r2']:.4f}")
```

### C. Cross-validate Against Analytical Baseline

```python
# Compare SPICE dataset Vref vs analytical prediction
from examples.generate_reference_dataset import analytical_vref

df["vref_analytical"] = df.apply(
    lambda r: analytical_vref(N=r["N"], R1=r["R1"], R2=r["R2"]),
    axis=1,
)
delta = (df["vref_V"] - df["vref_analytical"]).abs() * 1000
print(f"SPICE vs Analytical: mean Δ = {delta.mean():.2f} mV, max = {delta.max():.2f} mV")
```

This tells you how much the real process deviates from the ideal model —
a key input for surrogate accuracy assessment.

---

## 5. Third-Party Datasets and Benchmarks

| Source | URL | What you get |
|--------|-----|--------------|
| MAGICAL (UT Austin) | <https://github.com/magical-eda/MAGICAL> | Analog layout benchmarks (OTA, bandgap, comparator) |
| CircuitNet (CUHK) | <https://circuitnet.github.io> | Netlist-to-layout congestion/DRC data |
| OpenROAD flow | <https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts> | Digital + mixed-signal reference flows |
| ALIGN (U Minnesota) | <https://github.com/ALIGN-analern/ALIGN-public> | Analog layout automation with examples |

These can provide layout training data for the `layout/patch_model.py` UNet.

---

## 6. Data Validation Checklist

Before using any dataset (generated or external) with the ML pipeline:

- [ ] Verify column names match `FEATURES` in `ml/surrogate.py`
- [ ] Check for NaN/error rows and decide: drop or impute
- [ ] Confirm units are SI (V, Ω, m — not µm, kΩ)
- [ ] Validate Vref range is physically plausible (0.5–2.0 V for bandgap)
- [ ] Record provenance: PDK version, ngspice version, sweep config, date
- [ ] Compare a few points against hand calculation
