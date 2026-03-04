# SKY130 Model Extraction for Bandgap Reference

Minimal SPICE model extraction from the SkyWater SKY130 open-source PDK,
containing only the devices needed for bandgap voltage reference simulation.

## Source

- **Repository**: [google/skywater-pdk-libs-sky130_fd_pr](https://github.com/google/skywater-pdk-libs-sky130_fd_pr)
- **License**: Apache 2.0 (see LICENSE in this directory)
- **Corner**: TT (typical-typical)

## Files

| File | Size | Description |
|------|------|-------------|
| `sky130_bandgap_tt.lib` | ~5 KB | Top-level library — defines PNP + resistor inline, includes PFET |
| `sky130_fd_pr__pfet_01v8__tt.pm3.spice` | ~770 KB | BSIM4 Level-54 PMOS model (52 geometry bins) |

## Devices Included

| Device | Model | Use in Bandgap |
|--------|-------|----------------|
| `sky130_fd_pr__pfet_01v8` | BSIM4 (Level 54) | Current mirror PMOS pair |
| `sky130_fd_pr__pnp_05v5_W3p40L3p40` | Gummel-Poon (Level 1) | BJT pair for ΔV_BE generation |
| `sky130_fd_pr__res_generic_po` | R subckt | PTAT/CTAT ratio resistors |

## Usage

In your SPICE netlist:

```spice
.include "pdk/sky130/sky130_bandgap_tt.lib"

* Then use devices normally:
XM1 drain gate source body sky130_fd_pr__pfet_01v8 w=2u l=1u
XQ1 collector base emitter substrate sky130_fd_pr__pnp_05v5_W3p40L3p40
XR1 a b sky130_fd_pr__res_generic_po w=0.35u l=10u
```

## Limitations

- **TT corner only** — no process corners (FF/SS/SF/FS)
- **No mismatch** — statistical slopes set to zero
- **No other devices** — only PMOS, PNP, and poly resistor
- For full-corner simulation, install the complete PDK:
  `pip install volare && volare enable --pdk sky130 [version]`
