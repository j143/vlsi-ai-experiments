* Brokaw Bandgap Reference — SkyWater SKY130
* =========================================================
* This netlist targets the SkyWater SKY130 open-source 130nm CMOS PDK.
* It uses the bundled minimal TT-corner model extraction in pdk/sky130/.
*
* HOW TO USE:
*   Option A (bundled models — works out of the box):
*     ngspice examples/sky130_bandgap.sp
*
*   Option B (full PDK install for corner sweeps):
*     pip install volare && volare enable --pdk sky130
*     Uncomment the full-PDK .lib line below and comment out the bundled one.
*
* NOTE: SKY130 has parasitic PNP (sky130_fd_pr__pnp_05v5_W3p40L3p40),
*       not a standalone NPN. We use PNP in a modified Brokaw topology.
*       For a true NPN-based Brokaw, see ihp_sg13g2_bandgap.sp.
*
* References:
*   - https://github.com/google/skywater-pdk-libs-sky130_fd_pr (Apache 2.0)
*   - https://skywater-pdk.readthedocs.io/en/main/

.title Brokaw Bandgap Reference — SKY130

* -----------------------------------------------------------------------
* PDK model include — choose ONE option
* -----------------------------------------------------------------------
* Option A: Bundled minimal TT extraction (default, no install needed)
.include "../pdk/sky130/sky130_bandgap_tt.lib"

* Option B: Full PDK via open_pdks / volare
* .lib "/usr/share/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice" tt

* -----------------------------------------------------------------------
* Parameters
* -----------------------------------------------------------------------
.param N    = 8        $ BJT emitter area ratio [dimensionless]
.param R1   = 100k     $ Series resistor [Ω]
.param R2   = 10k      $ PTAT resistor [Ω]
.param W_P  = 4u       $ PMOS width [m]  (SKY130 min: 0.42u)
.param L_P  = 1u       $ PMOS length [m] (SKY130 min: 0.15u)

* -----------------------------------------------------------------------
* Supply — SKY130 nominal 1.8 V
* -----------------------------------------------------------------------
VDD VDD 0 DC 1.8

* -----------------------------------------------------------------------
* PMOS current mirror  (sky130_fd_pr__pfet_01v8)
* -----------------------------------------------------------------------
M1 VB  VB  VDD VDD sky130_fd_pr__pfet_01v8 W={W_P} L={L_P} m=1
M2 VC1 VB  VDD VDD sky130_fd_pr__pfet_01v8 W={W_P} L={L_P} m=1

* -----------------------------------------------------------------------
* Parasitic PNP BJTs (sky130_fd_pr__pnp_05v5_W3p40L3p40)
* Subckt terminals: collector base emitter substrate
* SKY130 PNP is a parasitic device — collector tied to substrate (gnd).
* -----------------------------------------------------------------------
XQ1 0 VE1 VB  0 sky130_fd_pr__pnp_05v5_W3p40L3p40
XQ2 0 VE2 VC1 0 sky130_fd_pr__pnp_05v5_W3p40L3p40 m={N}

* -----------------------------------------------------------------------
* Resistor ladder  (sky130_fd_pr__res_generic_po — poly resistor, ~48 Ω/□)
* -----------------------------------------------------------------------
* Use subckt form with W (width) and L (length) parameters.
* R = rp1 * L / W , where rp1 ≈ 48.2 Ω/□ at TT corner.
* For R1 = 100k: L = R1 * W / rp1 = 100e3 * 0.35e-6 / 48.2 ≈ 726u
* For R2 = 10k:  L = R2 * W / rp1 = 10e3 * 0.35e-6 / 48.2 ≈ 72.6u

XR1a VE1  VMID sky130_fd_pr__res_generic_po w=0.35u l=363u
XR1b VMID VE2  sky130_fd_pr__res_generic_po w=0.35u l=363u
XR2  VE2  0    sky130_fd_pr__res_generic_po w=0.35u l=72.6u

* -----------------------------------------------------------------------
* Startup bias
* -----------------------------------------------------------------------
Ibias VDD VB 1u

* -----------------------------------------------------------------------
* Analysis
* -----------------------------------------------------------------------
.op

* Temperature sweep for TC extraction
* .dc temp -40 125 5

* AC for PSRR
* .ac dec 10 1 100Meg

.end
