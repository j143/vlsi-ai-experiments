* Brokaw Bandgap Reference — Simple Netlist
* ============================================
* Topology: Brokaw cell with two BJTs (Q1 1×, Q2 N×), PMOS current mirror load,
*           and resistor ladder R1/R2 setting Vref.
*
* Technology: *** PLACEHOLDER — see config/tech_placeholder.yaml ***
*             Replace .model lines below with real PDK models.
*
* Design Variables (all parameterized via .param):
*   N    : BJT emitter area ratio Q2/Q1.  Sets PTAT swing.
*   R1   : Top resistor [Ω].  Sets overall bias current.
*   R2   : Bottom resistor [Ω].  Sets PTAT portion of Vref.
*   W_P  : PMOS mirror width [µm].
*   L_P  : PMOS mirror length [µm].
*
* Operating point (illustrative, NOT silicon-verified):
*   Vref ≈ Vbe + (R2/R1)*VT*ln(N)
*   With N=8, R2/R1≈10, Vbe≈0.65V: Vref ≈ 0.65 + 10*(26mV*2.08) ≈ 1.19V
*
* Units: all values in ngspice default SI (V, A, Ω, F, H, s).
* References:
*   [1] Brokaw, A.P., "A simple three-terminal IC bandgap reference," IEEE JSSC 1974.

.title Brokaw Bandgap Reference — Placeholder

* -----------------------------------------------------------------------
* Parameters (design variables — edit these to explore the design space)
* -----------------------------------------------------------------------
.param N    = 8        $ BJT emitter area ratio (Q2 area / Q1 area) [dimensionless]
.param R1   = 100k     $ Series resistor — sets bias current [Ω]
.param R2   = 10k      $ PTAT resistor in feedback path [Ω]
.param W_P  = 4u       $ PMOS mirror transistor width [m]
.param L_P  = 1u       $ PMOS mirror transistor length [m]  (use long L for matching)

* -----------------------------------------------------------------------
* Supply
* -----------------------------------------------------------------------
VDD VDD 0 DC 1.8       $ Nominal VDD = 1.8 V  TODO(human): match tech_placeholder.yaml vdd_nom_V

* -----------------------------------------------------------------------
* PMOS current mirror (M1 = diode-connected reference, M2 = mirror copy)
* -----------------------------------------------------------------------
* TODO(human): replace 'pmos_placeholder' with actual PDK PMOS model name
M1 VB  VB  VDD VDD pmos_placeholder W={W_P} L={L_P}   $ diode-connected
M2 VC1 VB  VDD VDD pmos_placeholder W={W_P} L={L_P}   $ mirror copy

* -----------------------------------------------------------------------
* NPN BJTs (Q2 has N× emitter area — sets PTAT voltage)
* -----------------------------------------------------------------------
* TODO(human): replace 'npn_placeholder' with actual PDK NPN model name
Q1 VB   VE1 0 npn_placeholder          $ unit BJT
Q2 VC1  VE2 0 npn_placeholder AREA={N} $ N× BJT for PTAT

* -----------------------------------------------------------------------
* Resistor ladder
* -----------------------------------------------------------------------
* TODO(human): replace 'res_placeholder' with actual PDK resistor model name
*              (or use ideal R if PDK resistors not available yet)
R1a VE1  VMID res_placeholder R={R1/2}   $ upper half of R1 (symmetric layout)
R1b VMID VE2  res_placeholder R={R1/2}   $ lower half of R1
R2  VE2  0    res_placeholder R={R2}     $ PTAT resistor to GND

* -----------------------------------------------------------------------
* Output node (Vref = VE1 minus ground drop across R2 when current flows)
* For a real Brokaw, Vref is taken at a buffered output. Here we expose VE1
* as the reference node for simplicity. Add an op-amp buffer for real use.
* -----------------------------------------------------------------------
* Vref = VE1 in this simplified netlist.

* -----------------------------------------------------------------------
* Bias / startup (minimal — a real design needs a proper startup circuit)
* -----------------------------------------------------------------------
* TODO(human): add proper startup circuit before tapeout
* Ibias provides startup path; remove or replace once startup circuit is added.
Ibias VDD VB 1u   $ 1 µA startup bias  [A]

* -----------------------------------------------------------------------
* Device models (PLACEHOLDER — replace with PDK .lib include)
* -----------------------------------------------------------------------
* TODO(human): replace the .model lines below with a .lib statement pointing
*              to your PDK model file, e.g.:
*   .lib "/path/to/pdk/models/spice/sky130.lib.spice" tt
*
* The models below are generic LEVEL-1 placeholders for syntax checking only.
* They produce INCORRECT electrical behavior — do not use for any real evaluation.
.model npn_placeholder NPN (IS=1e-17 BF=100 VAF=50 RB=100)
.model pmos_placeholder PMOS (LEVEL=1 VTO=-0.45 KP=50u LAMBDA=0.05 GAMMA=0.5 PHI=0.6)
.model res_placeholder R

* -----------------------------------------------------------------------
* Analysis — DC operating point
* -----------------------------------------------------------------------
.op

* -----------------------------------------------------------------------
* Temperature sweep (for TC extraction)
* -----------------------------------------------------------------------
* Uncomment to sweep temperature from -40°C to 125°C in 5°C steps:
* .dc temp -40 125 5

* -----------------------------------------------------------------------
* AC analysis (for PSRR)
* -----------------------------------------------------------------------
* Uncomment to run AC analysis and measure PSRR at VDD:
* .ac dec 10 1 100Meg
* .measure AC psrr_db FIND vdb(VE1)-vdb(VDD) AT=1

.end
