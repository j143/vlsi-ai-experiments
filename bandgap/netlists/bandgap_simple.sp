* Brokaw Bandgap Reference — Corrected Topology
* ================================================
* Topology: Brokaw cell — two NPN BJTs with COMMON BASE at VOUT (= Vref),
*           PMOS current mirror forcing equal collector currents,
*           PTAT resistor R1 between emitters, emitter resistor R2 to GND.
*
* Technology: *** PLACEHOLDER — see config/tech_placeholder.yaml ***
*             Replace .model lines below with real PDK .lib include.
*
* Design Variables (all parameterized via .param):
*   N    : BJT emitter area ratio Q2/Q1.  Sets PTAT voltage swing.
*   R1   : PTAT resistor between emitters [Ω].  Sets bias current I = VT·ln(N)/R1.
*   R2   : Emitter-to-GND resistor [Ω].  Sets Vref level (CTAT + PTAT cancellation).
*   W_P  : PMOS mirror transistor width [m].  Use large W for matching.
*   L_P  : PMOS mirror transistor length [m].  Use long L to reduce channel-length
*          modulation and improve mirror accuracy.
*
* Circuit operation:
*   Q1 (1×) and Q2 (N×) share a common base at VOUT.
*   PMOS mirror forces IC(Q1) = IC(Q2) = I.
*   PTAT condition: VBE(Q1) - VBE(Q2) = VT·ln(N) = I·R1  →  I = VT·ln(N)/R1
*   KCL at VE1: IE(Q1) + I_R1 = I_R2 = 2·I  →  VE1 = 2·I·R2
*   Vref = VOUT = VBE(Q1) + VE1 = VBE + 2·(R2/R1)·VT·ln(N)
*
* Temperature coefficient:
*   dVref/dT = dVBE/dT + 2·(R2/R1)·(k/q)·ln(N)
*   Zero-TC condition: R2/R1 ≈ |dVBE/dT| / (2·(k/q)·ln(N))
*                            ≈ 2.2 mV/°C / (2 × 0.0862 mV/°C × ln(8)) ≈ 6.1  [N=8]
*
* Default parameter sizing (R1=20 kΩ, R2=100 kΩ, N=8, IS=1e-17 placeholder):
*   I_PTAT = VT·ln(8)/20k ≈ 2.7 µA
*   VE1 = 2·2.7µA·100k ≈ 0.537 V
*   VBE(Q1) ≈ VT·ln(I/IS) ≈ 0.663 V
*   Vref ≈ 0.663 + 0.537 = 1.200 V  ✓
*   Iq = 2·I ≈ 5.4 µA  (well within 50 µA quiescent-current spec)
*
* Node inventory:
*   VOUT : Vref output = common base of Q1 and Q2
*   VB   : Q1 collector = M1 (diode-connected PMOS) drain
*   VC1  : Q2 collector = M2 (mirror copy PMOS) drain
*   VE1  : Q1 emitter (lower emitter — Q1 is 1× unit BJT)
*   VE2  : Q2 emitter (higher emitter — Q2 has larger area → lower VBE → VE2 > VE1)
*   VMID : Mid-tap of R1 (symmetric split for layout matching)
*
* Units: all values in ngspice default SI (V, A, Ω, F, H, s).
* References:
*   [1] Brokaw, A.P., "A simple three-terminal IC bandgap reference," IEEE JSSC 1974.
*   [2] Razavi, B., "Design of Analog CMOS Integrated Circuits," 2nd ed., McGraw-Hill.

.title Brokaw Bandgap Reference — Corrected Topology

* -----------------------------------------------------------------------
* Parameters (design variables — edit these to explore the design space)
* -----------------------------------------------------------------------
.param N    = 8        $ BJT emitter area ratio Q2/Q1  [dimensionless, typ 4–16]
.param R1   = 20k      $ PTAT resistor between emitters [Ω, sets I_PTAT = VT·ln(N)/R1]
.param R2   = 100k     $ Emitter-to-GND resistor [Ω, R2/R1 ≈ 5 for Vref≈1.2V]
.param W_P  = 10u      $ PMOS mirror width  [m, larger W → better matching]
.param L_P  = 2u       $ PMOS mirror length [m, longer L → lower λ, better mirror]

* -----------------------------------------------------------------------
* Supply
* -----------------------------------------------------------------------
* AC 1 enables AC analysis for PSRR measurement (does not affect DC .op).
* TODO(human): match VDD to tech_placeholder.yaml vdd_nom_V
VDD VDD 0 DC 1.8 AC 1  $ Nominal VDD = 1.8 V

* -----------------------------------------------------------------------
* PMOS current mirror (M1 = diode-connected reference, M2 = mirror copy)
* -----------------------------------------------------------------------
* TODO(human): replace 'pmos_placeholder' with actual PDK PMOS model name
M1 VB  VB  VDD VDD pmos_placeholder W={W_P} L={L_P}   $ diode-connected
M2 VC1 VB  VDD VDD pmos_placeholder W={W_P} L={L_P}   $ mirror copy

* -----------------------------------------------------------------------
* NPN BJTs — COMMON BASE = VOUT (the Vref output node)
* -----------------------------------------------------------------------
* Q1: unit (1×) BJT — collector at M1 drain (VB), base at VOUT, emitter at VE1
* Q2: N× BJT       — collector at M2 drain (VC1), base at VOUT, emitter at VE2
* VE2 > VE1 because Q2 has larger emitter area → smaller VBE for the same current.
* TODO(human): replace 'npn_placeholder' with actual PDK NPN model name
Q1 VB   VOUT VE1 npn_placeholder           $ 1× BJT: C=VB,  B=VOUT, E=VE1
Q2 VC1  VOUT VE2 npn_placeholder AREA={N}  $ N× BJT: C=VC1, B=VOUT, E=VE2

* -----------------------------------------------------------------------
* Resistors
* -----------------------------------------------------------------------
* R1 (split symmetrically for layout matching): connects VE2 to VE1.
*   PTAT voltage VT·ln(N) develops across R1; PTAT current I = VT·ln(N)/R1.
* R2: VE1 to GND.  Carries both emitter currents (IE1 + I_R1 = 2·I).
*   VE1 = 2·I·R2 sets the PTAT contribution to Vref.
* TODO(human): replace 'res_placeholder' with PDK resistor model, or use ideal R
*              For low-TC, use a poly resistor matched to the PTAT compensation.
R1a VE2  VMID res_placeholder R={R1/2}  $ upper half of R1 (VE2 side, higher potential)
R1b VMID VE1  res_placeholder R={R1/2}  $ lower half of R1 (VE1 side, lower potential)
R2  VE1  0    res_placeholder R={R2}    $ emitter-to-GND resistor (carries 2·I_PTAT)

* -----------------------------------------------------------------------
* Output node
* -----------------------------------------------------------------------
* Vref = VOUT = VBE(Q1) + VE1 = common base of Q1 and Q2.
* For production use, add a unity-gain output buffer (source follower or op-amp)
* to isolate the reference from load variations.
* TODO(human): add output buffer before tapeout

* -----------------------------------------------------------------------
* Bias / startup
* -----------------------------------------------------------------------
* Ibias provides a startup path out of the zero-current state.
* TODO(human): replace with a proper startup circuit (e.g., weak PMOS pull-up
*              plus a feedback disabling element) before tapeout.
Ibias VDD VB 1u   $ 1 µA startup bias [A] — remove once startup circuit is added

* -----------------------------------------------------------------------
* Device models (PLACEHOLDER — replace with PDK .lib include)
* -----------------------------------------------------------------------
* TODO(human): replace the .model lines below with a .lib statement, e.g.:
*   .lib "/path/to/pdk/models/spice/sky130.lib.spice" tt
*
* WARNING: The models below are LEVEL-1 generic placeholders for topology
* verification only.  They do NOT represent any real silicon process.
* Do NOT use for design sign-off.
.model npn_placeholder NPN (IS=1e-17 BF=100 VAF=50 RB=100)
.model pmos_placeholder PMOS (LEVEL=1 VTO=-0.45 KP=50u LAMBDA=0.05 GAMMA=0.5 PHI=0.6)
.model res_placeholder R

* -----------------------------------------------------------------------
* Analysis — DC operating point (always active; used by the BO sweep loop)
* -----------------------------------------------------------------------
.op

* -----------------------------------------------------------------------
* Temperature sweep (for TC extraction — enable in run_full() or uncomment here)
* -----------------------------------------------------------------------
* Sweeps Tnom from -40 °C to +125 °C in 5 °C steps.
* When enabled, use .print to output v(VOUT) at each temperature point.
* .dc temp -40 125 5
* .print dc v(VOUT)

* -----------------------------------------------------------------------
* AC analysis (for PSRR — enable in run_full() or uncomment here)
* -----------------------------------------------------------------------
* Measures how well Vref rejects VDD ripple.
* VDD source has AC 1 (above), so v(VOUT) AC magnitude = 1/PSRR.
* .ac dec 10 1 100Meg
* .print ac vdb(VOUT)

.end
