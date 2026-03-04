* Brokaw Bandgap Reference — IHP SG13G2 Adaptation
* ====================================================
* This netlist targets the IHP SG13G2 130nm SiGe BiCMOS PDK, which provides
* real standalone NPN and PNP bipolar junction transistors — ideal for a
* classic Brokaw bandgap reference.
*
* HOW TO USE:
*   1. Clone the PDK:  git clone https://github.com/IHP-GmbH/IHP-Open-PDK.git
*   2. Update the .lib path below (line ~14) to your local install.
*   3. Run:  ngspice examples/ihp_sg13g2_bandgap.sp
*
* The IHP PDK is one of the few open PDKs with real bipolar devices,
* making it the best open-source match for a Brokaw topology.
*
* References:
*   - https://github.com/IHP-GmbH/IHP-Open-PDK
*   - IHP SG13G2 documentation: ihp-sg13g2/libs.doc/

.title Brokaw Bandgap Reference — IHP SG13G2

* -----------------------------------------------------------------------
* PDK model include
* -----------------------------------------------------------------------
* TODO: update this path to your local IHP PDK install
* .lib "/path/to/IHP-Open-PDK/ihp-sg13g2/libs.tech/ngspice/sg13g2_moslevel.lib" typ
* .include "/path/to/IHP-Open-PDK/ihp-sg13g2/libs.tech/ngspice/npn13G2.scs"
*
* For CI/testing without PDK, fall back to inline models (below).

* -----------------------------------------------------------------------
* Parameters
* -----------------------------------------------------------------------
.param N    = 8        $ BJT emitter area ratio [dimensionless]
.param R1   = 100k     $ Top resistor [Ω]
.param R2   = 10k      $ PTAT resistor [Ω]
.param W_P  = 4u       $ PMOS width [m]
.param L_P  = 0.5u     $ PMOS length [m] (SG13G2 min: 0.13u)

* -----------------------------------------------------------------------
* Supply — SG13G2 nominal 1.2 V (core) or 3.3 V (I/O)
* -----------------------------------------------------------------------
VDD VDD 0 DC 1.2

* -----------------------------------------------------------------------
* PMOS current mirror  (sg13_lv_pmos — low-voltage PMOS, 1.2V devices)
* -----------------------------------------------------------------------
M1 VB  VB  VDD VDD sg13_lv_pmos W={W_P} L={L_P}
M2 VC1 VB  VDD VDD sg13_lv_pmos W={W_P} L={L_P}

* -----------------------------------------------------------------------
* NPN BJTs  (npn13G2 — real SiGe HBT, fT > 200 GHz)
* This is the key advantage of this PDK: real bipolar devices.
* -----------------------------------------------------------------------
Q1 VB   VE1 0 npn13G2 Nx=1
Q2 VC1  VE2 0 npn13G2 Nx={N}

* -----------------------------------------------------------------------
* Resistor ladder  (rsil — silicided poly resistor, ~7 Ω/□)
* For high-value R1, use rppd (p+ poly, ~300 Ω/□) or rhigh (~1.4 kΩ/□)
* -----------------------------------------------------------------------
R1a VE1  VMID rhigh R={R1/2}
R1b VMID VE2  rhigh R={R1/2}
R2  VE2  0    rhigh R={R2}

* -----------------------------------------------------------------------
* Startup bias
* -----------------------------------------------------------------------
Ibias VDD VB 1u

* -----------------------------------------------------------------------
* Fallback inline models (for syntax checking when PDK not installed)
* -----------------------------------------------------------------------
.model sg13_lv_pmos PMOS (LEVEL=1 VTO=-0.35 KP=60u LAMBDA=0.08)
.model npn13G2 NPN (IS=2.5e-18 BF=200 VAF=80 RB=50 TF=0.5p)
.model rhigh R (RSH=1400)

* -----------------------------------------------------------------------
* Analysis
* -----------------------------------------------------------------------
.op

* Temperature sweep
* .dc temp -40 125 5

.end
