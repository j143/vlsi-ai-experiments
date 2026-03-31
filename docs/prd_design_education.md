# PRD: Interactive Design Space Education (Teaching / Onboarding)

**Feature area**: Frontend — new "Learn" tab  
**Target users**: Junior analog designers, students, first-time contributors  
**Status**: Implemented (v0.2)

---

## 1. Problem Statement

Junior analog designers and students have no intuition for how Brokaw bandgap
parameters trade off against each other. Trial-and-error in a SPICE GUI is slow
and opaque. There is no guided entry point in the current UI.

---

## 2. Goals

| # | Goal |
|---|------|
| G1 | A new user can understand the Brokaw formula and what each parameter controls within 3 minutes. |
| G2 | Three curated "guided experiments" let the user load pre-set design points and observe spec outcomes instantly. |
| G3 | Every design variable has a plain-language description and effect summary visible without documentation. |
| G4 | The feature requires zero backend changes — all content is static, "Try It" buttons reuse the existing `/api/simulate` path. |

---

## 3. Non-Goals

- Interactive simulation within the Learn tab itself (use the Optimization tab for that).
- Persistent progress tracking across sessions.
- Video or animated content.

---

## 4. Functional Requirements

### 4.1 New "Learn" tab

- Appears in the sidebar navigation after "Logs", labeled **Learn** with a `BookOpen` icon.
- Contains three cards stacked vertically:
  1. **Brokaw Formula Card** — shows the formula with annotated variable descriptions.
  2. **Parameter Reference Card** — table of all design variables with physical meaning, range, and effect on outputs.
  3. **Guided Experiments Card** — three numbered experiments, each with a short explanation and a "Load & Try" button.

### 4.2 Brokaw Formula Card

Displays:

```
Vref = Vbe + 2 · (R2/R1) · VT · ln(N)
```

With labeled annotations:
- **Vbe** ≈ 0.65 V — base-emitter voltage (temperature-dependent, CTAT)
- **VT** = kT/q ≈ 26 mV at 27 °C — thermal voltage (PTAT)
- **N** — BJT area ratio; primary TC trim knob
- **R2/R1** — resistor ratio; sets PTAT gain; must satisfy ≈ 10 for Vref ≈ 1.20 V at N = 8

### 4.3 Parameter Reference Card

| Parameter | Controls | Typical Range | Primary Effect |
|-----------|----------|--------------|---------------|
| N (BJT ratio) | PTAT current magnitude | 4–12 | ↑N → lower TC, lower Vref sensitivity |
| R2/R1 ratio | PTAT gain | ~10 (fixed by I-Bias knob) | Sets Vref magnitude |
| I-Bias (µA) | Quiescent current | 1–50 µA | ↑I-Bias → lower Vref error, higher Iq |
| W (M1/M2) (µm) | PMOS current mirror matching | 1–20 µm | ↑W → better matching/PSRR, larger area |

### 4.4 Guided Experiments

Each experiment has:
- A numbered title
- A one-sentence objective
- A "what to observe" note
- A **"Load & Try"** button that pre-loads `designValues` and switches the user to the Optimization tab.

**Experiment 1 — N Sweep: Find the TC Sweet Spot**  
Objective: See how increasing N from 4 to 8 to 12 changes the temperature coefficient.  
Pre-loads: N=4 (low), N=8 (nominal), N=12 (high)  
Loads nominal preset (N=8, W=5 µm, I-Bias=10 µA).

**Experiment 2 — Power Budget: Low-Iq Design**  
Objective: Find the minimum I-Bias that keeps Vref within ±10 mV.  
Pre-loads: I-Bias=2 µA (aggressive low power)  
Observe: Vref error increases significantly at low bias.

**Experiment 3 — PMOS Sizing: Maximize PSRR**  
Objective: See how larger PMOS width improves power-supply rejection.  
Pre-loads: W=15 µm (wide) with nominal N=8 and I-Bias=10 µA  
Observe: PSRR improves, quiescent current is unchanged.

---

## 5. Non-Functional Requirements

- The tab renders instantly (no API call on load).
- "Load & Try" must switch the active tab to `optimization` and update `designValues` atomically.
- Content is readable on a 1280×800 viewport without horizontal scroll.

---

## 6. Acceptance Criteria

| ID | Criterion |
|----|-----------|
| AC1 | A "Learn" nav item appears in the sidebar and is reachable by click. |
| AC2 | The Brokaw formula is visible with all four annotated variables. |
| AC3 | The parameter reference table has all four rows. |
| AC4 | All three experiments have a "Load & Try" button. |
| AC5 | Clicking "Load & Try" switches to the Optimization tab with the experiment's design values pre-loaded. |
| AC6 | Existing Playwright E2E tests (27 tests) continue to pass. |
| AC7 | `npm run build` reports 0 errors. |

---

## 7. Out of Scope

- Backend changes
- New API endpoints
- Persistent user progress or quizzes
