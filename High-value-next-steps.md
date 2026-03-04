High-Value Next Steps (Prioritized)

1. Wire a Real API Backend (Highest ROI)
The biggest gap is the frontend running on synthetic data. Add a lightweight FastAPI server (api/server.py) that exposes:
​

POST /optimize → calls ml/optimize.py → returns candidates JSON

GET /simulate/{id} → calls bandgap/runner.py → returns Vref, PSRR, Iq
This alone transforms the project from a mockup into a working tool.

2. Integrate SkyWater SKY130 PDK
Fill in config/tech_placeholder.yaml with real SKY130 values and swap .model lines in bandgap_simple.sp with .lib includes. This is the prerequisite for Milestone 1 — without real process data, surrogate training has no ground truth.
​

3. Fix Optimizer Convergence
The best result is 1181.6mV vs. 1200mV target — 18.4mV off, exceeding the ±10mV tolerance. Diagnose:
​

The N range (4–12) and W range (1–20µm) may not cover the true optimum; expand the search bounds

Check that the GP kernel (RBF vs. Matérn) is appropriate for this response surface

Increase budget from 20 to at least 50 iterations for meaningful convergence

4. Show All Four Spec Metrics in the UI
PSRR and Iq are already in the specs.yaml and likely computed in runner.py but the frontend renders —. Add these to the candidate row object returned by the optimizer and display them in the "Selected" panel. This directly shows the multi-objective tradeoff which is the project's core value proposition.
​

5. Implement the Layout Viewer Tab
Milestone 3 targets a UNet patch completion model. Start the layout tab with a simple GDS/patch rasterization viewer using the synthetic data from layout/data_stub.py. Even a 2D colored grid showing layer types (metal, poly, diffusion) adds huge demo value.
​

6. Add Simulation Progress Streaming
The 0/20 spec-pass counter is static. Wire a WebSocket or SSE endpoint so that as the optimizer runs, the convergence chart updates live, candidates populate row-by-row, and the pass counter increments in real time. This makes the tool feel alive.
​

7. Multi-Corner Joint Optimization (Future)
Currently corners are checked post-hoc. The highest research value step is making corner performance part of the objective function — jointly optimizing Vref@TT while constraining max deviation at FF/SS corners. This is the "Monte Carlo integration in surrogate training" item in your ROADMAP future ideas and directly maps to publishable results.
​