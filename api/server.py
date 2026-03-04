"""
api/server.py — Flask REST API for VLSI-AI Design Studio
=========================================================
Exposes the bandgap simulation and optimization back-end to the React frontend.

Endpoints
---------
GET  /api/status           Returns server health and ngspice availability.
POST /api/simulate         Runs a single ngspice simulation for given params.
POST /api/optimize         Runs the Bayesian optimization loop and returns results.
GET  /api/optimize/stream  Streams optimizer progress via SSE.
GET  /api/layout/preview   Returns synthetic layout patch + DRC summary.

Usage::

    python api/server.py                  # default port 5000
    python api/server.py --port 5001      # custom port
    FLASK_DEBUG=1 python api/server.py    # debug mode

CORS is enabled for all origins so the Vite dev server (port 5173) can call it.
"""

import argparse
import json
import logging
import math
import sys
import threading
from queue import Queue
from pathlib import Path

import numpy as np
from flask import Flask, Response, request
from flask_cors import CORS

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from bandgap.runner import BandgapRunner  # noqa: E402
from layout.data_stub import LAYER_MAP, PatchConfig, generate_synthetic_patch  # noqa: E402
from layout.evaluate import run_drc  # noqa: E402
from ml.optimize import SyntheticBandgapRunner  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


def _sanitize_json(data):
    """Return a JSON-safe object, replacing NaN/Inf with null-equivalent.

    Python's ``json`` module emits ``NaN`` and ``Infinity`` for the
    corresponding float values, but these are not valid JSON and cause
    ``JSON.parse`` to fail in JavaScript.
    """

    def _sanitize(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    return _sanitize(data)


def _safe_json(data) -> Response:
    """Serialize *data* to a JSON Response, replacing NaN/Inf with null.

    Python's ``json`` module emits ``NaN`` and ``Infinity`` for the
    corresponding float values, but these are not valid JSON and cause
    ``JSON.parse`` to fail in JavaScript.  We do a recursive walk and
    replace them with ``None`` (→ JSON ``null``) before serialising.
    """

    return Response(
        json.dumps(_sanitize_json(data)),
        status=200,
        mimetype="application/json",
    )

# Maximum number of ngspice/surrogate calls per optimization request.
# Prevents runaway computation in the synchronous request handler.
_MAX_BUDGET = 100

# Single shared runner instance (stateless — safe to share across requests).
# BandgapRunner.__init__ only reads files; it does NOT invoke ngspice, so this
# is safe to call at import time even when ngspice is absent.
_runner = BandgapRunner()


def _get_runner(use_synthetic: bool = False):
    """Return the real runner if ngspice is available, otherwise the synthetic one."""
    if use_synthetic or not _runner.is_ngspice_available():
        return SyntheticBandgapRunner()
    return _runner


# ---------------------------------------------------------------------------
# /api/status
# ---------------------------------------------------------------------------

@app.get("/api/status")
def status():
    """Return server health and whether ngspice is on PATH."""
    return _safe_json(
        {
            "ok": True,
            "ngspice_available": _runner.is_ngspice_available(),
        }
    )


# ---------------------------------------------------------------------------
# /api/simulate
# ---------------------------------------------------------------------------

@app.post("/api/simulate")
def simulate():
    """Run a single ngspice simulation.

    Request body (JSON)::

        {
            "params": {
                "N":   8,
                "R1":  100000,
                "R2":  10000,
                "W_P": 4e-6,
                "L_P": 1e-6
            }
        }

    Response::

        {
            "params":       { ... },
            "vref_V":       1.2,
            "tc_ppm_C":     null,
            "iq_uA":        9.8,
            "spec_checks":  { "vref": true, "tc": false, "iq": true },
            "error":        null
        }
    """
    body = request.get_json(silent=True) or {}
    params = body.get("params", {})

    if not isinstance(params, dict):
        return Response(json.dumps({"error": "params must be a JSON object"}), status=400, mimetype="application/json")

    # Validate that all values are numeric
    for k, v in params.items():
        try:
            float(v)
        except (TypeError, ValueError):
            return Response(
                json.dumps({"error": f"Non-numeric value for param '{k}': {v!r}"}),
                status=400,
                mimetype="application/json",
            )

    result = _get_runner().run(params)
    # raw_output can be large — omit it from the API response
    result.pop("raw_output", None)
    return _safe_json(result)


# ---------------------------------------------------------------------------
# /api/optimize
# ---------------------------------------------------------------------------

@app.post("/api/optimize")
def optimize():
    """Run the Bayesian optimization loop.

    Request body (JSON, all fields optional)::

        {
            "budget":  20,
            "n_init":  5,
            "seed":    42
        }

    Response::

        {
            "best_params":     { "N": 8, "R1": ..., ... },
            "best_vref_V":     1.2,
            "n_simulations":   20,
            "n_spec_pass":     15,
            "spec_pass_rate":  0.75,
            "history": [
                {
                    "iteration": 0,
                    "params":    { ... },
                    "vref_V":    1.198,
                    "error_V":   0.002,
                    "spec_vref_pass": true,
                    "source":    "lhs"
                },
                ...
            ],
            "convergence": [
                { "iter": 1, "best_error_V": 0.045 },
                ...
            ],
            "error": null
        }

    Note: ngspice must be installed for real simulation. If it is not, every
    simulated point will contain an error string but the optimizer still runs.
    """
    body = request.get_json(silent=True) or {}
    budget = int(body.get("budget", 20))
    n_init = int(body.get("n_init", 5))
    seed = int(body.get("seed", 42))

    budget = max(1, min(budget, _MAX_BUDGET))  # prevent excessive computation
    n_init = max(1, min(n_init, budget))

    try:
        # Import here to avoid startup cost when only /status or /simulate is used
        from ml.optimize import BayesianOptimizer  # noqa: PLC0415

        runner = _get_runner()
        opt = BayesianOptimizer(runner=runner, budget=budget, n_init=n_init)
        result = opt.run(seed=seed)
    except Exception as exc:
        logger.exception("Optimizer error")
        return Response(
            json.dumps({"error": str(exc)}),
            status=500,
            mimetype="application/json",
        )

    # Build convergence curve: running best |vref - target| per iteration
    vref_target = opt.specs["vref"]["target_V"]
    convergence = []
    best_so_far = float("inf")
    best_std_so_far = None
    for entry in result.history:
        vref = entry.get("vref_V")
        if vref is not None:
            err = abs(vref - vref_target)
            if err < best_so_far:
                best_so_far = err
                best_std_so_far = entry.get("pred_error_std_V")
        convergence.append(
            {
                "iter": entry.get("iteration", len(convergence)),
                "best_error_V": best_so_far if best_so_far < float("inf") else None,
                "best_error_std_V": best_std_so_far,
            }
        )

    payload = result.to_dict()
    payload["convergence"] = convergence
    return _safe_json(payload)


@app.get("/api/optimize/stream")
def optimize_stream():
    """Stream optimizer progress via Server-Sent Events (SSE).

    Query params (all optional): budget, n_init, seed.
    Emits events:
      - progress: per-iteration update (entry + running summary)
      - final: full optimization payload, same shape as /api/optimize
      - api_error: optimizer failure details
      - done: stream completion marker
    """
    budget = int(request.args.get("budget", 20))
    n_init = int(request.args.get("n_init", 5))
    seed = int(request.args.get("seed", 42))

    budget = max(1, min(budget, _MAX_BUDGET))
    n_init = max(1, min(n_init, budget))

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(_sanitize_json(data))}\n\n"

    def event_stream():
        queue: Queue[tuple[str, dict]] = Queue()

        def worker() -> None:
            try:
                from ml.optimize import BayesianOptimizer  # noqa: PLC0415

                runner = _get_runner()
                opt = BayesianOptimizer(runner=runner, budget=budget, n_init=n_init)

                def on_progress(entry: dict, summary: dict) -> None:
                    queue.put(
                        (
                            "progress",
                            {
                                "entry": entry,
                                "iteration": entry.get("iteration", 0),
                                **summary,
                            },
                        )
                    )

                result = opt.run(seed=seed, progress_callback=on_progress)

                vref_target = opt.specs["vref"]["target_V"]
                convergence = []
                best_so_far = float("inf")
                best_std_so_far = None
                for entry in result.history:
                    vref = entry.get("vref_V")
                    if vref is not None:
                        err = abs(vref - vref_target)
                        if err < best_so_far:
                            best_so_far = err
                            best_std_so_far = entry.get("pred_error_std_V")
                    convergence.append(
                        {
                            "iter": entry.get("iteration", len(convergence)),
                            "best_error_V": best_so_far if best_so_far < float("inf") else None,
                            "best_error_std_V": best_std_so_far,
                        }
                    )

                payload = result.to_dict()
                payload["convergence"] = convergence
                queue.put(("final", payload))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Optimizer streaming error")
                queue.put(("api_error", {"error": str(exc)}))
            finally:
                queue.put(("done", {"ok": True}))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            event_name, payload = queue.get()
            yield _sse(event_name, payload)
            if event_name == "done":
                break

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/layout/preview")
def layout_preview():
    """Return a synthetic layout patch for UI visualization.

    Query params:
      - seed (int, optional): random seed for deterministic preview.
      - patch_size (int, optional): square patch dimension, default 32.
    """
    seed = int(request.args.get("seed", 42))
    patch_size = max(16, min(int(request.args.get("patch_size", 32)), 64))

    cfg = PatchConfig(patch_size=patch_size)
    rng = np.random.default_rng(seed)
    patch = generate_synthetic_patch(cfg, rng)
    drc = run_drc(np.expand_dims(patch, axis=0))

    return _safe_json(
        {
            "patch_size": patch_size,
            "n_layers": int(patch.shape[0]),
            "layer_map": {str(k): v for k, v in LAYER_MAP.items()},
            "patch": patch.tolist(),
            "drc": drc,
        }
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="VLSI-AI API server")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logger.info("Starting VLSI-AI API server on %s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=False)
