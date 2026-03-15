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
POST /api/project/save     Saves project state to results/projects/*.json.
POST /api/netlist/export   Renders a parameterized SPICE netlist.

Usage::

    python api/server.py                  # default port 5000
    python api/server.py --port 5001      # custom port
    FLASK_DEBUG=1 python api/server.py    # debug mode

CORS is enabled for all origins so the Vite dev server (port 5173) can call it.

When the environment variable ``STATIC_DIR`` is set (e.g. to
``/app/frontend/dist``), Flask will also serve the pre-built React SPA from
that directory.  All non-API routes fall through to ``index.html`` so that
client-side routing works correctly.  This is the mode used inside the
combined Docker image (``Dockerfile.app``).
"""

import argparse
import json
import logging
import math
import os
import sys
import threading
from datetime import datetime
from queue import Queue
from pathlib import Path

import numpy as np
from flask import Flask, Response, request, send_from_directory
from flask_cors import CORS

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_DATASETS_DIR = _REPO_ROOT / "datasets"

from bandgap.runner import BandgapRunner, NETLIST_TEMPLATE, _render_netlist  # noqa: E402
from layout.data_stub import LAYER_MAP, PatchConfig, generate_synthetic_patch  # noqa: E402
from layout.evaluate import run_drc  # noqa: E402
from ml.optimize import SyntheticBandgapRunner  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Optional static-file serving for the pre-built React SPA
# ---------------------------------------------------------------------------
# When STATIC_DIR is set (e.g. /app/frontend/dist) Flask serves the SPA.
# All non-API paths fall through to index.html so client-side routing works.
#
# Flask's send_from_directory resolves *relative* directory paths against the
# Flask app's root_path (the directory containing this file, i.e. api/).
# We therefore always resolve STATIC_DIR to an absolute path here.  Relative
# values are resolved against _REPO_ROOT so that
#   STATIC_DIR=frontend/dist python api/server.py
# works regardless of the current working directory.

_raw_static = os.environ.get("STATIC_DIR")
if _raw_static:
    _p = Path(_raw_static)
    _STATIC_DIR = _p if _p.is_absolute() else (_REPO_ROOT / _p).resolve()
else:
    _STATIC_DIR = None  # type: ignore[assignment]


if _STATIC_DIR:
    logger.info("Serving React SPA from %s", _STATIC_DIR)

    @app.get("/", defaults={"path": ""})
    @app.get("/<path:path>")
    def serve_spa(path: str):
        """Serve pre-built React SPA.

        Flask's routing engine prioritises specific patterns (e.g. ``/api/*``)
        over this catch-all, so all API endpoints registered below continue to
        work as expected.
        """
        if path:
            # Resolve the candidate path and verify it stays inside _STATIC_DIR
            # to prevent directory-traversal attacks.
            try:
                candidate = (_STATIC_DIR / path).resolve()
                if candidate.is_relative_to(_STATIC_DIR) and candidate.is_file():
                    return send_from_directory(str(_STATIC_DIR), path)
            except (ValueError, OSError):
                pass
        return send_from_directory(str(_STATIC_DIR), "index.html")


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
_PROJECTS_DIR = _REPO_ROOT / "results" / "projects"
_DATASETS_DIR = _REPO_ROOT / "datasets"
_NETLISTS_DIR = _REPO_ROOT / "bandgap" / "netlists"

# Single shared runner instance (stateless — safe to share across requests).
# BandgapRunner.__init__ only reads files; it does NOT invoke ngspice, so this
# is safe to call at import time even when ngspice is absent.
_runner = BandgapRunner()


def _get_runner(use_synthetic: bool = False):
    """Return runner selected by caller.

    Real ngspice flow is the default path. Synthetic runner is only used when
    explicitly requested by the client.
    """
    if use_synthetic:
        return SyntheticBandgapRunner()
    return _runner


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_preset(preset_id: str | None) -> dict | None:
    """Return the preset dict for *preset_id*, or *None* if not found."""
    if not preset_id:
        return None
    return next((p for p in _PRESETS if p["id"] == preset_id), None)


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
    use_synthetic = bool(body.get("use_synthetic", False))

    if not isinstance(params, dict):
        return Response(
            json.dumps({"error": "params must be a JSON object"}),
            status=400,
            mimetype="application/json",
        )

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

    if not use_synthetic and not _runner.is_ngspice_available():
        return Response(
            json.dumps(
                {
                    "error": (
                        "ngspice is not available. Real ngspice-backed simulation is the default. "
                        "Set use_synthetic=true to run the synthetic fallback."
                    )
                }
            ),
            status=503,
            mimetype="application/json",
        )

    result = _get_runner(use_synthetic=use_synthetic).run(params)
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
            "budget":  50,
            "n_init":  10,
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
    budget = int(body.get("budget", 50))
    n_init = int(body.get("n_init", 10))
    seed = int(body.get("seed", 42))
    use_synthetic = bool(body.get("use_synthetic", False))
    early_stop = bool(body.get("early_stop", False))
    # Accept preset id; preset values are used as defaults (body overrides them)
    preset = _resolve_preset(body.get("preset"))
    weights = None
    if preset:
        budget = int(body.get("budget", preset["budget"]))
        n_init = int(body.get("n_init", preset["n_init"]))
        early_stop = bool(body.get("early_stop", preset.get("early_stop", False)))
        weights = preset.get("weights")

    budget = max(1, min(budget, _MAX_BUDGET))  # prevent excessive computation
    n_init = max(1, min(n_init, budget))

    if not use_synthetic and not _runner.is_ngspice_available():
        return Response(
            json.dumps(
                {
                    "error": (
                        "ngspice is not available. "
                        "Real ngspice-backed optimization is the default. "
                        "Set use_synthetic=true to run the synthetic fallback."
                    )
                }
            ),
            status=503,
            mimetype="application/json",
        )

    try:
        # Import here to avoid startup cost when only /status or /simulate is used
        from ml.optimize import BayesianOptimizer  # noqa: PLC0415

        runner = _get_runner(use_synthetic=use_synthetic)
        opt = BayesianOptimizer(
            runner=runner, budget=budget, n_init=n_init, early_stop=early_stop,
            weights=weights,
        )
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

    Query params (all optional): budget, n_init, seed, early_stop, preset.
    Emits events:
      - progress: per-iteration update (entry + running summary)
      - final: full optimization payload, same shape as /api/optimize
      - api_error: optimizer failure details
      - done: stream completion marker
    """
    budget = int(request.args.get("budget", 50))
    n_init = int(request.args.get("n_init", 10))
    seed = int(request.args.get("seed", 42))
    use_synthetic = _parse_bool(request.args.get("use_synthetic"), default=False)
    early_stop = _parse_bool(request.args.get("early_stop"), default=False)
    # Accept preset id; preset values are used as defaults (query params override)
    preset = _resolve_preset(request.args.get("preset"))
    weights = None
    if preset:
        budget = int(request.args.get("budget", preset["budget"]))
        n_init = int(request.args.get("n_init", preset["n_init"]))
        early_stop = _parse_bool(
            request.args.get("early_stop"), default=preset.get("early_stop", False)
        )
        weights = preset.get("weights")

    budget = max(1, min(budget, _MAX_BUDGET))
    n_init = max(1, min(n_init, budget))

    if not use_synthetic and not _runner.is_ngspice_available():
        return Response(
            json.dumps(
                {
                    "error": (
                        "ngspice is not available. "
                        "Real ngspice-backed optimization is the default. "
                        "Set use_synthetic=true to run the synthetic fallback."
                    )
                }
            ),
            status=503,
            mimetype="application/json",
        )

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(_sanitize_json(data))}\n\n"

    def event_stream():
        queue: Queue[tuple[str, dict]] = Queue()

        def worker() -> None:
            try:
                from ml.optimize import BayesianOptimizer  # noqa: PLC0415

                runner = _get_runner(use_synthetic=use_synthetic)
                opt = BayesianOptimizer(
                    runner=runner, budget=budget, n_init=n_init, early_stop=early_stop,
                    weights=weights,
                )

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


@app.get("/api/accuracy")
def accuracy():
    """Evaluate surrogate accuracy against synthetic or real SKY130 ground truth.

    Trains a GP surrogate on *n_train* samples, then evaluates on *n_test*
    held-out samples.  Reports the fraction of test points where
    |surrogate_vref − ground_truth_vref| ≤ tolerance_mV.

    Query params (all optional):
      - source (str, default "synthetic"): "synthetic" uses the Brokaw formula;
        "real" uses datasets/bandgap_sweep_real_sky130.csv.
      - n_test (int, default 20): held-out evaluation points.
      - n_train (int, default 200): training set size.
      - seed (int, default 42): random seed for reproducibility.
      - tolerance_mV (float, default 10.0): pass/fail threshold in mV.

    Response::

        {
            "ok": true,
            "source": "synthetic",
            "n_test": 20,
            "n_train": 200,
            "tolerance_mV": 10.0,
            "accuracy_pct": 92.0,
            "mean_error_mV": 4.3,
            "mean_std_mV": 3.1,
            "confidence": "High"
        }
    """
    source = request.args.get("source", "synthetic").lower()
    n_test = max(5, min(int(request.args.get("n_test", 20)), 100))
    n_train = max(20, min(int(request.args.get("n_train", 200)), 500))
    seed = int(request.args.get("seed", 42))
    tolerance_mV = float(request.args.get("tolerance_mV", 10.0))

    try:
        from ml.surrogate import (  # noqa: PLC0415
            GaussianProcessSurrogate,
            _generate_synthetic_data,
            FEATURES,
            accuracy_confidence,
        )
        import pandas as pd  # noqa: PLC0415

        if source == "real":
            csv_path = _DATASETS_DIR / "bandgap_sweep_real_sky130.csv"
            if not csv_path.exists():
                return Response(
                    json.dumps({"error": "Real SKY130 dataset not found. Run data generation first."}),
                    status=404,
                    mimetype="application/json",
                )
            df_all = pd.read_csv(csv_path)
            # Drop rows with missing or non-physical vref (plausible range: 0–3.5 V)
            df_all = df_all.dropna(subset=["vref_V"])
            df_all = df_all[(df_all["vref_V"] >= 0.0) & (df_all["vref_V"] < 3.5)]
            if len(df_all) < n_train + n_test:
                # Use all available rows; adjust sizes proportionally
                total = len(df_all)
                n_train = max(5, total * n_train // (n_train + n_test))
                n_test = max(5, total - n_train)
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(df_all))
            df_train = df_all.iloc[idx[:n_train]]
            df_test = df_all.iloc[idx[n_train:n_train + n_test]]
        else:
            source = "synthetic"  # normalise any unrecognised value
            df_all = _generate_synthetic_data(n=n_train + n_test, seed=seed)
            df_train = df_all.iloc[:n_train]
            df_test = df_all.iloc[n_train:]

        X_train = df_train[FEATURES].values
        y_train = df_train["vref_V"].values
        X_test = df_test[FEATURES].values
        y_test = df_test["vref_V"].values

        model = GaussianProcessSurrogate(n_restarts=3)
        model.fit(X_train, y_train)

        mean, std = model.predict_with_uncertainty(X_test)
        errors_mV = np.abs(mean - y_test) * 1000
        within_tol = float((errors_mV <= tolerance_mV).mean())
        mean_error_mV = float(errors_mV.mean())
        mean_std_mV = float(std.mean() * 1000)

        confidence = accuracy_confidence(within_tol)

        logger.info(
            "Surrogate accuracy [%s]: %.0f%% within ±%.0f mV (%d test pts)",
            source, within_tol * 100, tolerance_mV, n_test,
        )

        return _safe_json({
            "ok": True,
            "source": source,
            "n_test": int(n_test),
            "n_train": int(n_train),
            "tolerance_mV": tolerance_mV,
            "accuracy_pct": round(within_tol * 100, 1),
            "mean_error_mV": round(mean_error_mV, 2),
            "mean_std_mV": round(mean_std_mV, 2),
            "confidence": confidence,
        })
    except Exception as exc:
        logger.exception("Accuracy evaluation error")
        return Response(
            json.dumps({"error": str(exc)}),
            status=500,
            mimetype="application/json",
        )


# ---------------------------------------------------------------------------
# /api/presets
# ---------------------------------------------------------------------------

# Named design presets.  Each preset is a bundle of optimizer settings that
# represent a common design intent.  The budget/n_init values are treated as
# *hints* — the client may override them.
_PRESETS = [
    {
        "id": "balanced",
        "label": "Balanced",
        "description": "Balanced trade-off: Vref accuracy vs. power (default)",
        "budget": 30,
        "n_init": 10,
        "early_stop": False,
        "weights": {"vref": 1.0, "iq": 0.3, "psrr": 0.2},
    },
    {
        "id": "low_power",
        "label": "Low Power",
        "description": "Minimize quiescent current; accepts slightly relaxed Vref tolerance",
        "budget": 25,
        "n_init": 8,
        "early_stop": False,
        "weights": {"vref": 1.0, "iq": 1.5, "psrr": 0.2},
    },
    {
        "id": "tight_vref",
        "label": "Tight Vref",
        "description": "Maximize Vref accuracy (±10 mV); accepts higher power",
        "budget": 40,
        "n_init": 12,
        "early_stop": False,
        "weights": {"vref": 2.0, "iq": 0.1, "psrr": 0.1},
    },
]


@app.get("/api/presets")
def presets():
    """Return named design presets for quick-start optimizer configuration.

    Response::

        {
            "presets": [
                {
                    "id": "balanced",
                    "label": "Balanced",
                    "description": "...",
                    "budget": 30,
                    "n_init": 10,
                    "early_stop": false
                },
                ...
            ]
        }
    """
    return _safe_json({"presets": _PRESETS})


# ---------------------------------------------------------------------------
# /api/layout/preview
# ---------------------------------------------------------------------------

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


@app.post("/api/project/save")
def project_save():
    """Persist frontend project state as JSON under results/projects."""
    body = request.get_json(silent=True) or {}
    state = body.get("state", {})
    project_name = str(body.get("project_name", "vlsi_ai_project")).strip() or "vlsi_ai_project"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in project_name)
    out_path = _PROJECTS_DIR / f"{safe_name}_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "project_name": project_name,
        "saved_at": datetime.now().isoformat(),
        "state": state,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    return _safe_json(
        {
            "ok": True,
            "project_name": project_name,
            "saved_path": str(out_path.relative_to(_REPO_ROOT)),
        }
    )


@app.get("/api/project/files")
def project_files():
    """List project files for lightweight UI navigation actions."""
    kind = str(request.args.get("kind", "datasets")).strip().lower()
    if kind == "datasets":
        base = _DATASETS_DIR
        patterns = ["*.csv", "*.parquet", "*.npy"]
    elif kind == "netlists":
        base = _NETLISTS_DIR
        patterns = ["*.sp", "*.cir"]
    else:
        return Response(
            json.dumps({"error": "kind must be one of: datasets, netlists"}),
            status=400,
            mimetype="application/json",
        )

    files: list[str] = []
    if base.exists():
        for pattern in patterns:
            files.extend(sorted(path.name for path in base.glob(pattern) if path.is_file()))

    return _safe_json({"ok": True, "kind": kind, "files": files})


@app.post("/api/netlist/export")
def netlist_export():
    """Render and return a parameterized SPICE netlist from given params."""
    body = request.get_json(silent=True) or {}
    params = body.get("params", {})
    if not isinstance(params, dict) or not params:
        return Response(
            json.dumps({"error": "params must be a non-empty JSON object"}),
            status=400,
            mimetype="application/json",
        )

    cleaned: dict[str, float] = {}
    for key, value in params.items():
        try:
            cleaned[key] = float(value)
        except (TypeError, ValueError):
            return Response(
                json.dumps({"error": f"Non-numeric value for param '{key}': {value!r}"}),
                status=400,
                mimetype="application/json",
            )

    netlist_text = _render_netlist(Path(NETLIST_TEMPLATE), cleaned)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bandgap_export_{timestamp}.sp"

    return _safe_json(
        {
            "ok": True,
            "filename": filename,
            "netlist_text": netlist_text,
            "params": cleaned,
        }
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="VLSI-AI API server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)),
                        help="Port to listen on (default: 5000, or $PORT env var)")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"),
                        help="Host to bind to (default: 127.0.0.1, or $HOST env var)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logger.info("Starting VLSI-AI API server on %s:%d", args.host, args.port)
    if _STATIC_DIR:
        logger.info("Combined mode: serving React SPA from %s", _STATIC_DIR)
    app.run(host=args.host, port=args.port, debug=False)
