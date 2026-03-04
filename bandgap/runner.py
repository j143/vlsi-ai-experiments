"""
bandgap/runner.py — ngspice Runner for Bandgap Reference
=========================================================
Provides functions to:
  - Render a parameterized SPICE netlist with a specific design point.
  - Run ngspice and capture output.
  - Parse ngspice output to extract key metrics (Vref, TC, PSRR, etc.).
  - Check results against specs defined in bandgap/specs.yaml.
  - Run a full multi-analysis verification (DC op + temperature sweep).

Usage example (fast single-point for BO sweep)::

    from bandgap.runner import BandgapRunner
    runner = BandgapRunner()
    result = runner.run({"N": 8, "R1": 20e3, "R2": 100e3, "W_P": 10e-6, "L_P": 2e-6})
    print(result)  # dict with vref_V, iq_uA, spec_checks, ...

Usage example (full verification of a candidate design)::

    full = runner.run_full({"N": 8, "R1": 20e3, "R2": 100e3, "W_P": 10e-6, "L_P": 2e-6})
    print(full)  # adds tc_ppm_C, ptat_swing_mV, headroom checks

Requires ngspice to be installed and on PATH.
Set environment variable NGSPICE_BIN to override the ngspice binary path.
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default path to the golden netlist template
_REPO_ROOT = Path(__file__).parent.parent
NETLIST_TEMPLATE = _REPO_ROOT / "bandgap" / "netlists" / "bandgap_simple.sp"
SPECS_FILE = _REPO_ROOT / "bandgap" / "specs.yaml"


def _find_ngspice() -> str:
    """Return the path to the ngspice binary, or raise if not found."""
    override = os.environ.get("NGSPICE_BIN")
    if override:
        return override
    found = shutil.which("ngspice")
    if found is None:
        raise FileNotFoundError(
            "ngspice binary not found on PATH. "
            "Install ngspice or set NGSPICE_BIN environment variable."
        )
    return found


def _render_netlist(template_path: Path, params: dict[str, Any]) -> str:
    """Replace .param values in the netlist with the provided parameters.

    Parameters
    ----------
    template_path:
        Path to the SPICE netlist template.
    params:
        Dict mapping parameter names to numeric values (SI units).

    Returns
    -------
    str
        Modified netlist text with updated .param lines.
    """
    text = template_path.read_text()
    for name, value in params.items():
        # Match lines like: .param N    = 8
        # Replace value after '=' with the new value.
        pattern = rf"(\.param\s+{re.escape(name)}\s*=\s*)[^\s$]+"
        replacement = rf"\g<1>{value}"
        new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        if new_text == text:
            logger.warning("Parameter '%s' not found in netlist template.", name)
        text = new_text
    return text


def _parse_op_output(ngspice_output: str) -> dict[str, float]:
    """Parse ngspice .op output to extract node voltages and device currents.

    Parameters
    ----------
    ngspice_output:
        Raw stdout/stderr from ngspice.

    Returns
    -------
    dict
        Mapping of variable names to float values.
    """
    results: dict[str, float] = {}
    # ngspice prints operating point as:  v(node) = <value>  or  i(source) = <value>
    pattern = re.compile(r"(v\(\w+\)|i\(\w+\))\s*=\s*([-+eE\d.]+)", re.IGNORECASE)
    for match in pattern.finditer(ngspice_output):
        var_name = match.group(1).lower()
        try:
            results[var_name] = float(match.group(2))
        except ValueError:
            pass
    return results


def _parse_meas_output(ngspice_output: str) -> dict[str, float]:
    """Parse .meas results from ngspice batch output.

    ngspice prints measurement results as::

        vref_tnom           =  1.20000e+00 at temp =  2.70000e+01
        vref_max            =  1.20300e+00 at temp =  1.25000e+02
        vref_min            =  1.19600e+00 at temp = -4.00000e+01

    Parameters
    ----------
    ngspice_output:
        Raw stdout/stderr from ngspice.

    Returns
    -------
    dict
        Mapping of measurement name (lowercase) to float value.
        Only the primary measurement value is returned (the ``at …`` clause
        is ignored).
    """
    results: dict[str, float] = {}
    # Match lines like:  vref_tnom   =  1.20000e+00 ...
    # A measurement name is an identifier at the start of a line (possibly with leading
    # whitespace), followed by '=', then a number.
    pattern = re.compile(
        r"^\s*([a-z_]\w*)\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
        re.IGNORECASE | re.MULTILINE,
    )
    for match in pattern.finditer(ngspice_output):
        name = match.group(1).lower()
        try:
            results[name] = float(match.group(2))
        except ValueError:
            pass
    return results


def _compute_tc(
    vref_min: float,
    vref_max: float,
    vref_nom: float,
    temp_min_C: float,
    temp_max_C: float,
) -> float:
    """Compute temperature coefficient from min/max Vref over a temperature range.

    Uses the box method::

        TC = (Vref_max - Vref_min) / (Vref_nom × ΔT) × 1e6  [ppm/°C]

    Parameters
    ----------
    vref_min:
        Minimum Vref over the temperature range [V].
    vref_max:
        Maximum Vref over the temperature range [V].
    vref_nom:
        Nominal Vref (at nominal temperature, typically 27 °C) [V].
    temp_min_C, temp_max_C:
        Temperature range endpoints [°C].

    Returns
    -------
    float
        TC in ppm/°C.  Returns 0.0 if vref_nom is zero or ΔT is zero.
    """
    delta_T = temp_max_C - temp_min_C
    if delta_T == 0 or vref_nom == 0:
        return 0.0
    return (vref_max - vref_min) / (vref_nom * delta_T) * 1e6


def _check_sanity(op_results: dict[str, float], specs: dict) -> dict[str, bool]:
    """Check analog sanity constraints on the DC operating point.

    Checks performed (all using node voltages from ngspice .op output):
    - **Headroom**: VCE(Q1) = V(VB) − V(VE1) > min_headroom_V
    - **Headroom**: VCE(Q2) = V(VC1) − V(VE2) > min_headroom_V
    - **PTAT swing**: V(VE2) − V(VE1) = VT·ln(N) > min_ptat_swing_mV / 1000

    Parameters
    ----------
    op_results:
        Dict of node voltages from ``_parse_op_output()``.
    specs:
        Loaded specs.yaml dict.

    Returns
    -------
    dict
        Mapping sanity check name to True (pass) / False (fail).
        A check is skipped (True) if the required nodes are not present in
        op_results, so as not to penalise non-ngspice runs.
    """
    sc = specs.get("sanity_checks", {})
    min_headroom = sc.get("min_headroom_V", 0.1)
    min_ptat_mV = sc.get("min_ptat_swing_mV", 40)

    checks: dict[str, bool] = {}

    vb = op_results.get("v(vb)")
    vc1 = op_results.get("v(vc1)")
    ve1 = op_results.get("v(ve1)")
    ve2 = op_results.get("v(ve2)")

    # Q1 collector–emitter headroom
    if vb is not None and ve1 is not None:
        checks["headroom_q1"] = (vb - ve1) >= min_headroom
    # Q2 collector–emitter headroom
    if vc1 is not None and ve2 is not None:
        checks["headroom_q2"] = (vc1 - ve2) >= min_headroom
    # PTAT voltage swing across R1
    if ve2 is not None and ve1 is not None:
        ptat_swing_V = ve2 - ve1
        checks["ptat_swing"] = ptat_swing_V >= (min_ptat_mV / 1000.0)

    return checks


def _extract_vref(op_results: dict[str, float], specs: dict) -> float | None:
    """Extract Vref from operating point results.

    Parameters
    ----------
    op_results:
        Parsed ngspice operating point dict.
    specs:
        Loaded specs.yaml dict.

    Returns
    -------
    float or None
        Vref in volts, or None if the measurement node was not found.
    """
    node = specs["vref"]["measurement_node"].lower()
    key = f"v({node})"
    return op_results.get(key)


def _check_specs(metrics: dict[str, Any], specs: dict) -> dict[str, bool]:
    """Check extracted metrics against spec targets.

    Parameters
    ----------
    metrics:
        Dict of extracted metrics (vref_V, tc_ppm_C, iq_uA, sanity_checks, etc.).
    specs:
        Loaded specs.yaml dict.

    Returns
    -------
    dict
        Mapping of spec name to True (pass) / False (fail).
        A metric that could not be extracted is marked as False (unknown = fail-safe).
    """
    checks: dict[str, bool] = {}

    # Vref check
    vref = metrics.get("vref_V")
    if vref is not None:
        target = specs["vref"]["target_V"]
        tol = specs["vref"]["tolerance_V"]
        checks["vref"] = abs(vref - target) <= tol
    else:
        checks["vref"] = False

    # TC check
    tc = metrics.get("tc_ppm_C")
    if tc is not None:
        checks["tc"] = abs(tc) <= specs["temperature_coefficient"]["max_ppm_C"]
    else:
        checks["tc"] = False

    # Quiescent current check
    iq = metrics.get("iq_uA")
    if iq is not None:
        checks["iq"] = iq <= specs["quiescent_current"]["max_uA"]
    else:
        checks["iq"] = False

    # Propagate any sanity checks from the metrics dict
    for name, passed in metrics.get("sanity_checks", {}).items():
        checks[f"sanity_{name}"] = passed

    return checks


class BandgapRunner:
    """Manages ngspice simulation runs for the Brokaw bandgap reference.

    Parameters
    ----------
    netlist_template:
        Path to the SPICE netlist template. Defaults to
        ``bandgap/netlists/bandgap_simple.sp``.
    specs_file:
        Path to the specs YAML file. Defaults to ``bandgap/specs.yaml``.
    timeout_s:
        Maximum time (seconds) to wait for ngspice before killing the process.
    """

    def __init__(
        self,
        netlist_template: Path | str = NETLIST_TEMPLATE,
        specs_file: Path | str = SPECS_FILE,
        timeout_s: float = 60.0,
    ) -> None:
        self.netlist_template = Path(netlist_template)
        self.specs_file = Path(specs_file)
        self.timeout_s = timeout_s

        with open(self.specs_file) as f:
            self.specs = yaml.safe_load(f)

        if not self.netlist_template.exists():
            raise FileNotFoundError(f"Netlist template not found: {self.netlist_template}")

    def run(self, params: dict[str, Any]) -> dict[str, Any]:
        """Run ngspice .op for a single design point and return extracted metrics.

        This is the fast path used by the Bayesian optimisation sweep — it runs
        only the DC operating point (.op) analysis, which takes milliseconds.
        For comprehensive verification (TC, sanity checks) use :meth:`run_full`.

        Parameters
        ----------
        params:
            Design variable dict, e.g.::

                {"N": 8, "R1": 20e3, "R2": 100e3, "W_P": 10e-6, "L_P": 2e-6}

        Returns
        -------
        dict
            Keys: ``vref_V``, ``iq_uA``, ``spec_checks`` (dict[str, bool]),
            ``sanity_checks`` (dict[str, bool]), ``raw_output`` (str),
            ``error`` (str or None).  ``tc_ppm_C`` is None (not run here).
        """
        result: dict[str, Any] = {
            "params": params,
            "vref_V": None,
            "tc_ppm_C": None,
            "iq_uA": None,
            "sanity_checks": {},
            "spec_checks": {},
            "raw_output": "",
            "error": None,
        }

        netlist_text = _render_netlist(self.netlist_template, params)

        with tempfile.TemporaryDirectory() as tmpdir:
            netlist_path = Path(tmpdir) / "run.sp"
            netlist_path.write_text(netlist_text)

            try:
                ngspice_bin = _find_ngspice()
            except FileNotFoundError as exc:
                result["error"] = str(exc)
                return result

            try:
                proc = subprocess.run(
                    [ngspice_bin, "-b", str(netlist_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    cwd=tmpdir,
                )
                output = proc.stdout + proc.stderr
                result["raw_output"] = output

                if proc.returncode != 0:
                    result["error"] = f"ngspice exited with code {proc.returncode}"
                    logger.warning("ngspice error: %s", output[-500:])
                else:
                    op_results = _parse_op_output(output)
                    vref = _extract_vref(op_results, self.specs)
                    result["vref_V"] = vref

                    # Quiescent current from VDD source
                    iq_raw = op_results.get("i(vdd)")
                    if iq_raw is not None:
                        result["iq_uA"] = abs(iq_raw) * 1e6

                    # Analog sanity checks from node voltages
                    result["sanity_checks"] = _check_sanity(op_results, self.specs)
                    result["spec_checks"] = _check_specs(result, self.specs)

            except subprocess.TimeoutExpired:
                result["error"] = f"ngspice timed out after {self.timeout_s}s"

        return result

    def run_full(self, params: dict[str, Any]) -> dict[str, Any]:
        """Run full multi-analysis verification: DC op + temperature sweep.

        Extends :meth:`run` with a temperature sweep that extracts the
        temperature coefficient (TC).  Runs two separate ngspice invocations:
        one for the fast .op (same as ``run()``) and one for the DC temperature
        sweep used to compute TC via the box method.

        This is the correct path for verifying spec-passing candidate designs
        before reporting them as final results.

        Parameters
        ----------
        params:
            Design variable dict (same as :meth:`run`).

        Returns
        -------
        dict
            Same keys as :meth:`run`, plus ``tc_ppm_C`` populated from the
            temperature sweep.
        """
        # Start with the fast DC op
        result = self.run(params)
        if result["error"] is not None:
            return result

        # Temperature sweep for TC
        tc, tc_error = self._run_temp_sweep(params)
        if tc_error:
            logger.warning("Temperature sweep failed: %s", tc_error)
        else:
            result["tc_ppm_C"] = tc

        # Refresh spec checks now that TC is populated
        result["spec_checks"] = _check_specs(result, self.specs)
        return result

    def _run_temp_sweep(self, params: dict[str, Any]) -> tuple[float | None, str | None]:
        """Run a DC temperature sweep and return (tc_ppm_C, error_str).

        Injects the following analysis lines into the netlist (before `.end`):

        .. code-block:: spice

            .dc temp -40 125 5
            .meas dc vref_tnom find v(VOUT) WHEN temp=27
            .meas dc vref_max  MAX  v(VOUT)
            .meas dc vref_min  MIN  v(VOUT)

        Parameters
        ----------
        params:
            Design variable dict.

        Returns
        -------
        tuple of (tc_ppm_C or None, error_str or None)
        """
        tc_spec = self.specs.get("temperature_coefficient", {})
        temp_range = tc_spec.get("temp_range_C", [-40, 125])
        temp_min, temp_max = float(temp_range[0]), float(temp_range[1])
        temp_nom = float(self.specs.get("temperature", {}).get("nom_C", 27))
        vref_node = self.specs["vref"]["measurement_node"].lower()

        netlist_text = _render_netlist(self.netlist_template, params)

        # Inject temperature sweep before .end
        sweep_lines = (
            f"\n* --- Temperature sweep (injected by run_full) ---\n"
            f".dc temp {temp_min} {temp_max} 5\n"
            f".meas dc vref_tnom find v({vref_node}) WHEN temp={temp_nom}\n"
            f".meas dc vref_max  MAX  v({vref_node})\n"
            f".meas dc vref_min  MIN  v({vref_node})\n"
        )
        netlist_text = re.sub(r"(?i)^\.end\s*$", sweep_lines + ".end", netlist_text,
                              flags=re.MULTILINE)

        with tempfile.TemporaryDirectory() as tmpdir:
            netlist_path = Path(tmpdir) / "run_tc.sp"
            netlist_path.write_text(netlist_text)

            try:
                ngspice_bin = _find_ngspice()
            except FileNotFoundError as exc:
                return None, str(exc)

            try:
                proc = subprocess.run(
                    [ngspice_bin, "-b", str(netlist_path)],
                    capture_output=True,
                    text=True,
                    # Temperature sweep runs many .op solves; allow 5× the single-point
                    # timeout (165 temperature steps × fast LEVEL-1 model overhead).
                    timeout=self.timeout_s * 5,
                    cwd=tmpdir,
                )
                output = proc.stdout + proc.stderr

                if proc.returncode != 0:
                    return None, f"ngspice TC sweep exited with code {proc.returncode}"

                meas = _parse_meas_output(output)
                vref_nom = meas.get("vref_tnom")
                vref_max = meas.get("vref_max")
                vref_min = meas.get("vref_min")

                if vref_nom is None or vref_max is None or vref_min is None:
                    return None, "Could not parse .meas output from temperature sweep"

                tc = _compute_tc(vref_min, vref_max, vref_nom, temp_min, temp_max)
                logger.info(
                    "TC sweep: Vref_nom=%.4f V, Vref_max=%.4f V, Vref_min=%.4f V → "
                    "TC=%.1f ppm/°C",
                    vref_nom, vref_max, vref_min, tc,
                )
                return tc, None

            except subprocess.TimeoutExpired:
                return None, "ngspice TC sweep timed out"

    def is_ngspice_available(self) -> bool:
        """Return True if ngspice binary is found on PATH."""
        try:
            _find_ngspice()
            return True
        except FileNotFoundError:
            return False
