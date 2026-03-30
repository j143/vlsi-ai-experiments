"""
bandgap/runner.py — ngspice Runner for Bandgap Reference
=========================================================
Provides functions to:
  - Render a parameterized SPICE netlist with a specific design point.
  - Run ngspice and capture output.
  - Parse ngspice output to extract key metrics (Vref, TC, PSRR, etc.).
  - Check results against specs defined in bandgap/specs.yaml.

Usage example::

    from bandgap.runner import BandgapRunner
    runner = BandgapRunner()
    result = runner.run({"N": 8, "R1": 100e3, "R2": 10e3, "W_P": 4e-6, "L_P": 1e-6})
    print(result)  # dict with vref_V, tc_ppm_C, psrr_dB, spec_pass, ...

Requires ngspice to be installed and on PATH.
Set environment variable NGSPICE_BIN to override the ngspice binary path.
"""

import logging
import math
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
        if not isinstance(name, str):
            logger.warning("Skipping non-string parameter name: %r", name)
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            logger.warning("Skipping non-numeric parameter '%s': %r", name, value)
            continue
        if not math.isfinite(numeric_value):
            logger.warning("Skipping non-finite parameter '%s': %r", name, value)
            continue
        # Match lines like: .param N    = 8
        # Replace value after '=' with the new value.
        pattern = rf"(\.param\s+{re.escape(name)}\s*=\s*)[^\s$]+"
        replacement = rf"\g<1>{numeric_value:g}"
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
    pattern = re.compile(r"(v\([^)]+\)|i\([^)]+\))\s*=\s*([-+eE\d.]+)", re.IGNORECASE)
    for match in pattern.finditer(ngspice_output):
        var_name = match.group(1).lower()
        try:
            value = float(match.group(2))
            if math.isfinite(value):
                results[var_name] = value
        except ValueError:
            pass
    return results


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
    node = specs.get("vref", {}).get("measurement_node")
    if not isinstance(node, str) or not node.strip():
        logger.warning("Missing or invalid vref.measurement_node in specs; cannot extract Vref.")
        return None
    node = node.lower().strip()
    key = f"v({node})"
    return op_results.get(key)


def _check_specs(metrics: dict[str, Any], specs: dict) -> dict[str, bool]:
    """Check extracted metrics against spec targets.

    Parameters
    ----------
    metrics:
        Dict of extracted metrics (vref_V, tc_ppm_C, etc.).
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
        target = float(specs.get("vref", {}).get("target_V", 1.2))
        tol = max(float(specs.get("vref", {}).get("tolerance_V", 0.0)), 0.0)
        checks["vref"] = abs(vref - target) <= tol
    else:
        checks["vref"] = False

    # TC check
    tc = metrics.get("tc_ppm_C")
    if tc is not None:
        tc_max = max(float(specs.get("temperature_coefficient", {}).get("max_ppm_C", 0.0)), 0.0)
        checks["tc"] = abs(tc) <= tc_max
    else:
        checks["tc"] = False

    # Quiescent current check
    iq = metrics.get("iq_uA")
    if iq is not None:
        iq_max = max(float(specs.get("quiescent_current", {}).get("max_uA", 0.0)), 0.0)
        checks["iq"] = iq <= iq_max
    else:
        checks["iq"] = False

    # PSRR check (optional in result, but enforced if present in specs)
    psrr = metrics.get("psrr_dB")
    if psrr is not None:
        min_psrr = max(float(specs.get("psrr", {}).get("min_dc_dB", 0.0)), 0.0)
        checks["psrr"] = abs(psrr) >= min_psrr
    else:
        checks["psrr"] = False

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

        with open(self.specs_file, encoding="utf-8") as f:
            self.specs = yaml.safe_load(f)

        if not self.netlist_template.exists():
            raise FileNotFoundError(f"Netlist template not found: {self.netlist_template}")

    def run(self, params: dict[str, Any]) -> dict[str, Any]:
        """Run ngspice for a single design point and return extracted metrics.

        Parameters
        ----------
        params:
            Design variable dict, e.g.::

                {"N": 8, "R1": 100e3, "R2": 10e3, "W_P": 4e-6, "L_P": 1e-6}

        Returns
        -------
        dict
            Keys include: ``vref_V``, ``tc_ppm_C`` (if temperature sweep run),
            ``iq_uA``, ``spec_checks`` (dict[str, bool]), ``raw_output`` (str),
            ``error`` (str or None).
        """
        result: dict[str, Any] = {
            "params": params,
            "vref_V": None,
            "tc_ppm_C": None,
            "iq_uA": None,
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

                    # Estimate quiescent current from VDD source current (i(vdd))
                    iq_raw = op_results.get("i(vdd)")
                    if iq_raw is not None:
                        result["iq_uA"] = abs(iq_raw) * 1e6

                    result["spec_checks"] = _check_specs(result, self.specs)

            except subprocess.TimeoutExpired:
                result["error"] = f"ngspice timed out after {self.timeout_s}s"
            except (OSError, ValueError) as exc:
                result["error"] = f"ngspice invocation failed: {exc}"

        return result

    def is_ngspice_available(self) -> bool:
        """Return True if ngspice binary is found on PATH."""
        try:
            _find_ngspice()
            return True
        except FileNotFoundError:
            return False
