"""
tests/test_bandgap_runner.py — Tests for bandgap/runner.py
============================================================
Tests cover:
  - Netlist template rendering (parameter substitution).
  - Output parsing from mock ngspice output.
  - Spec checking logic.
  - BandgapRunner initialization and ngspice availability check.

ngspice is NOT required — all tests mock or skip ngspice calls.
"""

import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bandgap.runner import (  # noqa: E402
    BandgapRunner,
    _check_specs,
    _parse_op_output,
    _render_netlist,
    NETLIST_TEMPLATE,
    SPECS_FILE,
)


class TestRenderNetlist:
    """Tests for _render_netlist()."""

    def test_renders_known_param(self):
        template = Path(NETLIST_TEMPLATE)
        result = _render_netlist(template, {"N": 12})
        assert ".param N" in result
        # The value should now be 12 (not the original 8)
        assert "= 12" in result or "=12" in result

    def test_renders_multiple_params(self):
        template = Path(NETLIST_TEMPLATE)
        result = _render_netlist(template, {"R1": 120000, "R2": 15000})
        assert "120000" in result
        assert "15000" in result

    def test_unknown_param_logs_warning(self, caplog):
        import logging
        template = Path(NETLIST_TEMPLATE)
        with caplog.at_level(logging.WARNING, logger="bandgap.runner"):
            _render_netlist(template, {"NONEXISTENT_PARAM": 999})
        assert "not found" in caplog.text.lower() or "NONEXISTENT_PARAM" in caplog.text

    def test_non_numeric_param_is_skipped(self, caplog):
        template = Path(NETLIST_TEMPLATE)
        original = template.read_text()
        with caplog.at_level(logging.WARNING, logger="bandgap.runner"):
            rendered = _render_netlist(template, {"N": "not-a-number"})
        assert "non-numeric" in caplog.text.lower()
        assert rendered == original

    def test_original_template_unchanged(self):
        """Rendering should not modify the template file on disk."""
        template = Path(NETLIST_TEMPLATE)
        original = template.read_text()
        _render_netlist(template, {"N": 99, "R1": 50000})
        assert template.read_text() == original


class TestParseOpOutput:
    """Tests for _parse_op_output()."""

    def test_parses_node_voltage(self):
        output = """
Operating Point:
v(ve1) =   1.19500e+00
v(vb)  =   8.20000e-01
i(vdd) =  -1.50000e-05
        """
        results = _parse_op_output(output)
        assert pytest.approx(results["v(ve1)"], rel=1e-4) == 1.195
        assert pytest.approx(results["v(vb)"], rel=1e-4) == 0.820
        assert pytest.approx(results["i(vdd)"], rel=1e-4) == -1.5e-5

    def test_empty_output_returns_empty_dict(self):
        results = _parse_op_output("")
        assert results == {}

    def test_handles_scientific_notation(self):
        output = "v(ve1) = 1.2e+00\n"
        results = _parse_op_output(output)
        assert pytest.approx(results["v(ve1)"]) == 1.2

    def test_handles_negative_values(self):
        output = "v(ve2) = -3.5e-02\n"
        results = _parse_op_output(output)
        assert pytest.approx(results["v(ve2)"]) == -0.035


class TestCheckSpecs:
    """Tests for _check_specs()."""

    def _make_specs(self):
        """Return a minimal specs dict matching bandgap/specs.yaml structure."""
        return {
            "vref": {
                "target_V": 1.200,
                "tolerance_V": 0.010,
                "measurement_node": "VE1",
            },
            "temperature_coefficient": {
                "max_ppm_C": 30,
                "temp_range_C": [-40, 125],
            },
            "quiescent_current": {"max_uA": 50},
        }

    def test_vref_pass(self):
        specs = self._make_specs()
        metrics = {"vref_V": 1.205}
        checks = _check_specs(metrics, specs)
        assert checks["vref"] is True

    def test_vref_fail_too_high(self):
        specs = self._make_specs()
        metrics = {"vref_V": 1.215}  # > 1.200 + 0.010
        checks = _check_specs(metrics, specs)
        assert checks["vref"] is False

    def test_vref_fail_too_low(self):
        specs = self._make_specs()
        metrics = {"vref_V": 1.185}  # < 1.200 - 0.010
        checks = _check_specs(metrics, specs)
        assert checks["vref"] is False

    def test_vref_missing_fails_safe(self):
        """Missing metric should be treated as fail (safe default)."""
        specs = self._make_specs()
        metrics = {}
        checks = _check_specs(metrics, specs)
        assert checks["vref"] is False

    def test_iq_pass(self):
        specs = self._make_specs()
        metrics = {"vref_V": 1.200, "iq_uA": 30}
        checks = _check_specs(metrics, specs)
        assert checks["iq"] is True

    def test_iq_fail(self):
        specs = self._make_specs()
        metrics = {"vref_V": 1.200, "iq_uA": 60}
        checks = _check_specs(metrics, specs)
        assert checks["iq"] is False

    def test_tc_pass(self):
        specs = self._make_specs()
        metrics = {"vref_V": 1.200, "tc_ppm_C": 10.0}  # well below max_ppm_C=30
        checks = _check_specs(metrics, specs)
        assert checks["tc"] is True

    def test_tc_fail(self):
        specs = self._make_specs()
        metrics = {"vref_V": 1.200, "tc_ppm_C": 35.0}  # above max_ppm_C=30
        checks = _check_specs(metrics, specs)
        assert checks["tc"] is False

    def test_known_good_design_passes_all_specs(self):
        """A design within all spec limits must be PASS for every check (issue #16)."""
        specs = self._make_specs()
        metrics = {
            "vref_V": 1.200,    # exactly at target, within ±10 mV
            "tc_ppm_C": 12.0,   # below max_ppm_C=30 (direction: minimize)
            "iq_uA": 20.0,      # below max_uA=50
        }
        checks = _check_specs(metrics, specs)
        assert checks["vref"] is True, "Vref should PASS for on-target design"
        assert checks["tc"] is True, "TC should PASS when below max (one-sided max constraint)"
        assert checks["iq"] is True, "Iq should PASS when below max"

    def test_psrr_pass_when_above_minimum(self):
        specs = self._make_specs()
        specs["psrr"] = {"min_dc_dB": 60}
        metrics = {"vref_V": 1.200, "tc_ppm_C": 10.0, "iq_uA": 20.0, "psrr_dB": -65.0}
        checks = _check_specs(metrics, specs)
        assert checks["psrr"] is True

    def test_psrr_fails_safe_when_missing(self):
        specs = self._make_specs()
        specs["psrr"] = {"min_dc_dB": 60}
        metrics = {"vref_V": 1.200, "tc_ppm_C": 10.0, "iq_uA": 20.0}
        checks = _check_specs(metrics, specs)
        assert checks["psrr"] is False


class TestBandgapRunner:
    """Tests for BandgapRunner."""

    def test_init_with_defaults(self):
        runner = BandgapRunner()
        assert runner.netlist_template.exists()
        assert runner.specs_file.exists()

    def test_init_with_custom_paths(self):
        runner = BandgapRunner(
            netlist_template=NETLIST_TEMPLATE,
            specs_file=SPECS_FILE,
        )
        assert runner.netlist_template == Path(NETLIST_TEMPLATE)

    def test_init_invalid_netlist_raises(self):
        with pytest.raises(FileNotFoundError):
            BandgapRunner(netlist_template="/nonexistent/path/file.sp")

    def test_is_ngspice_available_false_when_not_found(self):
        runner = BandgapRunner()
        # Patch shutil.which to simulate ngspice not installed
        with patch("bandgap.runner.shutil.which", return_value=None):
            with patch.dict("os.environ", {}, clear=True):
                assert runner.is_ngspice_available() is False

    @pytest.mark.requires_ngspice
    def test_run_returns_dict_with_expected_keys(self):
        """Integration test — only runs when ngspice is installed."""
        runner = BandgapRunner()
        params = {"N": 8, "R1": 100e3, "R2": 10e3, "W_P": 4e-6, "L_P": 1e-6}
        result = runner.run(params)
        expected_keys = {"params", "vref_V", "iq_uA", "spec_checks", "raw_output", "error"}
        assert expected_keys.issubset(result.keys())

    def test_run_without_ngspice_returns_error(self):
        runner = BandgapRunner()
        with patch("bandgap.runner.shutil.which", return_value=None):
            with patch.dict("os.environ", {}, clear=True):
                result = runner.run({"N": 8, "R1": 100e3, "R2": 10e3})
        assert result["error"] is not None
        assert "ngspice" in result["error"].lower()
        assert result["vref_V"] is None
