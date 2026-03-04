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

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bandgap.runner import (  # noqa: E402
    BandgapRunner,
    _check_sanity,
    _check_specs,
    _compute_tc,
    _parse_meas_output,
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
        params = {"N": 8, "R1": 20e3, "R2": 100e3, "W_P": 10e-6, "L_P": 2e-6}
        result = runner.run(params)
        expected_keys = {"params", "vref_V", "iq_uA", "spec_checks", "raw_output", "error"}
        assert expected_keys.issubset(result.keys())

    def test_run_without_ngspice_returns_error(self):
        runner = BandgapRunner()
        with patch("bandgap.runner.shutil.which", return_value=None):
            with patch.dict("os.environ", {}, clear=True):
                result = runner.run({"N": 8, "R1": 20e3, "R2": 100e3})
        assert result["error"] is not None
        assert "ngspice" in result["error"].lower()
        assert result["vref_V"] is None

    def test_run_result_has_sanity_checks_key(self):
        """run() always returns a sanity_checks dict even without ngspice."""
        runner = BandgapRunner()
        with patch("bandgap.runner.shutil.which", return_value=None):
            with patch.dict("os.environ", {}, clear=True):
                result = runner.run({"N": 8, "R1": 20e3, "R2": 100e3})
        assert "sanity_checks" in result
        assert isinstance(result["sanity_checks"], dict)


class TestParseMeasOutput:
    """Tests for _parse_meas_output()."""

    def test_parses_named_measurement(self):
        output = "vref_tnom           =  1.20000e+00 at temp =  2.70000e+01\n"
        results = _parse_meas_output(output)
        assert pytest.approx(results["vref_tnom"], rel=1e-4) == 1.20

    def test_parses_max_and_min(self):
        output = (
            "vref_max            =  1.20300e+00 at temp =  1.25000e+02\n"
            "vref_min            =  1.19600e+00 at temp = -4.00000e+01\n"
        )
        results = _parse_meas_output(output)
        assert pytest.approx(results["vref_max"], rel=1e-4) == 1.203
        assert pytest.approx(results["vref_min"], rel=1e-4) == 1.196

    def test_empty_output_returns_empty_dict(self):
        assert _parse_meas_output("") == {}

    def test_handles_negative_measurement_value(self):
        output = "some_meas = -3.5e-02\n"
        results = _parse_meas_output(output)
        assert pytest.approx(results["some_meas"]) == -0.035


class TestComputeTC:
    """Tests for _compute_tc()."""

    def test_zero_tc_for_flat_vref(self):
        tc = _compute_tc(1.20, 1.20, 1.20, -40, 125)
        assert tc == 0.0

    def test_positive_tc(self):
        # Vref_max=1.2036, Vref_min=1.197 over 165 °C → TC = (1.2036-1.197)/(1.200*165)*1e6
        # = 0.0066/198 * 1e6 ≈ 33.3 ppm/°C
        tc = _compute_tc(1.197, 1.2036, 1.200, -40, 125)
        assert pytest.approx(tc, rel=0.01) == 33.33

    def test_zero_delta_t_returns_zero(self):
        tc = _compute_tc(1.19, 1.21, 1.20, 27, 27)
        assert tc == 0.0

    def test_zero_vref_nom_returns_zero(self):
        tc = _compute_tc(0.0, 0.1, 0.0, -40, 125)
        assert tc == 0.0


class TestCheckSanity:
    """Tests for _check_sanity()."""

    def _make_specs(self):
        return {
            "sanity_checks": {
                "min_headroom_V": 0.1,
                "min_ptat_swing_mV": 40,
            }
        }

    def test_all_pass_for_healthy_operating_point(self):
        # VCE(Q1)=0.7V, VCE(Q2)=0.65V, PTAT swing=0.054V (for N=8)
        op = {"v(vb)": 1.2, "v(vc1)": 1.15, "v(ve1)": 0.50, "v(ve2)": 0.554}
        checks = _check_sanity(op, self._make_specs())
        assert checks.get("headroom_q1") is True
        assert checks.get("headroom_q2") is True
        assert checks.get("ptat_swing") is True

    def test_headroom_q1_fail(self):
        op = {"v(vb)": 0.55, "v(vc1)": 1.15, "v(ve1)": 0.50, "v(ve2)": 0.554}
        checks = _check_sanity(op, self._make_specs())
        assert checks.get("headroom_q1") is False

    def test_ptat_swing_fail(self):
        # PTAT swing = 0.020 V < 40 mV
        op = {"v(vb)": 1.2, "v(vc1)": 1.15, "v(ve1)": 0.50, "v(ve2)": 0.520}
        checks = _check_sanity(op, self._make_specs())
        assert checks.get("ptat_swing") is False

    def test_missing_nodes_not_reported(self):
        """If node voltages are absent, no check is added (skipped, not failed)."""
        checks = _check_sanity({}, self._make_specs())
        assert "headroom_q1" not in checks
        assert "ptat_swing" not in checks

    def test_sanity_propagated_into_spec_checks(self):
        """_check_specs should forward sanity_checks items with sanity_ prefix."""
        specs = {
            "vref": {"target_V": 1.2, "tolerance_V": 0.01, "measurement_node": "VOUT"},
            "temperature_coefficient": {"max_ppm_C": 30},
            "quiescent_current": {"max_uA": 50},
            "sanity_checks": {"min_headroom_V": 0.1, "min_ptat_swing_mV": 40},
        }
        metrics = {
            "vref_V": 1.2, "tc_ppm_C": None, "iq_uA": 5.0,
            "sanity_checks": {"headroom_q1": True, "ptat_swing": False},
        }
        checks = _check_specs(metrics, specs)
        assert checks["sanity_headroom_q1"] is True
        assert checks["sanity_ptat_swing"] is False
