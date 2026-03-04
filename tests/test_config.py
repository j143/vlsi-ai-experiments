"""
tests/test_config.py — Validates config/tech_placeholder.yaml and bandgap/specs.yaml
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO_ROOT = Path(__file__).parent.parent
TECH_CONFIG = REPO_ROOT / "config" / "tech_placeholder.yaml"
SPECS_FILE = REPO_ROOT / "bandgap" / "specs.yaml"


class TestTechPlaceholder:
    def _load(self):
        with open(TECH_CONFIG) as f:
            return yaml.safe_load(f)

    def test_file_is_valid_yaml(self):
        cfg = self._load()
        assert isinstance(cfg, dict)

    def test_required_top_level_keys(self):
        cfg = self._load()
        for key in ["pdk", "supply", "temperature", "mosfet", "bjt", "bandgap_specs"]:
            assert key in cfg, f"Missing key: {key}"

    def test_supply_voltages_are_numeric(self):
        cfg = self._load()
        supply = cfg["supply"]
        assert isinstance(supply["vdd_nom_V"], (int, float))
        assert isinstance(supply["vdd_min_V"], (int, float))
        assert isinstance(supply["vdd_max_V"], (int, float))

    def test_supply_voltage_ordering(self):
        cfg = self._load()
        supply = cfg["supply"]
        assert supply["vdd_min_V"] < supply["vdd_nom_V"] < supply["vdd_max_V"], (
            "Expect vdd_min < vdd_nom < vdd_max"
        )

    def test_temperature_corners_are_list(self):
        cfg = self._load()
        corners = cfg["temperature"]["corners"]
        assert isinstance(corners, list)
        assert len(corners) >= 2

    def test_bjt_ratio_is_positive(self):
        cfg = self._load()
        ratio = cfg["bjt"]["npn"]["ratio_N"]
        assert ratio > 1, "BJT ratio N must be > 1 for PTAT operation"

    def test_bandgap_spec_vref_tolerance_is_positive(self):
        cfg = self._load()
        assert cfg["bandgap_specs"]["vref_tolerance_V"] > 0

    def test_bandgap_spec_tc_is_positive(self):
        cfg = self._load()
        assert cfg["bandgap_specs"]["tc_max_ppm_C"] > 0

    def test_bandgap_spec_psrr_is_positive(self):
        cfg = self._load()
        assert cfg["bandgap_specs"]["psrr_min_dB"] > 0


class TestBandgapSpecs:
    def _load(self):
        with open(SPECS_FILE) as f:
            return yaml.safe_load(f)

    def test_file_is_valid_yaml(self):
        specs = self._load()
        assert isinstance(specs, dict)

    def test_required_keys(self):
        specs = self._load()
        for key in ["vref", "temperature_coefficient", "psrr", "quiescent_current"]:
            assert key in specs, f"Missing spec key: {key}"

    def test_vref_target_is_numeric(self):
        specs = self._load()
        assert isinstance(specs["vref"]["target_V"], (int, float))

    def test_vref_tolerance_is_positive(self):
        specs = self._load()
        assert specs["vref"]["tolerance_V"] > 0

    def test_tc_max_is_positive(self):
        specs = self._load()
        assert specs["temperature_coefficient"]["max_ppm_C"] > 0

    def test_temp_range_has_two_elements(self):
        specs = self._load()
        temp_range = specs["temperature_coefficient"]["temp_range_C"]
        assert len(temp_range) == 2
        assert temp_range[0] < temp_range[1]

    def test_psrr_min_is_positive(self):
        specs = self._load()
        assert specs["psrr"]["min_dc_dB"] > 0

    def test_iq_max_is_positive(self):
        specs = self._load()
        assert specs["quiescent_current"]["max_uA"] > 0

    def test_sanity_checks_present(self):
        specs = self._load()
        assert "sanity_checks" in specs
        sc = specs["sanity_checks"]
        assert sc["min_headroom_V"] > 0
        assert sc["min_phase_margin_deg"] >= 0
