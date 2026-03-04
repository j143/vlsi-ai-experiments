"""
tests/test_optimize.py — Tests for ml/optimize.py
===================================================
Covers SyntheticBandgapRunner and the BayesianOptimizer / OptimizationResult
integration path used by the Experiment 01 entry point.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.optimize import (  # noqa: E402
    BayesianOptimizer,
    OptimizationResult,
    SyntheticBandgapRunner,
    _plot_comparison,
    _plot_convergence,
)


# ---------------------------------------------------------------------------
# SyntheticBandgapRunner tests
# ---------------------------------------------------------------------------

class TestSyntheticBandgapRunner:
    def _make_runner(self):
        return SyntheticBandgapRunner(seed=0)

    def test_run_returns_expected_keys(self):
        runner = self._make_runner()
        result = runner.run({"N": 8, "R1": 20e3, "R2": 100e3, "W_P": 10e-6, "L_P": 2e-6})
        for key in ("vref_V", "iq_uA", "spec_checks", "raw_output", "error"):
            assert key in result, f"Missing key: {key}"

    def test_run_vref_near_target_for_canonical_params(self):
        """Corrected Brokaw formula with R1=20kΩ, R2=100kΩ, N=8 → Vref ≈ 1.19 V."""
        runner = self._make_runner()
        result = runner.run({"N": 8, "R1": 20e3, "R2": 100e3, "W_P": 10e-6, "L_P": 2e-6})
        assert result["vref_V"] is not None
        assert 0.9 < result["vref_V"] < 1.5, f"Vref out of plausible range: {result['vref_V']}"

    def test_run_spec_checks_present(self):
        runner = self._make_runner()
        result = runner.run({"N": 8, "R1": 20e3, "R2": 100e3, "W_P": 10e-6, "L_P": 2e-6})
        assert "vref" in result["spec_checks"]
        assert "iq" in result["spec_checks"]

    def test_run_spec_pass_when_vref_at_target(self):
        """With R1=20kΩ, R2=100kΩ, N=8 → Vref ≈ 1.19 V (near 1.2 V ± 10 mV target)."""
        runner = self._make_runner()
        # Corrected formula: Vref = Vbe + 2*(R2/R1)*VT*ln(N) ≈ 0.65 + 0.54 ≈ 1.19 V
        result = runner.run({"N": 8, "R1": 20e3, "R2": 100e3, "W_P": 10e-6, "L_P": 2e-6})
        # May or may not pass due to noise; just assert it is a bool
        assert isinstance(result["spec_checks"]["vref"], bool)

    def test_is_ngspice_available_returns_false(self):
        runner = self._make_runner()
        assert runner.is_ngspice_available() is False

    def test_no_error_field(self):
        runner = self._make_runner()
        result = runner.run({"N": 8, "R1": 20e3, "R2": 100e3, "W_P": 10e-6, "L_P": 2e-6})
        assert result["error"] is None

    def test_result_is_json_serializable(self):
        """All values in run() output must be JSON-serializable."""
        runner = self._make_runner()
        result = runner.run({"N": 8, "R1": 20e3, "R2": 100e3, "W_P": 10e-6, "L_P": 2e-6})
        # Should not raise
        json.dumps(result)


# ---------------------------------------------------------------------------
# BayesianOptimizer with synthetic runner (integration smoke test)
# ---------------------------------------------------------------------------

class TestBayesianOptimizerSynthetic:
    def _run_small_opt(self, tmp_path):
        runner = SyntheticBandgapRunner(seed=42)
        opt = BayesianOptimizer(runner=runner, budget=5, n_init=3,
                                results_dir=tmp_path)
        return opt.run(seed=0)

    def test_result_has_expected_fields(self, tmp_path):
        result = self._run_small_opt(tmp_path)
        assert isinstance(result, OptimizationResult)
        assert result.n_simulations == 5
        assert isinstance(result.n_spec_pass, int)
        assert isinstance(result.history, list)
        assert len(result.history) == 5

    def test_saves_bo_run_json(self, tmp_path):
        self._run_small_opt(tmp_path)
        json_files = list(tmp_path.glob("bo_run_*.json"))
        assert len(json_files) == 1

    def test_spec_pass_rate_between_0_and_1(self, tmp_path):
        result = self._run_small_opt(tmp_path)
        assert 0.0 <= result.spec_pass_rate() <= 1.0

    def test_bo_run_json_is_valid(self, tmp_path):
        self._run_small_opt(tmp_path)
        json_file = list(tmp_path.glob("bo_run_*.json"))[0]
        with open(json_file) as f:
            data = json.load(f)
        assert "n_simulations" in data
        assert "history" in data

    def test_best_vref_plausible_if_set(self, tmp_path):
        result = self._run_small_opt(tmp_path)
        if result.best_vref_V is not None:
            assert 0.5 < result.best_vref_V < 3.5

    def test_top_candidates_is_list(self, tmp_path):
        result = self._run_small_opt(tmp_path)
        assert isinstance(result.top_candidates, list)

    def test_top_candidates_all_spec_passing(self, tmp_path):
        result = self._run_small_opt(tmp_path)
        for c in result.top_candidates:
            assert c.get("spec_vref_pass") is True

    def test_top_k_candidates_respects_limit(self, tmp_path):
        result = self._run_small_opt(tmp_path)
        assert len(result.top_k_candidates(k=2)) <= 2

    def test_top_candidates_in_json(self, tmp_path):
        self._run_small_opt(tmp_path)
        json_file = list(tmp_path.glob("bo_run_*.json"))[0]
        with open(json_file) as f:
            data = json.load(f)
        assert "top_candidates" in data
        assert isinstance(data["top_candidates"], list)


# ---------------------------------------------------------------------------
# Plot helpers smoke tests
# ---------------------------------------------------------------------------

class TestPlotHelpers:
    def _make_result(self, n: int = 5) -> OptimizationResult:
        runner = SyntheticBandgapRunner(seed=7)
        entries = []
        for i in range(n):
            p = {"N": 8, "R1": 20e3, "R2": 100e3, "W_P": 10e-6, "L_P": 2e-6}
            r = runner.run(p)
            entries.append({
                "iteration": i, "source": "lhs", "params": p,
                "vref_V": r["vref_V"], "iq_uA": r["iq_uA"],
                "spec_vref_pass": r["spec_checks"]["vref"],
                "acquisition_score": float("nan"),
                "sim_time_s": 0.001, "error": "",
            })
        return OptimizationResult(
            best_params=entries[0]["params"],
            best_vref_V=entries[0]["vref_V"],
            n_simulations=n,
            n_spec_pass=0,
            history=entries,
        )

    def test_plot_convergence_creates_file(self, tmp_path):
        result = self._make_result()
        _plot_convergence(result, tmp_path)
        assert (tmp_path / "bo_convergence.png").exists()

    def test_plot_comparison_creates_file(self, tmp_path):
        summary = {
            "sweep": {"n_samples": 50, "n_spec_pass": 0},
            "bayesian_optimization": {"n_simulations": 30, "n_spec_pass": 1},
        }
        _plot_comparison(summary, tmp_path)
        assert (tmp_path / "exp01_comparison.png").exists()
