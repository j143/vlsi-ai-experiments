"""
tests/test_evaluate_loop.py — Tests for ml/evaluate_surrogate_loop.py
=======================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from ml.evaluate_surrogate_loop import (  # noqa: E402
    FEATURES,
    TARGETS,
    build_and_evaluate,
    generate_synthetic_dataset,
)


class TestGenerateSyntheticDataset:
    def test_returns_dataframe(self):
        df = generate_synthetic_dataset(n_samples=20)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self):
        df = generate_synthetic_dataset(n_samples=30)
        assert len(df) == 30

    def test_feature_columns_present(self):
        df = generate_synthetic_dataset(n_samples=10)
        for col in FEATURES:
            assert col in df.columns, f"Missing feature column: {col}"

    def test_target_columns_present(self):
        df = generate_synthetic_dataset(n_samples=10)
        for col in TARGETS:
            assert col in df.columns, f"Missing target column: {col}"

    def test_sim_time_column_present(self):
        df = generate_synthetic_dataset(n_samples=10)
        assert "sim_time_s" in df.columns

    def test_vref_in_physically_plausible_range(self):
        df = generate_synthetic_dataset(n_samples=50, seed=0)
        # Brokaw Vref is typically 0.5–1.5 V for these parameter ranges
        assert df["vref_V"].between(0.3, 2.0).all(), "vref_V values outside expected range"

    def test_iq_non_negative(self):
        df = generate_synthetic_dataset(n_samples=50, seed=1)
        assert (df["iq_uA"] > 0).all()

    def test_reproducible_with_same_seed(self):
        df1 = generate_synthetic_dataset(n_samples=20, seed=7)
        df2 = generate_synthetic_dataset(n_samples=20, seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_give_different_data(self):
        df1 = generate_synthetic_dataset(n_samples=20, seed=0)
        df2 = generate_synthetic_dataset(n_samples=20, seed=1)
        assert not df1["vref_V"].equals(df2["vref_V"])


class TestBuildAndEvaluate:
    @pytest.fixture
    def small_df(self):
        """Small synthetic dataset sufficient for surrogate training."""
        return generate_synthetic_dataset(n_samples=50, seed=42)

    def test_returns_dict(self, small_df, tmp_path):
        result = build_and_evaluate(small_df, out_dir=tmp_path)
        assert isinstance(result, dict)

    def test_speedup_key_present(self, small_df, tmp_path):
        result = build_and_evaluate(small_df, out_dir=tmp_path)
        assert "speedup" in result

    def test_speedup_greater_than_one(self, small_df, tmp_path):
        result = build_and_evaluate(small_df, out_dir=tmp_path)
        assert result["speedup"]["speedup_x"] > 1.0, (
            "Surrogate should be faster than SPICE simulation"
        )

    def test_metric_keys_for_each_target(self, small_df, tmp_path):
        result = build_and_evaluate(small_df, out_dir=tmp_path)
        for model in ("GP", "RF"):
            for target in TARGETS:
                key = f"{model}/{target}"
                assert key in result, f"Missing key: {key}"

    def test_metric_values_have_expected_fields(self, small_df, tmp_path):
        result = build_and_evaluate(small_df, out_dir=tmp_path)
        for model in ("GP", "RF"):
            for target in TARGETS:
                key = f"{model}/{target}"
                vals = result[key]
                assert "mae" in vals
                assert "rmse" in vals
                assert "r2" in vals
                assert "n_train" in vals
                assert "n_test" in vals

    def test_mae_non_negative(self, small_df, tmp_path):
        result = build_and_evaluate(small_df, out_dir=tmp_path)
        for key, vals in result.items():
            if key == "speedup":
                continue
            assert vals["mae"] >= 0.0, f"Negative MAE for {key}"

    def test_rmse_non_negative(self, small_df, tmp_path):
        result = build_and_evaluate(small_df, out_dir=tmp_path)
        for key, vals in result.items():
            if key == "speedup":
                continue
            assert vals["rmse"] >= 0.0, f"Negative RMSE for {key}"

    def test_plot_saved(self, small_df, tmp_path):
        build_and_evaluate(small_df, out_dir=tmp_path)
        assert (tmp_path / "mae_per_spec.png").exists(), "Plot file not created"

    def test_n_train_plus_n_test_equals_total(self, small_df, tmp_path):
        n = len(small_df)
        result = build_and_evaluate(small_df, out_dir=tmp_path, test_frac=0.2)
        key = "GP/vref_V"
        assert result[key]["n_train"] + result[key]["n_test"] == n

    def test_gp_r2_positive_on_synthetic(self, tmp_path):
        """GP should capture most variance of the Brokaw physics proxy."""
        df = generate_synthetic_dataset(n_samples=100, seed=0)
        result = build_and_evaluate(df, out_dir=tmp_path, seed=0)
        r2 = result["GP/vref_V"]["r2"]
        assert r2 > 0.5, f"GP R² too low on synthetic data: {r2:.4f}"

    def test_spice_calls_saved_pct_in_range(self, small_df, tmp_path):
        result = build_and_evaluate(small_df, out_dir=tmp_path)
        pct = result["speedup"]["spice_calls_saved_pct"]
        assert 0.0 <= pct <= 100.0
