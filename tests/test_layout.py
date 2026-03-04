"""
tests/test_layout.py — Tests for layout/ module
=================================================
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from layout.data_stub import (  # noqa: E402
    N_LAYERS,
    PatchConfig,
    generate_dataset,
    generate_synthetic_patch,
    mask_patches,
)
from layout.evaluate import (  # noqa: E402
    evaluate_patches,
    iou_per_layer,
    pixel_accuracy_per_layer,
    run_drc,
)
from layout.patch_model import UNetPatchModel  # noqa: E402


class TestGenerateSyntheticPatch:
    def test_output_shape(self):
        cfg = PatchConfig(patch_size=32)
        rng = np.random.default_rng(0)
        patch = generate_synthetic_patch(cfg, rng)
        assert patch.shape == (N_LAYERS, 32, 32)

    def test_output_dtype(self):
        cfg = PatchConfig(patch_size=32)
        rng = np.random.default_rng(0)
        patch = generate_synthetic_patch(cfg, rng)
        assert patch.dtype == np.uint8

    def test_binary_values(self):
        cfg = PatchConfig(patch_size=32)
        rng = np.random.default_rng(0)
        patch = generate_synthetic_patch(cfg, rng)
        assert set(np.unique(patch)).issubset({0, 1})

    def test_has_nonzero_pixels(self):
        """At least some layers should have drawn shapes."""
        cfg = PatchConfig(patch_size=32)
        rng = np.random.default_rng(0)
        patch = generate_synthetic_patch(cfg, rng)
        assert patch.sum() > 0

    def test_different_seeds_differ(self):
        cfg = PatchConfig(patch_size=32)
        p1 = generate_synthetic_patch(cfg, np.random.default_rng(0))
        p2 = generate_synthetic_patch(cfg, np.random.default_rng(99))
        assert not np.array_equal(p1, p2)


class TestGenerateDataset:
    def test_shape(self):
        patches = generate_dataset(n_patches=10, patch_size=16, seed=0)
        assert patches.shape == (10, N_LAYERS, 16, 16)

    def test_dtype(self):
        patches = generate_dataset(n_patches=5, patch_size=16, seed=0)
        assert patches.dtype == np.uint8

    def test_saves_file(self, tmp_path):
        out = tmp_path / "test_patches.npy"
        patches = generate_dataset(n_patches=5, patch_size=16, seed=0, out_path=out)
        assert out.exists()
        loaded = np.load(out)
        np.testing.assert_array_equal(patches, loaded)


class TestMaskPatches:
    def test_masked_layer_is_zero(self):
        patches = generate_dataset(n_patches=20, patch_size=16, seed=0)
        masked, mask_idx, orig = mask_patches(patches, layer=2, mask_fraction=0.5)
        assert (masked[mask_idx, 2] == 0).all()

    def test_other_layers_unchanged(self):
        patches = generate_dataset(n_patches=20, patch_size=16, seed=0)
        masked, mask_idx, _ = mask_patches(patches, layer=2, mask_fraction=0.5)
        for layer in range(N_LAYERS):
            if layer == 2:
                continue
            np.testing.assert_array_equal(
                masked[:, layer], patches[:, layer]
            )

    def test_original_values_preserved(self):
        patches = generate_dataset(n_patches=20, patch_size=16, seed=0)
        _, mask_idx, orig = mask_patches(patches, layer=2, mask_fraction=0.5)
        np.testing.assert_array_equal(orig, patches[mask_idx, 2])


class TestIoUPerLayer:
    def test_perfect_match(self):
        arr = np.random.randint(0, 2, (5, N_LAYERS, 16, 16)).astype(np.uint8)
        iou = iou_per_layer(arr, arr)
        np.testing.assert_allclose(iou, np.ones(N_LAYERS))

    def test_zero_overlap(self):
        pred = np.ones((3, N_LAYERS, 8, 8), dtype=np.uint8)
        ref = np.zeros((3, N_LAYERS, 8, 8), dtype=np.uint8)
        iou = iou_per_layer(pred, ref)
        assert (iou == 0).all()

    def test_both_empty_returns_one(self):
        """IoU of two empty layers should be defined as 1 (perfect empty match)."""
        pred = np.zeros((3, 2, 8, 8), dtype=np.uint8)
        ref = np.zeros((3, 2, 8, 8), dtype=np.uint8)
        iou = iou_per_layer(pred, ref)
        assert (iou == 1.0).all()

    def test_shape(self):
        pred = np.random.randint(0, 2, (10, N_LAYERS, 16, 16)).astype(np.uint8)
        ref = pred.copy()
        iou = iou_per_layer(pred, ref)
        assert iou.shape == (N_LAYERS,)

    def test_values_in_0_1(self):
        pred = np.random.randint(0, 2, (5, N_LAYERS, 16, 16)).astype(np.uint8)
        ref = np.random.randint(0, 2, (5, N_LAYERS, 16, 16)).astype(np.uint8)
        iou = iou_per_layer(pred, ref)
        assert (iou >= 0).all() and (iou <= 1).all()


class TestPixelAccuracyPerLayer:
    def test_perfect_match(self):
        arr = np.random.randint(0, 2, (5, N_LAYERS, 16, 16)).astype(np.uint8)
        acc = pixel_accuracy_per_layer(arr, arr)
        np.testing.assert_allclose(acc, np.ones(N_LAYERS))

    def test_values_in_0_1(self):
        pred = np.random.randint(0, 2, (5, N_LAYERS, 16, 16)).astype(np.uint8)
        ref = np.random.randint(0, 2, (5, N_LAYERS, 16, 16)).astype(np.uint8)
        acc = pixel_accuracy_per_layer(pred, ref)
        assert (acc >= 0).all() and (acc <= 1).all()


class TestRunDRC:
    def test_returns_expected_keys(self):
        patches = generate_dataset(n_patches=5, patch_size=32, seed=0)
        result = run_drc(patches)
        assert "n_violations" in result
        assert "pass_rate" in result
        assert "note" in result

    def test_clean_empty_patches_pass(self):
        """All-zero patches should have no DRC violations."""
        patches = np.zeros((5, N_LAYERS, 32, 32), dtype=np.uint8)
        result = run_drc(patches)
        assert result["n_violations"] == 0
        assert result["pass_rate"] == pytest.approx(1.0)

    def test_pass_rate_between_0_and_1(self):
        patches = generate_dataset(n_patches=10, patch_size=32, seed=0)
        result = run_drc(patches)
        assert 0.0 <= result["pass_rate"] <= 1.0


class TestEvaluatePatches:
    def test_returns_expected_keys(self):
        pred = generate_dataset(n_patches=5, patch_size=16, seed=0)
        ref = generate_dataset(n_patches=5, patch_size=16, seed=1)
        metrics = evaluate_patches(pred.astype(np.float32), ref.astype(np.float32))
        expected = {"iou_per_layer", "pixel_acc_per_layer", "mean_iou", "mean_pixel_acc"}
        assert expected.issubset(metrics.keys())

    def test_shape_mismatch_raises(self):
        pred = np.zeros((5, N_LAYERS, 16, 16), dtype=np.float32)
        ref = np.zeros((5, N_LAYERS, 32, 32), dtype=np.float32)
        with pytest.raises(ValueError):
            evaluate_patches(pred, ref)

    def test_identical_patches_max_metrics(self):
        arr = generate_dataset(n_patches=5, patch_size=16, seed=0).astype(np.float32)
        metrics = evaluate_patches(arr, arr, run_drc_check=False)
        assert metrics["mean_iou"] == pytest.approx(1.0)
        assert metrics["mean_pixel_acc"] == pytest.approx(1.0)


class TestUNetPatchModel:
    def test_init(self):
        model = UNetPatchModel(n_layers=N_LAYERS, patch_size=32)
        assert model.n_layers == N_LAYERS

    def test_predict_output_shape(self):
        model = UNetPatchModel(n_layers=N_LAYERS, patch_size=32)
        x = np.random.rand(4, N_LAYERS, 32, 32).astype(np.float32)
        pred = model.predict(x)
        assert pred.shape == (4, N_LAYERS, 32, 32)

    def test_predict_wrong_channels_raises(self):
        model = UNetPatchModel(n_layers=N_LAYERS, patch_size=32)
        x = np.random.rand(2, N_LAYERS + 1, 32, 32).astype(np.float32)
        with pytest.raises(ValueError, match="channels"):
            model.predict(x)

    def test_predict_wrong_ndim_raises(self):
        model = UNetPatchModel(n_layers=N_LAYERS, patch_size=32)
        x = np.random.rand(32, 32).astype(np.float32)
        with pytest.raises(ValueError, match="4D"):
            model.predict(x)
