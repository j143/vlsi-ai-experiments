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
    generate_bad_patch,
    generate_dataset,
    generate_labeled_dataset,
    generate_synthetic_patch,
    mask_patches,
)
from layout.evaluate import (  # noqa: E402
    confusion_matrix_report,
    evaluate_patches,
    iou_per_layer,
    pixel_accuracy_per_layer,
    run_drc,
)
from layout.patch_model import BadPatchClassifier, UNetPatchModel  # noqa: E402


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


class TestGenerateBadPatch:
    def test_output_shape(self):
        cfg = PatchConfig(patch_size=32)
        rng = np.random.default_rng(0)
        patch = generate_bad_patch(cfg, rng)
        assert patch.shape == (N_LAYERS, 32, 32)

    def test_output_dtype(self):
        cfg = PatchConfig(patch_size=32)
        rng = np.random.default_rng(0)
        patch = generate_bad_patch(cfg, rng)
        assert patch.dtype == np.uint8

    def test_binary_values(self):
        cfg = PatchConfig(patch_size=32)
        rng = np.random.default_rng(0)
        for defect in ["empty", "noise", "missing_contacts", "short_circuit", "min_width"]:
            patch = generate_bad_patch(cfg, rng, defect_type=defect)
            assert set(np.unique(patch)).issubset({0, 1}), f"defect={defect}"

    def test_all_defect_types_produce_valid_shape(self):
        cfg = PatchConfig(patch_size=32)
        rng = np.random.default_rng(7)
        for defect in ["empty", "noise", "missing_contacts", "short_circuit", "min_width"]:
            patch = generate_bad_patch(cfg, rng, defect_type=defect)
            assert patch.shape == (N_LAYERS, 32, 32), f"defect={defect}"

    def test_random_defect_type(self):
        cfg = PatchConfig(patch_size=32)
        rng = np.random.default_rng(99)
        patch = generate_bad_patch(cfg, rng)  # defect_type=None → random
        assert patch.shape == (N_LAYERS, 32, 32)


class TestGenerateLabeledDataset:
    def test_shapes(self):
        patches, labels = generate_labeled_dataset(n_ok=10, n_bad=10, seed=0)
        assert patches.shape == (20, N_LAYERS, 32, 32)
        assert labels.shape == (20,)

    def test_label_counts(self):
        patches, labels = generate_labeled_dataset(n_ok=30, n_bad=20, seed=0)
        assert (labels == 0).sum() == 30
        assert (labels == 1).sum() == 20

    def test_label_dtype(self):
        _, labels = generate_labeled_dataset(n_ok=5, n_bad=5, seed=0)
        assert labels.dtype == np.int64

    def test_patches_dtype(self):
        patches, _ = generate_labeled_dataset(n_ok=5, n_bad=5, seed=0)
        assert patches.dtype == np.uint8

    def test_is_shuffled(self):
        """Labels should NOT all be 0s followed by 1s after shuffling."""
        _, labels = generate_labeled_dataset(n_ok=50, n_bad=50, seed=0)
        # Sorted labels != labels means shuffle happened.
        assert not np.array_equal(labels, np.sort(labels))


class TestBadPatchClassifier:
    def test_fit_predict_shapes(self):
        patches, labels = generate_labeled_dataset(n_ok=40, n_bad=40, seed=1)
        clf = BadPatchClassifier()
        clf.fit(patches[:60], labels[:60])
        preds = clf.predict(patches[60:])
        assert preds.shape == (20,)

    def test_predict_before_fit_raises(self):
        clf = BadPatchClassifier()
        patches = np.zeros((5, N_LAYERS, 32, 32), dtype=np.uint8)
        with pytest.raises(RuntimeError):
            clf.predict(patches)

    def test_predict_proba_shape(self):
        patches, labels = generate_labeled_dataset(n_ok=40, n_bad=40, seed=2)
        clf = BadPatchClassifier()
        clf.fit(patches[:60], labels[:60])
        proba = clf.predict_proba(patches[60:])
        assert proba.shape == (20, 2)

    def test_predictions_are_binary(self):
        patches, labels = generate_labeled_dataset(n_ok=40, n_bad=40, seed=3)
        clf = BadPatchClassifier()
        clf.fit(patches[:60], labels[:60])
        preds = clf.predict(patches[60:])
        assert set(np.unique(preds)).issubset({0, 1})

    def test_extract_features_shape(self):
        patches = np.random.randint(0, 2, (10, N_LAYERS, 32, 32), dtype=np.uint8)
        features = BadPatchClassifier.extract_features(patches)
        assert features.shape[0] == 10
        assert features.dtype == np.float32

    def test_lr_classifier(self):
        patches, labels = generate_labeled_dataset(n_ok=40, n_bad=40, seed=4)
        clf = BadPatchClassifier(classifier_type="lr")
        clf.fit(patches[:60], labels[:60])
        preds = clf.predict(patches[60:])
        assert preds.shape == (20,)

    def test_unknown_classifier_raises(self):
        with pytest.raises(ValueError):
            BadPatchClassifier(classifier_type="xgb")

    def test_reasonable_accuracy(self):
        """Classifier should do significantly better than random (>60%) on this dataset."""
        patches, labels = generate_labeled_dataset(n_ok=100, n_bad=100, seed=42)
        n_train = int(0.8 * len(patches))
        clf = BadPatchClassifier()
        clf.fit(patches[:n_train], labels[:n_train])
        preds = clf.predict(patches[n_train:])
        accuracy = (preds == labels[n_train:]).mean()
        assert accuracy > 0.60, f"Accuracy too low: {accuracy:.3f}"


class TestConfusionMatrixReport:
    def test_returns_expected_keys(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        result = confusion_matrix_report(y_true, y_pred)
        for key in ("confusion_matrix", "accuracy", "precision", "recall",
                    "f1_score", "class_names", "report"):
            assert key in result

    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 1])
        result = confusion_matrix_report(y, y)
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["f1_score"] == pytest.approx(1.0)

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1])
        result = confusion_matrix_report(y_true, y_pred)
        cm = result["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2

    def test_custom_class_names(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        result = confusion_matrix_report(y_true, y_pred, class_names=["good", "defect"])
        assert result["class_names"] == ["good", "defect"]

    def test_scores_in_0_1(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 50)
        y_pred = rng.integers(0, 2, 50)
        result = confusion_matrix_report(y_true, y_pred)
        for key in ("accuracy", "precision", "recall", "f1_score"):
            assert 0.0 <= result[key] <= 1.0, key
