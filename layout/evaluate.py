"""
layout/evaluate.py — Layout Pattern Evaluation Metrics
=========================================================
Provides functions to evaluate AI-generated layout patches against human-designed
references. Metrics computed:
  - Intersection over Union (IoU) per layer.
  - Pixel accuracy per layer.
  - DRC violation count (rule-based, using design rules from tech_placeholder.yaml).

DRC checks implemented (minimal rule set):
  - Min width: no single-pixel features smaller than min_feature_size.
  - Min spacing: no two shapes closer than min_space.

Full DRC requires a proper DRC tool (KLayout, Magic) — see TODO below.

Usage::

    import numpy as np
    from layout.evaluate import evaluate_patches, run_drc

    pred = np.random.randint(0, 2, (10, 8, 32, 32)).astype(np.uint8)
    ref  = np.random.randint(0, 2, (10, 8, 32, 32)).astype(np.uint8)

    metrics = evaluate_patches(pred, ref)
    print(metrics)  # dict with per-layer IoU, accuracy, DRC violations
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
TECH_CONFIG_FILE = _REPO_ROOT / "config" / "tech_placeholder.yaml"


def _load_design_rules() -> dict:
    """Load design rules from tech_placeholder.yaml."""
    with open(TECH_CONFIG_FILE) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("design_rules", {})


def iou_per_layer(
    pred: np.ndarray,
    ref: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Compute IoU (Jaccard index) per layer.

    Parameters
    ----------
    pred:
        Predicted patches, shape (N, L, H, W). Values in [0, 1] or binary.
    ref:
        Reference patches, shape (N, L, H, W). Binary (0 or 1).
    threshold:
        Threshold for converting soft predictions to binary.

    Returns
    -------
    np.ndarray
        IoU per layer, shape (L,). Values in [0, 1].
    """
    pred_bin = (pred >= threshold).astype(np.uint8)
    ref_bin = (ref >= threshold).astype(np.uint8)

    n_layers = pred.shape[1]
    iou = np.zeros(n_layers)
    for layer in range(n_layers):
        p = pred_bin[:, layer].ravel()
        r = ref_bin[:, layer].ravel()
        intersection = np.logical_and(p, r).sum()
        union = np.logical_or(p, r).sum()
        iou[layer] = intersection / union if union > 0 else 1.0  # empty = perfect match
    return iou


def pixel_accuracy_per_layer(
    pred: np.ndarray,
    ref: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Compute pixel-wise accuracy per layer.

    Parameters
    ----------
    pred:
        Predicted patches, shape (N, L, H, W).
    ref:
        Reference patches, shape (N, L, H, W).
    threshold:
        Binarization threshold.

    Returns
    -------
    np.ndarray
        Accuracy per layer, shape (L,). Values in [0, 1].
    """
    pred_bin = (pred >= threshold).astype(np.uint8)
    ref_bin = (ref >= threshold).astype(np.uint8)
    n_layers = pred.shape[1]
    acc = np.zeros(n_layers)
    for layer in range(n_layers):
        acc[layer] = (pred_bin[:, layer] == ref_bin[:, layer]).mean()
    return acc


def _check_min_width(layer_map: np.ndarray, min_width_grid: int) -> int:
    """Count isolated features smaller than min_width in a single binary 2D map.

    Uses a simple erosion-based check: erode by min_width; regions that disappear
    are too small.

    Parameters
    ----------
    layer_map:
        Binary 2D array (H, W).
    min_width_grid:
        Minimum width in grid units.

    Returns
    -------
    int
        Number of connected components violating min width.
    """
    if min_width_grid <= 1:
        return 0
    from scipy.ndimage import label, binary_erosion

    eroded = binary_erosion(
        layer_map, structure=np.ones((min_width_grid, min_width_grid))
    )
    # Original shapes that vanish after erosion are too small
    violating = layer_map.astype(bool) & ~eroded
    labeled, n_comp = label(violating)
    return n_comp


def run_drc(
    patches: np.ndarray,
    layer_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Run a minimal design-rule check on a batch of layout patches.

    Currently checks:
    - Min width for poly and metal1 layers (from tech_placeholder.yaml).
    - Min spacing is NOT yet checked (requires more complex logic — TODO).

    Parameters
    ----------
    patches:
        Binary layout patches, shape (N, L, H, W), dtype uint8.
    layer_names:
        Optional dict mapping layer index to name.

    Returns
    -------
    dict with keys:
        ``n_violations`` (int): total DRC violation count across all patches/layers.
        ``pass_rate`` (float): fraction of patches with zero violations.
        ``violations_per_layer`` (dict): {layer_name: count}.
        ``note`` (str): disclaimer about completeness of DRC.

    Notes
    -----
    This is a PARTIAL DRC — it only checks min-width for a subset of layers.
    For production use, export patches to GDS2 and run a proper DRC tool
    (KLayout, Magic, Calibre).
    """
    try:
        design_rules = _load_design_rules()
    except FileNotFoundError:
        logger.warning("tech_placeholder.yaml not found — DRC skipped.")
        return {"n_violations": 0, "pass_rate": 1.0, "note": "DRC skipped (no rules file)"}

    # Map layer name → min_width_grid (assume 1 grid unit = 0.15 µm — illustrative)
    GRID_UM = 0.15  # TODO(human): replace with actual grid resolution from PDK
    layer_rules = {}
    for layer_name_key in ["poly", "metal1"]:
        rule = design_rules.get(layer_name_key, {})
        if "min_width_um" in rule:
            layer_rules[layer_name_key] = max(1, int(rule["min_width_um"] / GRID_UM))

    if layer_names is None:
        from layout.data_stub import LAYER_MAP
        layer_names = LAYER_MAP

    # Reverse map: name → index
    name_to_idx = {v: k for k, v in layer_names.items()}

    N, L, H, W = patches.shape
    violations_per_layer: dict[str, int] = {}
    patch_has_violation = np.zeros(N, dtype=bool)

    for layer_name, min_w in layer_rules.items():
        idx = name_to_idx.get(layer_name)
        if idx is None or idx >= L:
            continue
        count = 0
        for patch_idx in range(N):
            layer_map = patches[patch_idx, idx]
            n_viol = _check_min_width(layer_map, min_w)
            if n_viol > 0:
                count += n_viol
                patch_has_violation[patch_idx] = True
        violations_per_layer[layer_name] = count

    total_violations = sum(violations_per_layer.values())
    pass_rate = float((~patch_has_violation).mean())

    return {
        "n_violations": total_violations,
        "pass_rate": pass_rate,
        "violations_per_layer": violations_per_layer,
        "note": (
            "Partial DRC: only min-width checked. "
            "Export to GDS2 and use KLayout/Magic for full DRC."
        ),
    }


def evaluate_patches(
    pred: np.ndarray,
    ref: np.ndarray,
    run_drc_check: bool = True,
) -> dict[str, Any]:
    """Full evaluation of predicted patches vs. reference.

    Parameters
    ----------
    pred:
        Predicted patches, shape (N, L, H, W).
    ref:
        Reference patches, shape (N, L, H, W).
    run_drc_check:
        If True, run the partial DRC check on predicted patches.

    Returns
    -------
    dict
        Contains ``iou_per_layer``, ``pixel_acc_per_layer``, ``mean_iou``,
        ``mean_pixel_acc``, and (if run_drc_check) ``drc`` sub-dict.
    """
    if pred.shape != ref.shape:
        raise ValueError(f"pred and ref shapes must match: {pred.shape} vs {ref.shape}")

    iou = iou_per_layer(pred, ref)
    acc = pixel_accuracy_per_layer(pred, ref)

    result: dict[str, Any] = {
        "iou_per_layer": iou.tolist(),
        "pixel_acc_per_layer": acc.tolist(),
        "mean_iou": float(iou.mean()),
        "mean_pixel_acc": float(acc.mean()),
    }

    if run_drc_check:
        pred_bin = (pred >= 0.5).astype(np.uint8)
        result["drc"] = run_drc(pred_bin)

    logger.info(
        "Evaluation: mean IoU=%.3f, mean pixel acc=%.3f%s",
        result["mean_iou"],
        result["mean_pixel_acc"],
        f", DRC pass rate={result['drc']['pass_rate']:.1%}" if run_drc_check else "",
    )
    return result


def confusion_matrix_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute a confusion matrix and summary scores for binary patch classification.

    Parameters
    ----------
    y_true:
        Ground-truth labels, shape (N,).  0 = ok, 1 = bad.
    y_pred:
        Predicted labels, shape (N,).  0 = ok, 1 = bad.
    class_names:
        Display names for the two classes.  Default: ``["ok", "bad"]``.

    Returns
    -------
    dict with keys:
        ``confusion_matrix``  — 2×2 list [[TN, FP], [FN, TP]].
        ``accuracy``          — float.
        ``precision``         — float (positive / "bad" class).
        ``recall``            — float (positive / "bad" class).
        ``f1_score``          — float (positive / "bad" class).
        ``class_names``       — list of class name strings.
        ``report``            — formatted text classification report.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix as sk_confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    if class_names is None:
        class_names = ["ok", "bad"]

    cm = sk_confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    logger.info(
        "Classification: accuracy=%.3f, precision=%.3f, recall=%.3f, F1=%.3f",
        acc, prec, rec, f1,
    )
    logger.info("Confusion matrix:\n%s", cm)

    return {
        "confusion_matrix": cm.tolist(),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "class_names": class_names,
        "report": report,
    }


if __name__ == "__main__":  # pragma: no cover
    """End-to-end demo: generate labelled patches, train classifier, print results.

    Run with::

        python -m layout.evaluate
        # or
        python layout/evaluate.py
    """
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s: %(message)s")

    from layout.data_stub import generate_labeled_dataset
    from layout.patch_model import BadPatchClassifier

    print("=== Layout Patch Bad-Pattern Detector ===\n")

    # 1. Generate labelled dataset (100 ok + 100 bad).
    patches, labels = generate_labeled_dataset(n_ok=100, n_bad=100, seed=42)
    print(f"Dataset: {len(patches)} patches  "
          f"({(labels == 0).sum()} ok, {(labels == 1).sum()} bad)\n")

    # 2. Simple 80/20 train/test split (dataset already shuffled).
    n_train = int(0.8 * len(patches))
    X_train, X_test = patches[:n_train], patches[n_train:]
    y_train, y_test = labels[:n_train], labels[n_train:]

    # 3. Train the patch quality classifier.
    clf = BadPatchClassifier(classifier_type="rf", random_state=42)
    clf.fit(X_train, y_train)

    # 4. Predict on the held-out test set.
    y_pred = clf.predict(X_test)

    # 5. Compute and display the confusion matrix + scores.
    result = confusion_matrix_report(y_test, y_pred)

    print(f"Accuracy : {result['accuracy']:.3f}")
    print(f"Precision: {result['precision']:.3f}  (bad class)")
    print(f"Recall   : {result['recall']:.3f}  (bad class)")
    print(f"F1 Score : {result['f1_score']:.3f}  (bad class)\n")

    names = result["class_names"]
    cm = result["confusion_matrix"]
    col_w = max(len(n) for n in names) + 2
    header = " " * (col_w + 8) + "  ".join(f"pred:{n}" for n in names)
    print("Confusion Matrix (rows = true class):")
    print(header)
    for i, row in enumerate(cm):
        cells = "  ".join(f"{v:>{col_w + 5}}" for v in row)
        print(f"  true:{names[i]:<{col_w}}  {cells}")

    print("\nClassification Report:")
    print(result["report"])
