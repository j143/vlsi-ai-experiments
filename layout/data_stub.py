"""
layout/data_stub.py — Synthetic Layout Patch Data Generator
=============================================================
Generates synthetic layout patches for self-supervised model training.

In the absence of real silicon layout data, this stub creates geometrically
plausible patches using basic design rules from config/tech_placeholder.yaml.

Real layout data sources (when available):
  - Open-source PDK standard cells (e.g., SKY130 digital cells).
  - Export patches via KLayout Python API: ``klayout.db.Layout``.
  - GDS2 parser: ``gdspy`` or ``gdstk``.

Output format:
  - NumPy array, shape (N_patches, N_layers, H, W), dtype uint8.
  - 0 = no material, 1 = drawn shape.
  - Layer ordering: see LAYER_MAP below.

Note: All dimensions are in grid units where 1 unit = min_poly_width.
      Current placeholder: 1 grid unit = 0.15 µm (illustrative).
"""

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
TECH_CONFIG_FILE = _REPO_ROOT / "config" / "tech_placeholder.yaml"

# Layer map: (index, name, description)
# TODO(human): align with actual PDK layer numbers
LAYER_MAP = {
    0: "diff",     # Active diffusion / OD
    1: "poly",     # Polysilicon gate
    2: "contact",  # Contact (metal1 to diff/poly)
    3: "metal1",   # Metal 1
    4: "via1",     # Via 1 (metal1 to metal2)
    5: "metal2",   # Metal 2
    6: "nwell",    # N-well
    7: "pwell",    # P-well / substrate
}
N_LAYERS = len(LAYER_MAP)


class PatchConfig(NamedTuple):
    """Configuration for synthetic patch generation."""

    patch_size: int = 32        # Patch width and height in grid units
    min_feature: int = 1        # Minimum feature size in grid units
    contact_size: int = 1       # Contact/via size in grid units
    contact_pitch: int = 3      # Contact-to-contact pitch in grid units
    metal_width_range: tuple = (2, 8)  # [min, max] metal width in grid units


def _place_rectangle(
    canvas: np.ndarray,
    layer: int,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> None:
    """Draw a filled rectangle on a canvas layer (in-place)."""
    h, w = canvas.shape[1:]
    x1 = min(x0 + width, w)
    y1 = min(y0 + height, h)
    if x1 > x0 and y1 > y0:
        canvas[layer, y0:y1, x0:x1] = 1


def _place_contact_array(
    canvas: np.ndarray,
    layer: int,
    x0: int,
    y0: int,
    nx: int,
    ny: int,
    pitch: int,
    size: int,
) -> None:
    """Place a regular array of contacts/vias."""
    for ix in range(nx):
        for iy in range(ny):
            cx = x0 + ix * pitch
            cy = y0 + iy * pitch
            _place_rectangle(canvas, layer, cx, cy, size, size)


def generate_synthetic_patch(
    cfg: PatchConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one synthetic layout patch.

    The patch represents a simplified transistor finger with:
    - An active diffusion region.
    - Polysilicon gate(s) crossing the diffusion.
    - Source/drain contacts over diffusion.
    - Metal1 straps connecting contacts.

    This is intentionally simplified to establish data pipeline infrastructure.
    Replace with real layout patches from a PDK for production use.

    Parameters
    ----------
    cfg:
        Patch configuration.
    rng:
        NumPy random generator.

    Returns
    -------
    np.ndarray
        Shape (N_LAYERS, H, W), dtype uint8.
    """
    canvas = np.zeros((N_LAYERS, cfg.patch_size, cfg.patch_size), dtype=np.uint8)
    P = cfg.patch_size

    # --- Diffusion (OD) region ---
    diff_x = rng.integers(2, P // 4)
    diff_y = rng.integers(2, P // 4)
    diff_w = rng.integers(P // 2, P - 2 * diff_x)
    diff_h = rng.integers(P // 3, P - 2 * diff_y)
    _place_rectangle(canvas, layer=0, x0=diff_x, y0=diff_y, width=diff_w, height=diff_h)

    # --- Poly gate(s) crossing diffusion ---
    n_gates = rng.integers(1, 4)
    poly_width = cfg.min_feature
    gate_margin = 2
    for _ in range(n_gates):
        gx = rng.integers(diff_x + gate_margin, diff_x + diff_w - gate_margin)
        # Poly extends beyond diffusion in y direction
        poly_y0 = max(0, diff_y - gate_margin)
        poly_y1 = min(P, diff_y + diff_h + gate_margin)
        _place_rectangle(canvas, layer=1, x0=gx, y0=poly_y0,
                         width=poly_width, height=poly_y1 - poly_y0)

    # --- Source/drain contacts over diffusion ---
    contact_zone_x = diff_x + 1
    contact_zone_y = diff_y + 1
    n_cx = max(1, (diff_w - 2) // cfg.contact_pitch)
    n_cy = max(1, (diff_h - 2) // cfg.contact_pitch)
    _place_contact_array(
        canvas, layer=2,
        x0=contact_zone_x, y0=contact_zone_y,
        nx=n_cx, ny=n_cy,
        pitch=cfg.contact_pitch, size=cfg.contact_size,
    )

    # --- Metal1 horizontal strap ---
    metal_y = rng.integers(diff_y, diff_y + diff_h)
    metal_w = rng.integers(*cfg.metal_width_range)
    _place_rectangle(canvas, layer=3, x0=diff_x, y0=metal_y,
                     width=diff_w, height=metal_w)

    # --- N-well (for PMOS — present in ~50% of patches) ---
    if rng.random() > 0.5:
        _place_rectangle(canvas, layer=6, x0=max(0, diff_x - 2), y0=max(0, diff_y - 2),
                         width=diff_w + 4, height=diff_h + 4)

    return canvas


def generate_dataset(
    n_patches: int = 200,
    patch_size: int = 32,
    seed: int = 42,
    out_path: Path | str | None = None,
) -> np.ndarray:
    """Generate a synthetic dataset of layout patches.

    Parameters
    ----------
    n_patches:
        Number of patches to generate.
    patch_size:
        Patch size in grid units (square).
    seed:
        Random seed.
    out_path:
        If provided, save the dataset as a .npy file.

    Returns
    -------
    np.ndarray
        Shape (n_patches, N_LAYERS, patch_size, patch_size), dtype uint8.
    """
    cfg = PatchConfig(patch_size=patch_size)
    rng = np.random.default_rng(seed=seed)
    patches = np.stack([generate_synthetic_patch(cfg, rng) for _ in range(n_patches)])

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, patches)
        logger.info("Saved %d patches to %s, shape %s", n_patches, out_path, patches.shape)

    return patches


def generate_bad_patch(
    cfg: PatchConfig,
    rng: np.random.Generator,
    defect_type: str | None = None,
) -> np.ndarray:
    """Generate one synthetic *bad* layout patch with a known layout violation.

    Bad patches are generated by one of five defect injection strategies:

    - ``"empty"``           — mostly empty; almost no geometry drawn.
    - ``"noise"``           — random pixel noise; no recognisable layout pattern.
    - ``"missing_contacts"``— valid diffusion/poly but metal-1 strap removed,
                              leaving contacts electrically floating.
    - ``"short_circuit"``   — metal-1 drawn directly over a poly gate (gate short).
    - ``"min_width"``       — isolated single-pixel poly/metal features
                              (min-width DRC violation).

    Parameters
    ----------
    cfg:
        Patch configuration.
    rng:
        NumPy random generator.
    defect_type:
        Explicit defect type.  If ``None``, one is chosen uniformly at random.

    Returns
    -------
    np.ndarray
        Shape (N_LAYERS, H, W), dtype uint8.
    """
    defect_types = ["empty", "noise", "missing_contacts", "short_circuit", "min_width"]
    if defect_type is None:
        defect_type = defect_types[int(rng.integers(len(defect_types)))]

    canvas = np.zeros((N_LAYERS, cfg.patch_size, cfg.patch_size), dtype=np.uint8)
    P = cfg.patch_size

    if defect_type == "empty":
        # A handful of isolated pixels — no real geometry.
        for _ in range(int(rng.integers(1, 4))):
            lyr = int(rng.integers(N_LAYERS))
            x, y = int(rng.integers(P)), int(rng.integers(P))
            canvas[lyr, y, x] = 1

    elif defect_type == "noise":
        # Random noise on all layers; no layout structure.
        density = float(rng.uniform(0.05, 0.3))
        canvas = (rng.random((N_LAYERS, P, P)) < density).astype(np.uint8)

    elif defect_type == "missing_contacts":
        # Start from a valid patch, then erase metal-1 (leaves contacts floating).
        canvas = generate_synthetic_patch(cfg, rng)
        canvas[3] = 0  # erase metal1 strap

    elif defect_type == "short_circuit":
        # Metal-1 drawn over a poly gate → gate short.
        diff_x = int(rng.integers(2, max(3, P // 4)))
        diff_y = int(rng.integers(2, max(3, P // 4)))
        diff_w = int(rng.integers(P // 2, max(P // 2 + 1, P - 2 * diff_x)))
        diff_h = int(rng.integers(P // 3, max(P // 3 + 1, P - 2 * diff_y)))
        _place_rectangle(canvas, 0, diff_x, diff_y, diff_w, diff_h)
        # Metal-1 blankets the entire diffusion region (wrong — crosses gate).
        _place_rectangle(canvas, 3, diff_x, diff_y, diff_w, diff_h)
        # Add a poly gate that is now shorted to metal.
        gx = int(rng.integers(diff_x + 2, max(diff_x + 3, diff_x + diff_w - 2)))
        _place_rectangle(canvas, 1, gx, max(0, diff_y - 2), 1, diff_h + 4)

    elif defect_type == "min_width":
        # Isolated single-pixel poly / metal-1 features → min-width violation.
        n_violations = int(rng.integers(3, 8))
        for _ in range(n_violations):
            lyr = int(rng.choice([1, 3]))  # poly or metal1
            x = int(rng.integers(2, P - 2))
            y = int(rng.integers(2, P - 2))
            canvas[lyr, y, x] = 1

    return canvas


def generate_labeled_dataset(
    n_ok: int = 100,
    n_bad: int = 100,
    patch_size: int = 32,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a labelled dataset of ok (0) and bad (1) layout patches.

    Parameters
    ----------
    n_ok:
        Number of "ok" (structurally valid) patches.
    n_bad:
        Number of "bad" (defective) patches.
    patch_size:
        Patch size in grid units (square).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    tuple of (patches, labels)
        - patches: shape (n_ok + n_bad, N_LAYERS, patch_size, patch_size), uint8.
        - labels: shape (n_ok + n_bad,), int64.  0 = ok, 1 = bad.
    """
    cfg = PatchConfig(patch_size=patch_size)
    rng = np.random.default_rng(seed)

    ok_patches = np.stack([generate_synthetic_patch(cfg, rng) for _ in range(n_ok)])
    bad_patches = np.stack([generate_bad_patch(cfg, rng) for _ in range(n_bad)])

    patches = np.concatenate([ok_patches, bad_patches], axis=0)
    labels = np.array([0] * n_ok + [1] * n_bad, dtype=np.int64)

    # Shuffle to avoid ordering bias.
    shuffle_idx = rng.permutation(len(labels))
    logger.info(
        "Generated labelled dataset: %d ok + %d bad = %d total patches",
        n_ok, n_bad, len(labels),
    )
    return patches[shuffle_idx], labels[shuffle_idx]


def mask_patches(
    patches: np.ndarray,
    layer: int,
    mask_fraction: float = 0.2,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly mask a fraction of patches on a given layer.

    Used for self-supervised pre-training: the model learns to predict the masked regions.

    Parameters
    ----------
    patches:
        Shape (N, N_LAYERS, H, W).
    layer:
        Layer index to mask.
    mask_fraction:
        Fraction of patches to mask (0–1).
    rng:
        NumPy random generator.

    Returns
    -------
    tuple of (masked_patches, mask_indices, original_layer_values)
        - masked_patches: copy with selected layer zeroed for chosen patches.
        - mask_indices: indices of patches that were masked.
        - original_layer_values: original values of the masked layer for those patches.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = patches.shape[0]
    n_mask = max(1, int(n * mask_fraction))
    mask_indices = rng.choice(n, size=n_mask, replace=False)

    masked = patches.copy()
    original = patches[mask_indices, layer].copy()
    masked[mask_indices, layer] = 0

    return masked, mask_indices, original
