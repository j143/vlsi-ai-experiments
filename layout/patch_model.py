"""
layout/patch_model.py — UNet-style Patch Model for Layout Prediction
======================================================================
Implements a lightweight UNet encoder-decoder for self-supervised layout
patch completion. The model:
  1. Takes a masked layout patch (N_layers × H × W) as input.
  2. Predicts the missing shapes (contacts, vias, metal segments).

Architecture follows foundation-model principles:
  - Pre-training: self-supervised masking (predict masked layer from context).
  - Fine-tuning: supervised prediction for a specific downstream task
    (e.g., contact insertion, via generation).

This module is intentionally framework-agnostic at the interface level but
uses PyTorch for the implementation. If PyTorch is not installed, a stub
class is provided so that the rest of the codebase (tests, data pipelines)
can import without error.

Usage::

    import numpy as np
    from layout.patch_model import UNetPatchModel

    model = UNetPatchModel(n_layers=8, patch_size=32)
    # batch of 4 patches, 8 layers, 32×32
    x = np.random.randint(0, 2, (4, 8, 32, 32)).astype(np.float32)
    pred = model.predict(x)  # shape (4, 8, 32, 32)

Training is covered by ``layout/train.py`` (see Milestone 3 in ROADMAP.md).
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch not found. UNetPatchModel will use a stub (identity) implementation. "
        "Install PyTorch to enable real model training: pip install torch"
    )


# ---------------------------------------------------------------------------
# UNet building blocks
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class _DoubleConv(nn.Module):  # type: ignore[misc]
        """Two consecutive Conv2d → BatchNorm → ReLU blocks."""

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.block(x)

    class _Down(nn.Module):  # type: ignore[misc]
        """Downsampling block: MaxPool2d + DoubleConv."""

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.MaxPool2d(2),
                _DoubleConv(in_channels, out_channels),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.block(x)

    class _Up(nn.Module):  # type: ignore[misc]
        """Upsampling block: bilinear upsample + DoubleConv + skip connection."""

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = _DoubleConv(in_channels, out_channels)

        def forward(
            self, x: "torch.Tensor", skip: "torch.Tensor"
        ) -> "torch.Tensor":
            x = self.up(x)
            # Handle odd-sized inputs: pad if necessary
            dy = skip.shape[2] - x.shape[2]
            dx = skip.shape[3] - x.shape[3]
            if dy > 0 or dx > 0:
                x = torch.nn.functional.pad(x, [0, dx, 0, dy])
            x = torch.cat([skip, x], dim=1)
            return self.conv(x)

    class _UNetCore(nn.Module):  # type: ignore[misc]
        """UNet encoder-decoder core."""

        def __init__(self, n_channels: int, n_classes: int, base_features: int = 32) -> None:
            super().__init__()
            f = base_features
            self.inc = _DoubleConv(n_channels, f)
            self.down1 = _Down(f, f * 2)
            self.down2 = _Down(f * 2, f * 4)
            self.down3 = _Down(f * 4, f * 8)
            self.bottleneck = _DoubleConv(f * 8, f * 8)
            self.up1 = _Up(f * 8 + f * 8, f * 4)
            self.up2 = _Up(f * 4 + f * 4, f * 2)
            self.up3 = _Up(f * 2 + f * 2, f)
            self.out_conv = nn.Conv2d(f + f, n_classes, kernel_size=1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.bottleneck(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            # Final upsample + skip connection with x1
            x = torch.nn.functional.interpolate(
                x, size=x1.shape[2:], mode="bilinear", align_corners=True
            )
            x = torch.cat([x1, x], dim=1)
            return self.out_conv(x)


# ---------------------------------------------------------------------------
# Public model class
# ---------------------------------------------------------------------------


class UNetPatchModel:
    """UNet-style layout patch completion model.

    Parameters
    ----------
    n_layers:
        Number of layout layers (input and output channels).
    patch_size:
        Expected patch height = width in grid units.
    base_features:
        Number of feature maps in the first encoder stage.
    device:
        Torch device string ('cpu', 'cuda', 'mps'). Auto-detected if None.
    """

    def __init__(
        self,
        n_layers: int = 8,
        patch_size: int = 32,
        base_features: int = 32,
        device: str | None = None,
    ) -> None:
        self.n_layers = n_layers
        self.patch_size = patch_size
        self._fitted = False

        if not _TORCH_AVAILABLE:
            logger.warning("UNetPatchModel: using stub (identity) — PyTorch not installed.")
            self._net = None
            self._device = "cpu"
            return

        if device is None:
            self._device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        else:
            self._device = device

        self._net = _UNetCore(
            n_channels=n_layers, n_classes=n_layers, base_features=base_features
        ).to(self._device)
        logger.info(
            "UNetPatchModel initialized (device=%s, params=%d)",
            self._device,
            sum(p.numel() for p in self._net.parameters()),
        )

    def predict(self, patches: np.ndarray) -> np.ndarray:
        """Run forward pass on a batch of patches.

        Parameters
        ----------
        patches:
            Input array, shape (N, n_layers, H, W), dtype float32 (values 0 or 1).

        Returns
        -------
        np.ndarray
            Predicted patch, shape (N, n_layers, H, W), values in [0, 1].
            Threshold at 0.5 to get binary predictions.
        """
        if patches.ndim != 4:
            raise ValueError(f"Expected 4D input, got shape {patches.shape}")
        if patches.shape[1] != self.n_layers:
            raise ValueError(
                f"Expected {self.n_layers} channels, got {patches.shape[1]}"
            )

        if not _TORCH_AVAILABLE or self._net is None:
            # Stub: return input unchanged (identity)
            return patches.copy()

        self._net.eval()
        with torch.no_grad():
            x = torch.tensor(patches, dtype=torch.float32, device=self._device)
            logits = self._net(x)
            probs = torch.sigmoid(logits)
            return probs.cpu().numpy()

    def save(self, path: str | Path) -> None:
        """Save model weights to a file."""
        if not _TORCH_AVAILABLE or self._net is None:
            raise RuntimeError("Cannot save: PyTorch not available.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), path)
        logger.info("Model saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load model weights from a file."""
        if not _TORCH_AVAILABLE or self._net is None:
            raise RuntimeError("Cannot load: PyTorch not available.")
        state = torch.load(path, map_location=self._device)
        self._net.load_state_dict(state)
        self._fitted = True
        logger.info("Model loaded from %s", path)
