"""
tests/test_cli.py — Tests for vlsi_ai/cli.py
=============================================
Covers the ``vlsi-ai`` CLI entry-point: argument parsing, sub-command dispatch,
and integration smoke-tests (no ngspice required).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlsi_ai.cli import _build_parser, main  # noqa: E402


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestParser:
    def test_sweep_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["sweep"])
        assert args.command == "sweep"
        assert args.n_samples == 80
        assert args.mode == "lhs"
        assert args.seed == 42

    def test_sweep_custom_args(self):
        parser = _build_parser()
        args = parser.parse_args([
            "sweep", "--n-samples", "20", "--mode", "grid",
            "--out", "/tmp/test.csv", "--seed", "7",
        ])
        assert args.n_samples == 20
        assert args.mode == "grid"
        assert args.out == "/tmp/test.csv"
        assert args.seed == 7

    def test_optimize_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["optimize"])
        assert args.command == "optimize"
        assert args.budget == 30
        assert args.n_init == 10
        assert args.seed == 42

    def test_optimize_custom_budget(self):
        parser = _build_parser()
        args = parser.parse_args(["optimize", "--budget", "15", "--seed", "99"])
        assert args.budget == 15
        assert args.seed == 99

    def test_demo_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["demo"])
        assert args.command == "demo"
        assert args.budget == 30
        assert args.n_init == 10
        assert args.grid_n == 3
        assert args.n_samples == 100
        assert args.dataset is None

    def test_demo_with_dataset(self):
        parser = _build_parser()
        args = parser.parse_args(["demo", "--dataset", "datasets/test.csv"])
        assert args.dataset == "datasets/test.csv"

    def test_no_subcommand_exits(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_version_flag(self):
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# sweep sub-command (integration, no ngspice)
# ---------------------------------------------------------------------------

class TestCmdSweep:
    def test_sweep_lhs_writes_csv(self, tmp_path):
        out_csv = tmp_path / "sweep_out.csv"
        rc = main(["sweep", "--n-samples", "5", "--mode", "lhs",
                   "--out", str(out_csv), "--seed", "0"])
        assert rc == 0
        assert out_csv.exists()
        import pandas as pd
        df = pd.read_csv(out_csv)
        assert len(df) == 5
        assert "vref_V" in df.columns

    def test_sweep_grid_writes_csv(self, tmp_path):
        out_csv = tmp_path / "grid_out.csv"
        # n_per_dim=2 → 2^5=32 points
        rc = main(["sweep", "--n-samples", "2", "--mode", "grid",
                   "--out", str(out_csv)])
        assert rc == 0
        import pandas as pd
        df = pd.read_csv(out_csv)
        assert len(df) == 32

    def test_sweep_to_directory_creates_file(self, tmp_path):
        rc = main(["sweep", "--n-samples", "3", "--out", str(tmp_path)])
        assert rc == 0
        csv_files = list(tmp_path.glob("bandgap_sweep_*.csv"))
        assert len(csv_files) == 1

    def test_sweep_csv_has_spec_columns(self, tmp_path):
        out_csv = tmp_path / "spec_test.csv"
        rc = main(["sweep", "--n-samples", "4", "--out", str(out_csv)])
        assert rc == 0
        import pandas as pd
        df = pd.read_csv(out_csv)
        # At least one spec column should be present
        spec_cols = [c for c in df.columns if c.startswith("spec_")]
        assert len(spec_cols) >= 1


# ---------------------------------------------------------------------------
# optimize sub-command (integration, no ngspice)
# ---------------------------------------------------------------------------

class TestCmdOptimize:
    def _make_dataset(self, tmp_path):
        """Create a minimal synthetic dataset CSV for tests."""
        from ml.surrogate import _generate_synthetic_data

        df = _generate_synthetic_data(n=30, seed=0)
        csv_path = tmp_path / "test_dataset.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_optimize_runs_and_returns_zero(self, tmp_path):
        csv_path = self._make_dataset(tmp_path)
        out_dir = tmp_path / "bo_out"
        rc = main([
            "optimize",
            "--dataset", str(csv_path),
            "--budget", "5",
            "--n-init", "3",
            "--out", str(out_dir),
            "--seed", "0",
        ])
        assert rc == 0

    def test_optimize_writes_json(self, tmp_path):
        csv_path = self._make_dataset(tmp_path)
        out_dir = tmp_path / "bo_out"
        main([
            "optimize",
            "--dataset", str(csv_path),
            "--budget", "5",
            "--n-init", "3",
            "--out", str(out_dir),
        ])
        json_files = list(out_dir.glob("bo_run_*.json"))
        assert len(json_files) == 1
        with open(json_files[0]) as f:
            data = json.load(f)
        assert "n_simulations" in data

    def test_optimize_missing_dataset_returns_nonzero(self, tmp_path):
        rc = main([
            "optimize",
            "--dataset", str(tmp_path / "nonexistent.csv"),
            "--budget", "5",
            "--out", str(tmp_path / "bo_out"),
        ])
        assert rc != 0


# ---------------------------------------------------------------------------
# demo sub-command (integration, no ngspice)
# ---------------------------------------------------------------------------

class TestCmdDemo:
    def test_demo_runs_and_returns_zero(self, tmp_path):
        rc = main([
            "demo",
            "--n-samples", "20",
            "--budget", "5",
            "--n-init", "3",
            "--grid-n", "2",
            "--out", str(tmp_path),
            "--seed", "0",
        ])
        assert rc == 0

    def test_demo_writes_report_md(self, tmp_path):
        main([
            "demo",
            "--n-samples", "20",
            "--budget", "5",
            "--n-init", "3",
            "--grid-n", "2",
            "--out", str(tmp_path),
            "--seed", "0",
        ])
        assert (tmp_path / "demo_report.md").exists()

    def test_demo_writes_summary_json(self, tmp_path):
        main([
            "demo",
            "--n-samples", "20",
            "--budget", "5",
            "--n-init", "3",
            "--grid-n", "2",
            "--out", str(tmp_path),
            "--seed", "0",
        ])
        summary_path = tmp_path / "demo_summary.json"
        assert summary_path.exists()
        with open(summary_path) as f:
            data = json.load(f)
        assert "grid" in data
        assert "bayesian_optimization" in data
        assert "simulation_reduction_pct" in data

    def test_demo_markdown_has_table(self, tmp_path):
        main([
            "demo",
            "--n-samples", "20",
            "--budget", "5",
            "--n-init", "3",
            "--grid-n", "2",
            "--out", str(tmp_path),
            "--seed", "0",
        ])
        content = (tmp_path / "demo_report.md").read_text()
        assert "Grid" in content
        assert "BO" in content
        assert "|" in content  # Markdown table row

    def test_demo_with_existing_dataset(self, tmp_path):
        from ml.surrogate import _generate_synthetic_data

        df = _generate_synthetic_data(n=30, seed=5)
        ds_path = tmp_path / "synth.csv"
        df.to_csv(ds_path, index=False)

        rc = main([
            "demo",
            "--dataset", str(ds_path),
            "--budget", "5",
            "--n-init", "3",
            "--grid-n", "2",
            "--out", str(tmp_path / "demo_out"),
            "--seed", "1",
        ])
        assert rc == 0
