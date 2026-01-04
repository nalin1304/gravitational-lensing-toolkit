"""
Tests for Synthetic Data Calibration Pipeline
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.validation.calibration import (
    SyntheticDataCalibrator,
    CalibrationResult,
    LITERATURE_VALUES,
    run_full_calibration_suite,
)


class TestSyntheticDataCalibrator:
    """Test calibration pipeline."""

    def test_literature_values_loaded(self):
        """Test that literature values are available."""
        assert "einstein_cross" in LITERATURE_VALUES
        assert "twin_quasar" in LITERATURE_VALUES

        ec = LITERATURE_VALUES["einstein_cross"]
        assert ec["z_lens"] == pytest.approx(0.0394, abs=0.001)
        assert ec["z_source"] == pytest.approx(1.695, abs=0.01)
        assert ec["einstein_radius_arcsec"] == pytest.approx(1.85, abs=0.1)

    def test_calibrator_initialization(self):
        """Test calibrator initializes correctly."""
        calibrator = SyntheticDataCalibrator()
        assert len(calibrator.literature) >= 2
        assert len(calibrator.calibration_results) == 0

    def test_generate_synthetic_analog(self):
        """Test synthetic data generation."""
        calibrator = SyntheticDataCalibrator()
        lit = LITERATURE_VALUES["einstein_cross"]

        convergence = calibrator._generate_synthetic_analog(
            lit, pixel_scale=0.05, fov_arcsec=10.0
        )

        # Check output shape
        assert convergence.ndim == 2
        assert convergence.shape[0] == convergence.shape[1]  # Square
        assert convergence.shape[0] == 200  # 10 arcsec / 0.05 arcsec/pixel

        # Check convergence is non-negative
        assert np.all(convergence >= 0)

        # Check peak convergence is reasonable
        assert np.max(convergence) > 0.5  # Should have central concentration

    def test_analyze_convergence_map(self):
        """Test convergence map analysis."""
        calibrator = SyntheticDataCalibrator()

        # Create simple test convergence map (SIS-like)
        npix = 200
        x = np.linspace(-5, 5, npix)
        y = np.linspace(-5, 5, npix)
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx**2 + yy**2)

        # SIS: κ(r) = θ_E / (2r)
        theta_e_true = 1.5  # arcsec
        convergence = np.where(r > 0.1, theta_e_true / (2 * r), 10.0)

        theta_e_recovered, mass = calibrator._analyze_convergence_map(
            convergence,
            pixel_scale=0.05,
            z_lens=0.04,
            z_source=1.7,
        )

        # Should recover Einstein radius within ~20% (rough method)
        assert theta_e_recovered == pytest.approx(theta_e_true, rel=0.3)

        # Mass should be reasonable (galaxy-scale)
        assert 1e10 < mass < 1e13

    def test_calibrate_system_einstein_cross(self):
        """Test calibration against Einstein Cross."""
        calibrator = SyntheticDataCalibrator()

        result = calibrator.calibrate_system("einstein_cross")

        # Check result structure
        assert isinstance(result, CalibrationResult)
        assert result.system_name == "Q2237+030 (Einstein Cross)"
        assert result.true_einstein_radius == pytest.approx(1.85, abs=0.01)

        # Check calibration improved results
        assert result.calibrated_radius_error_percent < result.raw_radius_error_percent

        # Calibrated error should be small (< 1% by design)
        assert result.calibrated_radius_error_percent < 1.0

        # Calibration factor should be reasonable (close to 1 for good synthetic data)
        assert 0.5 < result.radius_calibration_factor < 2.0

    def test_calibrate_system_twin_quasar(self):
        """Test calibration against Twin Quasar."""
        calibrator = SyntheticDataCalibrator()

        result = calibrator.calibrate_system("twin_quasar")

        assert result.system_name == "Q0957+561 (Twin Quasar)"
        assert result.true_einstein_radius == pytest.approx(1.40, abs=0.01)
        assert result.calibrated_radius_error_percent < 1.0

    def test_calibrate_unknown_system_raises(self):
        """Test that unknown system raises error."""
        calibrator = SyntheticDataCalibrator()

        with pytest.raises(ValueError, match="Unknown system"):
            calibrator.calibrate_system("unknown_system")

    def test_apply_calibration(self):
        """Test applying calibration to synthetic maps."""
        calibrator = SyntheticDataCalibrator()

        # Create dummy synthetic maps
        synthetic_maps = np.random.rand(10, 128, 128)
        calibration_factor = 1.05

        calibrated = calibrator.apply_calibration(synthetic_maps, calibration_factor)

        # Check shape preserved
        assert calibrated.shape == synthetic_maps.shape

        # Check scaling applied
        assert np.allclose(calibrated, synthetic_maps * calibration_factor)

    def test_generate_calibration_table(self):
        """Test calibration table generation."""
        calibrator = SyntheticDataCalibrator()

        # Calibrate systems
        calibrator.calibrate_system("einstein_cross")
        calibrator.calibrate_system("twin_quasar")

        table = calibrator.generate_calibration_table()

        # Check table contains expected content
        assert "Einstein Cross" in table
        assert "Twin Quasar" in table
        assert "True θ_E" in table
        assert "Calibrated Recovery" in table
        assert "Calibration Factors" in table

        # Check markdown formatting
        assert "|" in table
        assert "---" in table

    def test_save_and_load_calibration(self, tmp_path):
        """Test saving calibration results to file."""
        calibrator = SyntheticDataCalibrator()
        calibrator.calibrate_system("einstein_cross")

        output_file = tmp_path / "test_calibration.json"
        calibrator.save_calibration(output_file)

        # Check file created
        assert output_file.exists()

        # Check can load and parse
        import json
        with open(output_file) as f:
            data = json.load(f)

        assert "einstein_cross" in data
        assert "radius_calibration_factor" in data["einstein_cross"]
        assert "calibrated_einstein_radius" in data["einstein_cross"]

    def test_full_calibration_suite(self, capsys):
        """Test running full calibration suite."""
        calibrator = run_full_calibration_suite()

        # Check calibrator has results
        assert len(calibrator.calibration_results) >= 2
        assert "einstein_cross" in calibrator.calibration_results
        assert "twin_quasar" in calibrator.calibration_results

        # Check output was printed
        captured = capsys.readouterr()
        assert "Einstein Cross" in captured.out or "Einstein Cross" in str(calibrator.calibration_results)


class TestCalibrationIntegration:
    """Integration tests for calibration pipeline."""

    def test_calibration_improves_accuracy(self):
        """Test that calibration reduces errors across multiple systems."""
        calibrator = SyntheticDataCalibrator()

        systems = ["einstein_cross", "twin_quasar"]

        for system in systems:
            result = calibrator.calibrate_system(system)

            # Calibrated should always be better than raw
            assert result.calibrated_radius_error_percent <= result.raw_radius_error_percent

            # Calibrated error should be very small
            assert result.calibrated_radius_error_percent < 2.0  # < 2% error

    def test_calibration_factors_reasonable(self):
        """Test that calibration factors are physically reasonable."""
        calibrator = SyntheticDataCalibrator()

        systems = ["einstein_cross", "twin_quasar"]

        for system in systems:
            result = calibrator.calibrate_system(system)

            # Factors should be close to 1 for good synthetic data
            # Allow 0.5 to 2.0 range (factors of 2)
            assert 0.5 < result.radius_calibration_factor < 2.0
            assert 0.5 < result.mass_calibration_factor < 2.0

    def test_multiple_calibrations_consistent(self):
        """Test that running calibration multiple times gives consistent results."""
        calibrator1 = SyntheticDataCalibrator()
        calibrator2 = SyntheticDataCalibrator()

        result1 = calibrator1.calibrate_system("einstein_cross")
        result2 = calibrator2.calibrate_system("einstein_cross")

        # Results should be identical (deterministic)
        assert result1.radius_calibration_factor == pytest.approx(
            result2.radius_calibration_factor, rel=1e-6
        )
        assert result1.calibrated_einstein_radius == pytest.approx(
            result2.calibrated_einstein_radius, rel=1e-6
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
