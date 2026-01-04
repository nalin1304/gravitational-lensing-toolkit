"""
Unit tests for real data loading module (Phase 8).

Tests FITS loading, PSF modeling, and data preprocessing.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import warnings

# Import with optional dependency handling
try:
    from astropy.io import fits
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

from src.data.real_data_loader import (
    ObservationMetadata,
    PSFModel,
    preprocess_real_data,
    SCIPY_AVAILABLE
)

# Conditional imports
if ASTROPY_AVAILABLE:
    from src.data.real_data_loader import FITSDataLoader, load_real_data


@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="Astropy not installed")
class TestFITSDataLoader:
    """Test FITS file loading."""
    
    def create_test_fits(self, tmpdir, with_data=True):
        """Helper to create a test FITS file."""
        filepath = Path(tmpdir) / "test.fits"
        
        # Create test data
        if with_data:
            data = np.random.randn(100, 100).astype(np.float32)
        else:
            data = None
        
        # Create header with metadata
        hdu = fits.PrimaryHDU(data)
        hdu.header['TELESCOP'] = 'HST'
        hdu.header['INSTRUME'] = 'ACS/WFC'
        hdu.header['FILTER'] = 'F814W'
        hdu.header['EXPTIME'] = 1200.0
        hdu.header['CD1_1'] = -0.05 / 3600.0  # 0.05 arcsec/pixel
        hdu.header['CD2_2'] = 0.05 / 3600.0
        hdu.header['RA_TARG'] = 150.0
        hdu.header['DEC_TARG'] = 2.5
        hdu.header['DATE-OBS'] = '2023-01-15'
        
        hdul = fits.HDUList([hdu])
        hdul.writeto(filepath, overwrite=True)
        
        return filepath
    
    def test_loader_initialization(self):
        """Test loader initializes correctly."""
        loader = FITSDataLoader()
        assert loader is not None
    
    def test_load_fits_basic(self, tmpdir):
        """Test basic FITS loading."""
        filepath = self.create_test_fits(tmpdir)
        
        loader = FITSDataLoader()
        data, metadata = loader.load_fits(filepath)
        
        assert data.shape == (100, 100)
        assert data.dtype == np.float64
        assert isinstance(metadata, ObservationMetadata)
    
    def test_metadata_extraction(self, tmpdir):
        """Test metadata is correctly extracted."""
        filepath = self.create_test_fits(tmpdir)
        
        loader = FITSDataLoader()
        _, metadata = loader.load_fits(filepath)
        
        assert metadata.telescope == 'HST'
        assert metadata.instrument == 'ACS/WFC'
        assert metadata.filter_name == 'F814W'
        assert metadata.exposure_time == 1200.0
        assert metadata.pixel_scale == pytest.approx(0.05, rel=0.01)
        assert metadata.ra == 150.0
        assert metadata.dec == 2.5
        assert metadata.date_obs == '2023-01-15'
    
    def test_load_with_header(self, tmpdir):
        """Test loading with header return."""
        filepath = self.create_test_fits(tmpdir)
        
        loader = FITSDataLoader()
        data, metadata, header = loader.load_fits(filepath, return_header=True)
        
        assert data.shape == (100, 100)
        assert metadata.telescope == 'HST'
        assert 'TELESCOP' in header
    
    def test_file_not_found(self):
        """Test error handling for missing file."""
        loader = FITSDataLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_fits("nonexistent_file.fits")
    
    def test_empty_extension(self, tmpdir):
        """Test handling of empty extension."""
        filepath = Path(tmpdir) / "test.fits"
        
        # Create FITS with empty extension
        hdu1 = fits.PrimaryHDU()  # No data
        hdu2 = fits.ImageHDU(np.random.randn(50, 50))
        hdul = fits.HDUList([hdu1, hdu2])
        hdul.writeto(filepath)
        
        loader = FITSDataLoader()
        
        # Extension 0 (empty) should raise error
        with pytest.raises(ValueError, match="contains no data"):
            loader.load_fits(filepath, extension=0)
        
        # Extension 1 (has data) should work
        data, _ = loader.load_fits(filepath, extension=1)
        assert data.shape == (50, 50)
    
    def test_list_extensions(self, tmpdir):
        """Test listing FITS extensions."""
        filepath = Path(tmpdir) / "test_multi.fits"
        
        # Create multi-extension FITS
        hdu1 = fits.PrimaryHDU(np.ones((50, 50)))
        hdu2 = fits.ImageHDU(np.zeros((100, 100)), name='SCI')
        hdu3 = fits.ImageHDU(np.zeros((100, 100)), name='ERR')
        hdul = fits.HDUList([hdu1, hdu2, hdu3])
        hdul.writeto(filepath)
        
        loader = FITSDataLoader()
        extensions = loader.list_extensions(filepath)
        
        assert len(extensions) == 3
        assert extensions[0]['index'] == 0
        assert extensions[1]['name'] == 'SCI'
        assert extensions[2]['name'] == 'ERR'
        assert extensions[1]['shape'] == (100, 100)
    
    def test_pixel_scale_extraction_methods(self, tmpdir):
        """Test different methods of pixel scale extraction."""
        loader = FITSDataLoader()
        
        # Method 1: CD matrix
        filepath1 = Path(tmpdir) / "test_cd.fits"
        hdu = fits.PrimaryHDU(np.ones((10, 10)))
        hdu.header['CD1_1'] = -0.04 / 3600.0
        hdu.header['CD2_2'] = 0.04 / 3600.0
        fits.HDUList([hdu]).writeto(filepath1)
        
        _, meta1 = loader.load_fits(filepath1)
        assert meta1.pixel_scale == pytest.approx(0.04, rel=0.01)
        
        # Method 2: CDELT keywords
        filepath2 = Path(tmpdir) / "test_cdelt.fits"
        hdu = fits.PrimaryHDU(np.ones((10, 10)))
        hdu.header['CDELT1'] = -0.03 / 3600.0
        hdu.header['CDELT2'] = 0.03 / 3600.0
        fits.HDUList([hdu]).writeto(filepath2)
        
        _, meta2 = loader.load_fits(filepath2)
        assert meta2.pixel_scale == pytest.approx(0.03, rel=0.01)
        
        # Method 3: PIXSCALE keyword
        filepath3 = Path(tmpdir) / "test_pixscale.fits"
        hdu = fits.PrimaryHDU(np.ones((10, 10)))
        hdu.header['PIXSCALE'] = 0.13
        fits.HDUList([hdu]).writeto(filepath3)
        
        _, meta3 = loader.load_fits(filepath3)
        assert meta3.pixel_scale == 0.13


class TestPSFModel:
    """Test PSF modeling."""
    
    def test_psf_initialization(self):
        """Test PSF model initializes correctly."""
        psf = PSFModel(fwhm=0.1, pixel_scale=0.05)
        
        assert psf.fwhm == 0.1
        assert psf.pixel_scale == 0.05
        assert psf.sigma_pixels > 0
    
    def test_generate_psf_shape(self):
        """Test PSF generation produces correct shape."""
        psf = PSFModel(fwhm=0.1, pixel_scale=0.05)
        
        psf_image = psf.generate_psf(size=25)
        
        assert psf_image.shape == (25, 25)
        assert psf_image.dtype == np.float64
    
    def test_generate_psf_odd_size(self):
        """Test PSF generation enforces odd size."""
        psf = PSFModel(fwhm=0.1, pixel_scale=0.05)
        
        # Even size should be converted to odd
        psf_image = psf.generate_psf(size=24)
        assert psf_image.shape[0] % 2 == 1  # Odd
    
    def test_psf_normalization(self):
        """Test PSF is properly normalized."""
        psf = PSFModel(fwhm=0.1, pixel_scale=0.05)
        
        psf_image = psf.generate_psf(size=51)
        
        # Sum should be approximately 1
        assert np.sum(psf_image) == pytest.approx(1.0, rel=1e-6)
    
    def test_psf_symmetry(self):
        """Test PSF is symmetric."""
        psf = PSFModel(fwhm=0.1, pixel_scale=0.05)
        
        psf_image = psf.generate_psf(size=25)
        
        # Should be symmetric about center
        center = 12
        assert np.allclose(psf_image[center, :], psf_image[:, center])
    
    def test_psf_peak_at_center(self):
        """Test PSF has maximum at center."""
        psf = PSFModel(fwhm=0.1, pixel_scale=0.05)
        
        psf_image = psf.generate_psf(size=25)
        
        center = 12
        peak_val = psf_image[center, center]
        
        # Center should have maximum value
        assert peak_val == np.max(psf_image)
    
    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not installed")
    def test_convolve_image(self):
        """Test image convolution with PSF."""
        psf_model = PSFModel(fwhm=0.1, pixel_scale=0.05)
        
        # Create test image (point source)
        image = np.zeros((50, 50))
        image[25, 25] = 1.0
        
        # Convolve
        convolved = psf_model.convolve_image(image)
        
        assert convolved.shape == image.shape
        # Peak should still be near center
        assert np.unravel_index(np.argmax(convolved), convolved.shape) == (25, 25)
        # But now spread out (not a delta function)
        assert np.sum(convolved > 0.01) > 10


class TestPreprocessRealData:
    """Test data preprocessing."""
    
    def create_test_metadata(self):
        """Helper to create test metadata."""
        return ObservationMetadata(
            telescope='HST',
            instrument='ACS/WFC',
            filter_name='F814W',
            exposure_time=1200.0,
            pixel_scale=0.05
        )
    
    def test_preprocess_basic(self):
        """Test basic preprocessing."""
        data = np.random.randn(100, 100)
        metadata = self.create_test_metadata()
        
        processed = preprocess_real_data(data, metadata)
        
        assert processed.shape == data.shape
        assert np.all(np.isfinite(processed))
    
    def test_handle_nans_zero(self):
        """Test NaN handling with zero replacement."""
        data = np.random.randn(50, 50)
        data[10:20, 10:20] = np.nan
        metadata = self.create_test_metadata()
        
        processed = preprocess_real_data(
            data,
            metadata,
            handle_nans='zero',
            normalize=False  # Don't normalize to check raw value
        )
        
        assert not np.any(np.isnan(processed))
        assert processed[15, 15] == 0.0
    
    def test_handle_nans_median(self):
        """Test NaN handling with median replacement."""
        data = np.random.randn(50, 50)
        data[10:20, 10:20] = np.nan
        valid_median = np.median(data[~np.isnan(data)])
        metadata = self.create_test_metadata()
        
        processed = preprocess_real_data(
            data,
            metadata,
            handle_nans='median',
            normalize=False  # Don't normalize to check raw value
        )
        
        assert not np.any(np.isnan(processed))
        assert processed[15, 15] == pytest.approx(valid_median, rel=0.1)
    
    def test_normalization(self):
        """Test data normalization."""
        data = np.random.randn(50, 50) * 100 + 50
        metadata = self.create_test_metadata()
        
        processed = preprocess_real_data(
            data,
            metadata,
            normalize=True
        )
        
        assert processed.min() == pytest.approx(0.0, abs=1e-10)
        assert processed.max() == pytest.approx(1.0, abs=1e-10)
    
    def test_no_normalization(self):
        """Test preprocessing without normalization."""
        data = np.random.randn(50, 50) * 100 + 50
        metadata = self.create_test_metadata()
        
        processed = preprocess_real_data(
            data,
            metadata,
            normalize=False
        )
        
        # Should preserve scale
        assert not (processed.min() == 0.0 and processed.max() == 1.0)
    
    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not installed")
    def test_resize(self):
        """Test image resizing."""
        data = np.random.randn(100, 100)
        metadata = self.create_test_metadata()
        
        processed = preprocess_real_data(
            data,
            metadata,
            target_size=(64, 64)
        )
        
        assert processed.shape == (64, 64)
    
    def test_handle_inf_values(self):
        """Test handling of infinite values."""
        data = np.random.randn(50, 50)
        data[20, 20] = np.inf
        data[30, 30] = -np.inf
        metadata = self.create_test_metadata()
        
        processed = preprocess_real_data(
            data,
            metadata,
            handle_nans='zero'
        )
        
        assert np.all(np.isfinite(processed))


class TestObservationMetadata:
    """Test observation metadata dataclass."""
    
    def test_metadata_creation(self):
        """Test creating metadata object."""
        meta = ObservationMetadata(
            telescope='HST',
            instrument='ACS',
            filter_name='F814W',
            exposure_time=1200.0,
            pixel_scale=0.05
        )
        
        assert meta.telescope == 'HST'
        assert meta.instrument == 'ACS'
        assert meta.exposure_time == 1200.0
    
    def test_metadata_optional_fields(self):
        """Test optional metadata fields."""
        meta = ObservationMetadata(
            telescope='JWST',
            instrument='NIRCam',
            filter_name='F200W',
            exposure_time=600.0,
            pixel_scale=0.031,
            ra=53.1,
            dec=-27.8,
            date_obs='2024-03-15'
        )
        
        assert meta.ra == 53.1
        assert meta.dec == -27.8
        assert meta.date_obs == '2024-03-15'


@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="Astropy not installed")
class TestIntegration:
    """Integration tests for real data loading."""
    
    def test_load_real_data_convenience(self, tmpdir):
        """Test convenience function for loading real data."""
        # Create test FITS
        filepath = Path(tmpdir) / "test.fits"
        data = np.random.randn(100, 100).astype(np.float32)
        hdu = fits.PrimaryHDU(data)
        hdu.header['TELESCOP'] = 'HST'
        hdu.header['INSTRUME'] = 'WFC3'
        hdu.header['FILTER'] = 'F160W'
        hdu.header['EXPTIME'] = 900.0
        hdu.header['PIXSCALE'] = 0.13
        fits.HDUList([hdu]).writeto(filepath)
        
        # Load using convenience function
        processed, metadata = load_real_data(
            filepath,
            normalize=True
        )
        
        assert processed.shape == (100, 100)
        assert processed.min() >= 0.0
        assert processed.max() <= 1.0
        assert metadata.instrument == 'WFC3'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
