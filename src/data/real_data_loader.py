"""
Real Data Integration Module (Phase 8)

This module provides tools for loading and processing real observational data
from telescopes like HST (Hubble Space Telescope) and JWST (James Webb Space Telescope).

Features:
- FITS file loading and parsing
- HST/JWST data compatibility
- PSF (Point Spread Function) modeling
- Realistic noise characterization
- Data preprocessing and normalization
- Header metadata extraction

Supported Formats:
- FITS (Flexible Image Transport System)
- HST ACS/WFC3 images
- JWST NIRCam/MIRI images
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass
import warnings

# Optional dependencies - graceful fallback
try:
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    ASTROPY_AVAILABLE = True
except ImportError:
    fits = None
    WCS = None
    u = None
    SkyCoord = None
    ASTROPY_AVAILABLE = False
    warnings.warn(
        "Astropy not installed. Real data loading functionality limited. "
        "Install with: pip install astropy",
        ImportWarning
    )

try:
    from scipy.ndimage import gaussian_filter
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    gaussian_filter = None
    interp1d = None
    SCIPY_AVAILABLE = False


@dataclass
class ObservationMetadata:
    """
    Metadata from telescope observations.
    
    Attributes
    ----------
    telescope : str
        Telescope name (HST, JWST, etc.)
    instrument : str
        Instrument name (ACS, WFC3, NIRCam, etc.)
    filter_name : str
        Filter used for observation
    exposure_time : float
        Exposure time in seconds
    pixel_scale : float
        Pixel scale in arcseconds/pixel
    ra : float, optional
        Right Ascension of field center (degrees)
    dec : float, optional
        Declination of field center (degrees)
    date_obs : str, optional
        Observation date
    """
    telescope: str
    instrument: str
    filter_name: str
    exposure_time: float
    pixel_scale: float
    ra: Optional[float] = None
    dec: Optional[float] = None
    date_obs: Optional[str] = None


class FITSDataLoader:
    """
    Load and process FITS files from telescopes.
    
    This class provides methods to:
    - Load FITS images
    - Extract metadata
    - Handle multi-extension FITS files
    - Convert units
    - Extract WCS information
    
    Examples
    --------
    >>> loader = FITSDataLoader()
    >>> data, metadata = loader.load_fits("observation.fits")
    >>> print(f"Telescope: {metadata.telescope}")
    >>> print(f"Image shape: {data.shape}")
    """
    
    def __init__(self):
        """Initialize FITS data loader."""
        if not ASTROPY_AVAILABLE:
            raise ImportError(
                "Astropy is required for FITS loading. "
                "Install with: pip install astropy"
            )
    
    def load_fits(
        self,
        filepath: Union[str, Path],
        extension: int = 0,
        return_header: bool = False
    ) -> Union[Tuple[np.ndarray, ObservationMetadata], 
               Tuple[np.ndarray, ObservationMetadata, object]]:
        """
        Load FITS file and extract data with metadata.
        
        Parameters
        ----------
        filepath : str or Path
            Path to FITS file
        extension : int
            FITS extension to read (default: 0 for primary HDU)
        return_header : bool
            If True, also return the full FITS header
        
        Returns
        -------
        data : np.ndarray
            Image data
        metadata : ObservationMetadata
            Extracted metadata
        header : FITS header (optional)
            Full FITS header if return_header=True
        
        Raises
        ------
        FileNotFoundError
            If FITS file doesn't exist
        ValueError
            If FITS file is corrupted or invalid
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"FITS file not found: {filepath}")
        
        try:
            with fits.open(filepath) as hdul:
                # Get data from specified extension
                data = hdul[extension].data
                header = hdul[extension].header
                
                # Handle case where data is None (empty extension)
                if data is None:
                    raise ValueError(
                        f"Extension {extension} contains no data. "
                        f"Available extensions: {len(hdul)}"
                    )
                
                # Convert to float64 for consistency
                data = np.array(data, dtype=np.float64)
                
                # Extract metadata
                metadata = self._extract_metadata(header)
                
        except Exception as e:
            raise ValueError(f"Error reading FITS file: {e}")
        
        if return_header:
            return data, metadata, header
        else:
            return data, metadata
    
    def _extract_metadata(self, header) -> ObservationMetadata:
        """
        Extract metadata from FITS header.
        
        Parameters
        ----------
        header : FITS header
            FITS file header
        
        Returns
        -------
        metadata : ObservationMetadata
            Extracted observation metadata
        """
        # Try different header keywords for different telescopes
        telescope = (
            header.get('TELESCOP', 'Unknown') or 
            header.get('INSTRUME', 'Unknown')
        )
        
        instrument = header.get('INSTRUME', 'Unknown')
        
        filter_name = (
            header.get('FILTER', 'Unknown') or
            header.get('FILTER1', 'Unknown') or
            header.get('FILTNAM1', 'Unknown')
        )
        
        exposure_time = float(header.get('EXPTIME', 0.0) or 0.0)
        
        # Pixel scale (arcsec/pixel)
        # Try CD matrix first, then CDELT
        pixel_scale = self._extract_pixel_scale(header)
        
        # Coordinates
        ra = header.get('RA_TARG', None) or header.get('CRVAL1', None)
        dec = header.get('DEC_TARG', None) or header.get('CRVAL2', None)
        
        date_obs = header.get('DATE-OBS', None)
        
        return ObservationMetadata(
            telescope=str(telescope),
            instrument=str(instrument),
            filter_name=str(filter_name),
            exposure_time=exposure_time,
            pixel_scale=pixel_scale,
            ra=float(ra) if ra is not None else None,
            dec=float(dec) if dec is not None else None,
            date_obs=str(date_obs) if date_obs is not None else None
        )
    
    def _extract_pixel_scale(self, header) -> float:
        """
        Extract pixel scale from FITS header.
        
        Tries multiple methods:
        1. CD matrix (CD1_1, CD2_2)
        2. CDELT keywords
        3. PIXSCALE keyword
        4. Default value
        
        Parameters
        ----------
        header : FITS header
            FITS file header
        
        Returns
        -------
        pixel_scale : float
            Pixel scale in arcseconds/pixel
        """
        # Method 1: CD matrix (most accurate)
        cd1_1 = header.get('CD1_1', None)
        cd2_2 = header.get('CD2_2', None)
        
        if cd1_1 is not None and cd2_2 is not None:
            # Convert degrees to arcseconds
            pixel_scale = np.sqrt(abs(cd1_1 * cd2_2)) * 3600.0
            return float(pixel_scale)
        
        # Method 2: CDELT keywords
        cdelt1 = header.get('CDELT1', None)
        cdelt2 = header.get('CDELT2', None)
        
        if cdelt1 is not None and cdelt2 is not None:
            pixel_scale = np.sqrt(abs(cdelt1 * cdelt2)) * 3600.0
            return float(pixel_scale)
        
        # Method 3: Direct PIXSCALE keyword
        pixscale = header.get('PIXSCALE', None)
        if pixscale is not None:
            return float(pixscale)
        
        # Method 4: Instrument-specific defaults
        instrument = header.get('INSTRUME', '').upper()
        
        if 'ACS' in instrument and 'WFC' in instrument:
            return 0.05  # HST ACS/WFC: ~0.05 arcsec/pixel
        elif 'WFC3' in instrument and 'UVIS' in instrument:
            return 0.04  # HST WFC3/UVIS: ~0.04 arcsec/pixel
        elif 'WFC3' in instrument and 'IR' in instrument:
            return 0.13  # HST WFC3/IR: ~0.13 arcsec/pixel
        elif 'NIRCAM' in instrument:
            return 0.031  # JWST NIRCam short: ~0.031 arcsec/pixel
        elif 'MIRI' in instrument:
            return 0.11  # JWST MIRI: ~0.11 arcsec/pixel
        
        # Default fallback
        warnings.warn(
            "Could not determine pixel scale from header. Using default 0.05 arcsec/pixel",
            UserWarning
        )
        return 0.05
    
    def list_extensions(self, filepath: Union[str, Path]) -> List[Dict]:
        """
        List all extensions in a FITS file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to FITS file
        
        Returns
        -------
        extensions : list of dict
            Information about each extension
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"FITS file not found: {filepath}")
        
        extensions = []
        
        with fits.open(filepath) as hdul:
            for i, hdu in enumerate(hdul):
                ext_info = {
                    'index': i,
                    'name': hdu.name,
                    'type': type(hdu).__name__,
                    'shape': hdu.data.shape if hdu.data is not None else None,
                    'dtype': hdu.data.dtype if hdu.data is not None else None
                }
                extensions.append(ext_info)
        
        return extensions


class PSFModel:
    """
    Point Spread Function (PSF) model for telescopes.
    
    The PSF describes how a point source appears in the image,
    including effects of:
    - Telescope optics
    - Atmospheric seeing (ground-based)
    - Detector properties
    - Diffraction
    
    Examples
    --------
    >>> psf = PSFModel(fwhm=0.1, pixel_scale=0.05)
    >>> psf_image = psf.generate_psf(size=25)
    >>> convolved = psf.convolve_image(data, psf_image)
    """
    
    def __init__(
        self,
        fwhm: float,
        pixel_scale: float,
        model_type: str = 'gaussian'
    ):
        """
        Initialize PSF model.
        
        Parameters
        ----------
        fwhm : float
            Full Width at Half Maximum in arcseconds
        pixel_scale : float
            Pixel scale in arcseconds/pixel
        model_type : str
            PSF model type: 'gaussian', 'moffat', or 'airy'
            (currently only 'gaussian' implemented)
        """
        self.fwhm = fwhm
        self.pixel_scale = pixel_scale
        self.model_type = model_type
        
        # Convert FWHM to sigma (for Gaussian)
        # FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma
        self.sigma_pixels = (fwhm / pixel_scale) / 2.355
    
    def generate_psf(self, size: int = 25) -> np.ndarray:
        """
        Generate PSF kernel.
        
        Parameters
        ----------
        size : int
            Size of PSF kernel (will be size × size pixels)
            Should be odd number for symmetry
        
        Returns
        -------
        psf : np.ndarray
            Normalized PSF kernel
        """
        if size % 2 == 0:
            size += 1  # Make odd
        
        center = size // 2
        y, x = np.ogrid[-center:center+1, -center:center+1]
        r = np.sqrt(x**2 + y**2)
        
        if self.model_type == 'gaussian':
            # Gaussian PSF
            r2 = x**2 + y**2
            psf = np.exp(-r2 / (2 * self.sigma_pixels**2))
            
        elif self.model_type == 'airy':
            # Airy disk PSF (diffraction-limited circular aperture)
            # PSF(r) = [2 J₁(x) / x]²  where x = π D r / (λ F)
            # For HST: D ≈ 2.4m, F ≈ 24 (f/24), λ ≈ 0.5μm
            # First zero at radius ≈ 1.22 λ F / D
            
            # Calculate Airy radius from FWHM
            # FWHM_airy ≈ 1.028 × λ F / D for the Airy disk
            # Airy radius (first zero) ≈ 1.22 λ F / D
            airy_radius_pixels = (self.fwhm / self.pixel_scale) / 1.028
            
            # Compute dimensionless radius x
            # Avoid division by zero at center
            x = np.where(r > 0, 
                        np.pi * r / airy_radius_pixels,
                        1e-10)
            
            # Airy function: [2 J₁(x) / x]²
            # Use scipy.special.j1 for Bessel function of first kind
            from scipy.special import j1
            psf = np.where(r > 0,
                          (2 * j1(x) / x)**2,
                          1.0)  # Central value = 1.0
            
        elif self.model_type == 'moffat':
            # Moffat PSF (better for ground-based seeing)
            # PSF(r) = [1 + (r/α)²]⁻ᵝ
            # where α is scale radius and β is power index
            # β = 4.765 gives good fit to atmospheric seeing
            
            beta = 4.765  # Typical atmospheric seeing parameter
            
            # Relate FWHM to α: FWHM = 2α√(2^(1/β) - 1)
            # Solving: α = FWHM / (2√(2^(1/β) - 1))
            alpha_pixels = (self.fwhm / self.pixel_scale) / (2 * np.sqrt(2**(1/beta) - 1))
            
            # Moffat profile
            psf = (1 + (r / alpha_pixels)**2)**(-beta)
            
        else:
            raise NotImplementedError(
                f"PSF model '{self.model_type}' not yet implemented. "
                f"Supported models: 'gaussian', 'airy', 'moffat'"
            )
        
        # Normalize
        psf = psf / np.sum(psf)
        
        return psf
    
    def convolve_image(
        self,
        image: np.ndarray,
        psf: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convolve image with PSF.
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        psf : np.ndarray, optional
            PSF kernel (if None, will generate one)
        
        Returns
        -------
        convolved : np.ndarray
            PSF-convolved image
        """
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "SciPy required for convolution. "
                "Install with: pip install scipy"
            )
        
        if psf is None:
            psf = self.generate_psf()
        
        # Use scipy's convolve or fftconvolve for efficiency
        from scipy.signal import fftconvolve
        
        convolved = fftconvolve(image, psf, mode='same')
        
        return convolved


def preprocess_real_data(
    data: np.ndarray,
    metadata: ObservationMetadata,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
    handle_nans: str = 'zero'
) -> np.ndarray:
    """
    Preprocess real observational data for lensing analysis.
    
    Steps:
    1. Handle NaN/inf values
    2. Resize if needed
    3. Normalize
    4. Remove background
    
    Parameters
    ----------
    data : np.ndarray
        Raw image data
    metadata : ObservationMetadata
        Image metadata
    target_size : tuple of int, optional
        Target (height, width) for resizing
    normalize : bool
        Whether to normalize to [0, 1]
    handle_nans : str
        How to handle NaNs: 'zero', 'median', or 'interpolate'
    
    Returns
    -------
    processed : np.ndarray
        Preprocessed image data
    """
    processed = data.copy()
    
    # Step 1: Handle NaN/inf values
    mask_invalid = ~np.isfinite(processed)
    
    if np.any(mask_invalid):
        if handle_nans == 'zero':
            processed[mask_invalid] = 0.0
        elif handle_nans == 'median':
            valid_median = np.median(processed[np.isfinite(processed)])
            processed[mask_invalid] = valid_median
        elif handle_nans == 'interpolate':
            # Simple nearest-neighbor interpolation
            from scipy.ndimage import distance_transform_edt
            
            if SCIPY_AVAILABLE:
                indices = distance_transform_edt(
                    mask_invalid,
                    return_distances=False,
                    return_indices=True
                )
                processed = processed[tuple(indices)]
            else:
                processed[mask_invalid] = 0.0
    
    # Step 2: Resize if needed
    if target_size is not None:
        from scipy.ndimage import zoom
        
        if SCIPY_AVAILABLE:
            zoom_factors = (
                target_size[0] / processed.shape[0],
                target_size[1] / processed.shape[1]
            )
            processed = zoom(processed, zoom_factors, order=1)
        else:
            warnings.warn(
                "SciPy not available. Skipping resize.",
                UserWarning
            )
    
    # Step 3: Normalize
    if normalize:
        pmin = float(processed.min())
        pmax = float(processed.max())
        
        if pmax > pmin:
            processed = (processed - pmin) / (pmax - pmin)
        else:
            processed = np.zeros_like(processed)
    
    return processed


# Module-level convenience functions
def load_real_data(
    filepath: Union[str, Path],
    extension: int = 0,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True
) -> Tuple[np.ndarray, ObservationMetadata]:
    """
    Convenience function to load and preprocess real data.
    
    Parameters
    ----------
    filepath : str or Path
        Path to FITS file
    extension : int
        FITS extension to read
    target_size : tuple of int, optional
        Target size for resizing
    normalize : bool
        Whether to normalize
    
    Returns
    -------
    data : np.ndarray
        Preprocessed image data
    metadata : ObservationMetadata
        Image metadata
    """
    loader = FITSDataLoader()
    data, metadata = loader.load_fits(filepath, extension=extension)
    
    data = preprocess_real_data(
        data,
        metadata,
        target_size=target_size,
        normalize=normalize
    )
    
    return data, metadata
