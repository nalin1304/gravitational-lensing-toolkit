#!/usr/bin/env python3
"""
JWST Data Downloader

Usage:
    python download_jwst_data.py
    
Data available at:
    - JWST Archive: https://jwst.stsci.edu/
    - MAST: https://mast.stsci.edu/
"""

# For JWST data, use the Space Telescope Science Institute archive
# Direct download links require authentication

JWST_TARGETS = {
    "smacs_0723": {"ra": 110.8167, "dec": -73.4500},
    "smacs_0329": {"ra": 52.3500, "dec": -40.8500},
    "abell_2744": {"ra": 3.5875, "dec": -30.3883},
}

print("JWST Data Download")
print("=" * 60)
print("To download JWST data:")
print("1. Go to: https://jwst.stsci.edu/")
print("2. Search for target coordinates")
print("3. Download FITS files to: data/real_data/jwst_fits/")
print()
print("Alternatively, use astroquery:")
print("  from astroquery.mast import Observations")
print("  result = Observations.query_region('110.8 -73.4', radius='0.5 deg')")
