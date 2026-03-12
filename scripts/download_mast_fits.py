#!/usr/bin/env python3
"""
MAST Data Downloader Script

Usage:
    python download_mast_data.py
    
Requires:
    pip install astroquery
    
Note: You'll need to authenticate with MAST for some data.
"""

import os
import sys
from pathlib import Path

try:
    from astroquery.mast import Observations
    import astropy.units as u
except ImportError:
    print("ERROR: astroquery not installed. Run: pip install astroquery")
    sys.exit(1)

DATA_DIR = Path("data/real_data/mast_fits")
DATA_DIR.mkdir(parents=True, exist_ok=True)

MAST_TARGETS = {
    "SDSS_J1004+4112": {"ra": 10.0425, "dec": 41.1928},
    "Q0957+561": {"ra": 9.9586, "dec": 55.9503},
    "Q2237+0305": {"ra": 9.9745, "dec": 3.2583},
    "RXJ_1131-1231": {"ra": 11.3132, "dec": -12.3106},
    "HE0435-1223": {"ra": 4.3857, "dec": -12.3989},
    "B1608+656": {"ra": 16.0873, "dec": 65.5429},
    "PG_1115+080": {"ra": 11.3961, "dec": 7.8706},
    "B1422+231": {"ra": 14.5421, "dec": 22.9233},
}

print("=" * 60)
print("MAST HST DATA DOWNLOADER")
print("=" * 60)

# Try to download without authentication first
for name, pos in MAST_TARGETS.items():
    print(f"\nQuerying {name}...")
    try:
        # Query for ACS/WFC3 observations
        result = Observations.query_region(
            f"{pos['ra']} {pos['dec']}",
            radius="0.1 deg",
            filters=["F814W", "F606W", "F435W", "F110W", "F160W"]
        )
        print(f"  Found {len(result)} observations")
        
        # Try to download first 2 products
        if len(result) > 0:
            manifest = Observations.download_products(result[:2])
            print(f"  Downloaded: {len(manifest)} files")
    except Exception as e:
        print(f"  Note: {str(e)[:80]}")
        print(f"  → Manual download may be required from: https://mast.stsci.edu/")

print("\n" + "=" * 60)
print("MAST download complete. Check data/real_data/mast_fits/")
print("=" * 60)
