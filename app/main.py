"""
⚠️ DEPRECATION NOTICE ⚠️

This file (app/main.py) has been DEPRECATED as of November 5, 2025.

The application has been refactored into a multi-page structure for better
maintainability, security, and scalability.

NEW LAUNCH COMMAND:
    streamlit run app/Home.py

MIGRATION GUIDE:
- Old monolithic structure (3,142 lines): app/main.py ❌
- New multi-page structure: app/Home.py + app/pages/*.py ✅

WHY THIS CHANGE?
1. Better Code Organization: Each feature in its own file
2. Easier Maintenance: Smaller, focused modules
3. Team Collaboration: Multiple developers can work simultaneously
4. Performance: Streamlit loads only active pages
5. Security: Isolated page logic prevents cross-contamination

DIRECTORY STRUCTURE:
app/
├── Home.py                    # Main entry point (NEW)
├── main.py                    # DEPRECATED - DO NOT USE
├── main_legacy.py             # Backup of original implementation
├── styles.py                  # Shared styling
├── error_handler.py           # Shared error handling
├── utils/                     # Shared utilities
│   └── session_state.py
└── pages/                     # Individual feature pages
    ├── 01_Home.py             # Landing page (used by Home.py)
    ├── 02_Simple_Lensing.py   # Basic lensing demo
    ├── 03_PINN_Inference.py   # Neural network inference
    ├── 04_Multi_Plane.py      # Multi-plane lensing
    ├── 05_Real_Data.py        # FITS file analysis
    ├── 06_Training.py         # Model training
    ├── 07_Validation.py       # Scientific validation
    ├── 08_Bayesian_UQ.py      # Uncertainty quantification
    └── 09_Settings.py         # Configuration

WHAT TO DO IF YOU SEE THIS MESSAGE:
If you're running:
    streamlit run app/main.py

Change to:
    streamlit run app/Home.py

If you have scripts or documentation referencing app/main.py, update them.

TIMELINE:
- Nov 5, 2025: Multi-page structure created, main.py deprecated
- Dec 1, 2025: main.py will be renamed to main_legacy.py
- Jan 1, 2026: main_legacy.py may be removed

For questions, see:
- README.md
- MONOLITH_REFACTOR_GUIDE.md
- docs/ARCHITECTURE.md

This file is kept for backward compatibility but SHOULD NOT BE USED.
All new development should use the multi-page structure.
"""

import streamlit as st
import sys
from pathlib import Path

st.set_page_config(
    page_title="⚠️ Deprecated",
    page_icon="⚠️",
    layout="wide"
)

st.error("""
# ⚠️ DEPRECATION WARNING

This entry point (`app/main.py`) has been **DEPRECATED**.

## Use Instead:
```bash
streamlit run app/Home.py
```

## Why?
The application has been refactored into a **multi-page structure** for:
- ✅ Better maintainability (smaller files)
- ✅ Improved security (isolated pages)
- ✅ Easier collaboration (multiple devs)
- ✅ Better performance (lazy loading)

## What to Do?
1. Stop the current Streamlit server (Ctrl+C)
2. Run: `streamlit run app/Home.py`
3. Update any scripts/bookmarks

See `MONOLITH_REFACTOR_GUIDE.md` for details.
""")

st.stop()
