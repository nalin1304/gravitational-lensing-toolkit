"""
Real Data Analysis - Upload and analyze real FITS astronomical data

Process and analyze real gravitational lensing observations from HST, JWST, 
and other telescopes.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import tempfile

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from styles import inject_custom_css, render_header, render_card
from utils.session_state import init_session_state

# Configure page
st.set_page_config(
    page_title="Real Data Analysis - Gravitational Lensing Platform",
    page_icon="🔭",
    layout="wide"
)

init_session_state()
inject_custom_css()

# Check for astropy
ASTROPY_AVAILABLE = False
astropy_error = None
try:
    import astropy
    from astropy.io import fits
    from astropy.visualization import ZScaleInterval, ImageNormalize
    ASTROPY_AVAILABLE = True
except ImportError as e:
    astropy_error = str(e)


def plot_fits_image(data, title="FITS Image"):
    """Plot FITS image data with dark theme."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Dark Theme
    plt.style.use('dark_background')
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    
    # Scale
    if ASTROPY_AVAILABLE:
        interval = ZScaleInterval()
        try:
            vmin, vmax = interval.get_limits(data)
        except:
             # Fallback
             vmin, vmax = np.percentile(data, [1, 99])
    else:
        vmin, vmax = np.percentile(data, [1, 99])

    im = ax.imshow(data, cmap='inferno', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12, fontweight='bold', color='white')
    ax.axis('off')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    plt.tight_layout()
    return fig

def main():
    render_header(
        "Real Data Analysis",
        "Upload and analyze FITS astronomical data",
        "🔭 Observation"
    )

    if not ASTROPY_AVAILABLE:
        st.error(f"❌ Astropy not installed: {astropy_error}")
        return

    col_upload, col_preview = st.columns([1, 2])

    with col_upload:
        st.markdown("### 📂 Data Import")
        uploaded_file = st.file_uploader("Upload FITS File", type=['fits', 'fit', 'fts'])
        
        if uploaded_file:
            with st.spinner("Parsing FITS..."):
                try:
                     # Save temp
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.fits') as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    with fits.open(tmp_path) as hdul:
                        st.info(f"HDUs: {len(hdul)}")
                        
                        # Auto-detect image
                        primary = None
                        for i, hdu in enumerate(hdul):
                            if hasattr(hdu, 'data') and hdu.data is not None:
                                if len(hdu.data.shape) >= 2:
                                    primary = hdu
                                    st.success(f"Selected HDU {i} ({hdu.name})")
                                    break
                        
                        if primary:
                            data = primary.data
                            header = primary.header
                             # Handle 3D
                            if len(data.shape) == 3:
                                data = data[0]
                                st.warning("3D data detected, using first slice.")
                            
                            st.session_state['fits_data'] = data.astype(float)
                            st.session_state['fits_header'] = dict(header)
                            st.session_state['fits_name'] = uploaded_file.name
                        else:
                            st.error("No image data found.")
                            
                except Exception as e:
                    st.error(f"Error: {e}")

        # Preprocessing Tab
        if 'fits_data' in st.session_state:
            st.markdown("### ⚙️ Processing")
            
            data = st.session_state['fits_data']
            
            crop = st.checkbox("Crop Center", value=True)
            crop_size = st.slider("Size (px)", 32, 512, 128, 32) if crop else 0
            
            normalize = st.checkbox("Normalize (0-1)", value=True)
            denoise = st.checkbox("Gaussian Smooth", value=False)
            sigma = 1.0
            if denoise:
                sigma = st.slider("Sigma", 0.5, 3.0, 1.0)
            
            if st.button("Apply Processing", type="primary"):
                processed = data.copy()
                
                # Crop
                if crop:
                    cy, cx = processed.shape[0]//2, processed.shape[1]//2
                    h = crop_size//2
                    processed = processed[cy-h:cy+h, cx-h:cx+h]
                
                # Denoise
                if denoise:
                    from scipy.ndimage import gaussian_filter
                    processed = gaussian_filter(processed, sigma=sigma)
                
                # Normalize
                if normalize:
                     processed = (processed - np.nanmin(processed)) / (np.nanmax(processed) - np.nanmin(processed) + 1e-10)
                
                st.session_state['proc_data'] = processed
                st.success("Processed!")

    with col_preview:
        if 'proc_data' in st.session_state:
             st.markdown("### 🖼️ Analysis View")
             
             data = st.session_state['proc_data']
             
             # Visualize
             fig = plot_fits_image(data, f"Processed: {st.session_state.get('fits_name','Image')}")
             st.pyplot(fig)
             
             # Stats
             c1, c2, c3 = st.columns(3)
             c1.metric("Min", f"{np.min(data):.2f}")
             c2.metric("Max", f"{np.max(data):.2f}")
             c3.metric("RMS", f"{np.std(data):.3f}")
             
             # Actions
             st.markdown("---")
             col_act1, col_act2 = st.columns(2)
             
             with col_act1:
                 # Download
                 import io
                 buf = io.BytesIO()
                 np.save(buf, data)
                 buf.seek(0)
                 st.download_button("💾 Download .npy", buf, "processed_lens.npy")
            
             with col_act2:
                 # Send to PINN
                 if st.button("🚀 Send to Inference"):
                     st.session_state['last_map'] = data # Compatible key with inference page
                     st.toast("Data sent to Inference Page!", icon="📨")

        elif 'fits_data' in st.session_state:
            st.info("👈 Apply processing to visualize result")
            data = st.session_state['fits_data']
            fig = plot_fits_image(data, "Raw FITS Data")
            st.pyplot(fig)
        else:
            render_card("Instructions", 
                """
                1. Upload a FITS file (HST/JWST/Simulated)
                2. Use the **Processing** tools to crop and normalize
                3. Send the result to **PINN Inference** for analysis
                """, "ℹ️")

if __name__ == "__main__":
    main()
