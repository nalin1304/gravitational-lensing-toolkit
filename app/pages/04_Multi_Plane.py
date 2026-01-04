"""
Multi-Plane Lensing - Advanced gravitational lensing with multiple lens planes

Simulate light propagation through multiple lens planes at different redshifts,
enabling realistic modeling of line-of-sight structure.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from styles import inject_custom_css, render_header, render_card
from utils.session_state import init_session_state

# Import core modules
from src.lens_models.multi_plane import LensPlane, MultiPlaneLens
from src.lens_models.mass_profiles import NFWProfile
from src.lens_models.lens_system import LensSystem

# Configure page
st.set_page_config(
    page_title="Multi-Plane Lensing - Gravitational Lensing Platform",
    page_icon="🌌",
    layout="wide"
)

init_session_state()
inject_custom_css()

def create_multi_plane_system(plane_configs, z_source):
    """Create multi-plane lens system from configurations."""
    planes = []
    
    for config in plane_configs:
        # Create lens system for this plane
        # LensSystem handles cosmology internally (H0=70, Om0=0.3 default)
        lens_system = LensSystem(
            z_lens=config['redshift'],
            z_source=z_source
        )
        
        # Create NFW profile with correct parameters
        profile = NFWProfile(
            M_vir=config['mass'],
            concentration=config['concentration'],
            lens_system=lens_system
        )

        plane = LensPlane(
            redshift=config['redshift'],
            mass_profile=profile,
            position=(config['x_offset'], config['y_offset'])
        )
        planes.append(plane)
    
    multi_lens = MultiPlaneLens(planes, z_source)
    return multi_lens


def plot_ray_tracing(source_grid, image_grid, title="Ray Tracing"):
    """Plot source and image plane grids with dark theme."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Dark theme
    plt.style.use('dark_background')
    for ax in [ax1, ax2]:
        ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    
    # Source plane
    ax1.scatter(source_grid[:, 0], source_grid[:, 1], 
               c='#3b82f6', alpha=0.6, s=15, label='Source')
    ax1.set_xlabel('β_x (arcsec)', color='white')
    ax1.set_ylabel('β_y (arcsec)', color='white')
    ax1.set_title('Source Plane', fontsize=12, fontweight='bold', color='white')
    ax1.grid(alpha=0.1, linestyle='--')
    ax1.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')
    ax1.set_aspect('equal')
    
    # Image plane
    ax2.scatter(image_grid[:, 0], image_grid[:, 1], 
               c='#ef4444', alpha=0.6, s=15, label='Image')
    ax2.set_xlabel('θ_x (arcsec)', color='white')
    ax2.set_ylabel('θ_y (arcsec)', color='white')
    ax2.set_title('Image Plane', fontsize=12, fontweight='bold', color='white')
    ax2.grid(alpha=0.1, linestyle='--')
    ax2.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')
    ax2.set_aspect('equal')
    
    # Hide spines
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#334155')
        ax.spines['left'].set_color('#334155')
        ax.tick_params(colors='#94a3b8')

    plt.suptitle(title, fontsize=14, fontweight='bold', color='white')
    plt.tight_layout()
    return fig


def main():
    render_header(
        "Multi-Plane Lensing",
        "Simulate complex lensing with multiple lens planes",
        "🌌 Advanced"
    )

    col_conf, col_vis = st.columns([1, 2])

    with col_conf:
        st.markdown("### ⚙️ Configuration")
        
        with st.expander("🌍 Global Parameters", expanded=True):
            z_source = st.slider("Source Redshift", 1.0, 5.0, 2.0, 0.1)
            n_planes = st.slider("Number of Planes", 1, 5, 2)
            grid_size = st.select_slider("Grid Resolution", [32, 64, 128], value=64)
            field_size = st.slider("Field Size (\")", 5.0, 30.0, 10.0, 1.0)
        
        st.markdown(f"**🔍 Lens Planes ({n_planes})**")
        plane_configs = []
        
        # Tabs for planes configuration to save space
        plane_tabs = st.tabs([f"Plane {i+1}" for i in range(n_planes)])
        
        for i, tab in enumerate(plane_tabs):
            with tab:
                st.markdown(f"**Plane {i+1} Settings**")
                
                # Dynamic defaults based on plane index
                default_z = 0.3 + i * (z_source - 0.5) / max(n_planes, 1)
                default_mass = 1.0 + i * 0.5
                
                z_lens = st.number_input(f"Redshift (z_{i+1})", 0.1, z_source-0.1, default_z, 0.05, key=f"z_{i}")
                mass = st.number_input(f"Mass (10^12 M☉)", 0.1, 10.0, default_mass, 0.5, key=f"m_{i}")
                conc = st.number_input(f"Concentration", 2.0, 15.0, 5.0, 0.5, key=f"c_{i}")
                
                col_off1, col_off2 = st.columns(2)
                with col_off1:
                    x_off = st.number_input(f"X Offset", -5.0, 5.0, 0.0, 0.5, key=f"x_{i}")
                with col_off2:
                    y_off = st.number_input(f"Y Offset", -5.0, 5.0, 0.0, 0.5, key=f"y_{i}")
                
                plane_configs.append({
                    'redshift': z_lens,
                    'mass': mass * 1e12,
                    'concentration': conc,
                    'x_offset': x_off,
                    'y_offset': y_off
                })

        st.markdown("---")
        if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
            with st.spinner("Ray tracing through multiple planes..."):
                try:
                    multi_lens = create_multi_plane_system(plane_configs, z_source)
                    
                    # Grids
                    extent = field_size / 2
                    x = np.linspace(-extent, extent, grid_size)
                    y = np.linspace(-extent, extent, grid_size)
                    X, Y = np.meshgrid(x, y)
                    image_pos = np.column_stack([X.flatten(), Y.flatten()])
                    
                    # Trace
                    source_pos = multi_lens.ray_trace(image_pos[:, 0], image_pos[:, 1])
                    
                    # Cache
                    st.session_state['mp_res'] = {
                        'source': source_pos,
                        'image': image_pos,
                        'planes': plane_configs,
                        'n': n_planes
                    }
                    st.success("Simulation Complete!")
                    
                except Exception as e:
                    st.error(f"Simulation failed: {e}")

    with col_vis:
        if 'mp_res' in st.session_state:
            res = st.session_state['mp_res']
            st.markdown("### 📊 Simulation Results")
            
            # Stats Cards
            sc1, sc2, sc3 = st.columns(3)
            
            src = res['source']
            img = res['image']
            
            # Simple magnification estimate (area ratio)
            src_area = np.ptp(src[:, 0]) * np.ptp(src[:, 1])
            img_area = np.ptp(img[:, 0]) * np.ptp(img[:, 1])
            mag = img_area / src_area if src_area > 0 else 0
            
            with sc1:
                st.metric("🔍 Magnification", f"{mag:.2f}×")
            with sc2:
                st.metric("📐 Deflection (RMS)", f"{np.std(src - img):.2f}\"")
            with sc3:
                st.metric("🌟 Source Redshift", f"z = {z_source}")
            
            # Plot
            fig = plot_ray_tracing(src, img, f"Ray Tracing ({res['n']} Planes)")
            st.pyplot(fig)
            
            with st.expander("📚 Theory: Multi-Plane Lensing"):
                st.markdown(r"""
                In multi-plane lensing, the light ray is deflected at each plane $i$:
                
                $$ \beta = \theta - \sum_{i=1}^{N} \frac{D_{is}}{D_s} \hat{\alpha}_i(\xi_i) $$
                
                Where $\hat{\alpha}_i$ is the physical deflection angle at plane $i$.
                This recursive equation enables modeling of line-of-sight halos and complex mass distributions.
                """)
        else:
            # Empty State
            st.info("👈 Configure lens planes and click 'Run Simulation' to visualize ray tracing.")
            
            render_card("Features", 
                """
                • **Full Ray Tracing**: Computes deflection through N planes<br>
                • **Recursive Algorithm**: Efficiently solves the lens equation<br>
                • **Cosmologically Consistent**: Uses Astropy for distances
                """, "🌟")

if __name__ == "__main__":
    main()
