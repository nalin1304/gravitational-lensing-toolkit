import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

from styles import inject_custom_css, render_header, render_card
from utils.session_state import init_session_state
from src.ml.generate_dataset import generate_synthetic_convergence

# Page config
st.set_page_config(
    page_title="Simple Lensing Simulation",
    page_icon="🔭",
    layout="wide"
)

init_session_state()
inject_custom_css()

render_header(
    "Simple Lensing Simulation",
    "Generate synthetic gravitational lenses using NFW profiles",
    "✨ Interactive"
)

# Main layout
col_sidebar, col_main = st.columns([1, 3])

with col_sidebar:
    st.markdown("### 🛠️ Configuration")
    st.markdown("Customize your lens parameters below.")
    
    with st.expander("Lens Properties", expanded=True):
        profile_type = st.selectbox(
            "Mass Profile",
            ["NFW", "Elliptical NFW"],
            help="Navarro-Frenk-White profile is standard for dark matter halos."
        )
        
        mass = st.slider(
            "Virial Mass (10^12 M_sun)",
            min_value=0.1, max_value=10.0, value=2.0, step=0.1,
            format="%.1f"
        ) * 1e12
        
        scale_radius = st.slider(
            "Scale Radius (kpc)", 
            50.0, 500.0, 200.0, 10.0
        )
        
        ellipticity = 0.0
        if profile_type == "Elliptical NFW":
            ellipticity = st.slider("Ellipticity", 0.0, 0.5, 0.3, 0.05)
            
    with st.expander("Simulation Settings"):
        grid_size = st.select_slider("Grid Resolution", options=[32, 64, 128], value=64)
        noise_level = st.slider("Noise Level", 0.0, 0.1, 0.01, 0.001)

    generate_btn = st.button("Generate Lens", type="primary", use_container_width=True)

with col_main:
    if generate_btn:
        with st.spinner("Simulating light rays through dark matter..."):
            try:
                start_time = time.time()
                
                # Call backend generation function (wrapper)
                # Note: 'ellipticity' param is now correctly handled by backend fix
                convergence_map, X, Y = generate_synthetic_convergence(
                    profile_type=profile_type,
                    mass=mass,
                    scale_radius=scale_radius,
                    ellipticity=ellipticity,
                    grid_size=grid_size
                )
                
                # Add noise
                if noise_level > 0:
                    noise = np.random.normal(0, noise_level, convergence_map.shape)
                    convergence_map += noise
                
                duration = time.time() - start_time
                
                # Store in session state
                st.session_state['last_map'] = convergence_map
                st.session_state['last_params'] = {
                    "type": profile_type,
                    "mass": mass,
                    "r_s": scale_radius,
                    "ellipticity": ellipticity
                }
                
                st.success(f"Simulation complete in {duration*1000:.1f}ms")
                
            except Exception as e:
                st.error(f"Simulation failed: {str(e)}")
                st.session_state['last_map'] = None

    # Visualization Area
    if st.session_state.get('last_map') is not None:
        map_data = st.session_state['last_map']
        
        # Determine plot layout
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Style the plot to match theme
        plt.style.use('dark_background')
        ax.set_facecolor('#0f172a')
        fig.patch.set_facecolor('#0f172a')
        
        im = ax.imshow(
            map_data, 
            origin='lower', 
            cmap='inferno',
            extent=[-3, 3, -3, 3]
        )
        plt.colorbar(im, label="Convergence (κ)")
        ax.set_title(f"Generated {st.session_state['last_params']['type']} Lens")
        ax.set_xlabel("Arcseconds")
        ax.set_ylabel("Arcseconds")
        ax.grid(False)
        
        st.pyplot(fig)
        
        # Download button
        st.download_button(
            "Download Map (Numpy)",
            data=map_data.tobytes(),
            file_name="convergence_map.npy",
            mime="application/octet-stream"
        )
        
    else:
        # Placeholder state
        st.info("👈 Configure parameters and click 'Generate Lens' to start.")
        
        # Feature cards below placeholder
        col_a, col_b = st.columns(2)
        with col_a:
            render_card(
                "NFW Profile",
                "Models the density distribution of dark matter halos predicted by CDM simulations.",
                "🌌"
            )
        with col_b:
            render_card(
                "Ellipticity",
                "Most real halos are triaxial. Ellipticity adds realistic shear to the lensing potential.",
                "🥚"
            )
