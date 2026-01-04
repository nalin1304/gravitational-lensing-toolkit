"""
Application Settings - Configure application preferences and parameters

Centralized settings for model paths, computation preferences, visualization
options, and application behavior.
"""

import streamlit as st
import sys
from pathlib import Path
import json
import torch

project_root = Path(__file__).parent.parent.parent

# Import utilities
try:
    from app.utils.ui import render_header, inject_custom_css, show_success, show_error, show_info
except ImportError:
    from utils.ui import render_header, inject_custom_css, show_success, show_error, show_info

# Configure page
st.set_page_config(
    page_title="Settings - Gravitational Lensing Platform",
    page_icon="⚙️",
    layout="wide"
)

# Apply custom CSS
inject_custom_css()


def load_settings():
    """Load settings from config file."""
    config_path = project_root / "config" / "app_settings.json"
    
    default_settings = {
        "computation": {
            "device": "cpu",
            "num_threads": 4,
            "cache_enabled": True
        },
        "visualization": {
            "default_colormap": "viridis",
            "figure_dpi": 100,
            "plot_style": "default"
        },
        "paths": {
            "model_dir": "results",
            "data_dir": "data",
            "output_dir": "results"
        },
        "defaults": {
            "grid_size": 64,
            "batch_size": 32,
            "learning_rate": 0.001
        }
    }
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                settings = json.load(f)
            return settings
        except (json.JSONDecodeError, IOError, KeyError) as e:
            return default_settings
    else:
        return default_settings


def save_settings(settings):
    """Save settings to config file."""
    config_dir = project_root / "config"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "app_settings.json"
    
    with open(config_path, 'w') as f:
        json.dump(settings, f, indent=2)
    
    return True


def main():
    """Main page function."""
    render_header(
        "⚙️ Application Settings",
        "Configure preferences and default parameters",
        "Configuration"
    )
    
    st.markdown("""
    Customize the application behavior, computation settings, and visualization preferences.
    Settings are saved automatically and persist across sessions.
    """)
    
    # Load current settings
    if 'app_settings' not in st.session_state:
        st.session_state['app_settings'] = load_settings()
    
    settings = st.session_state['app_settings']
    
    # Tabs for different setting categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🖥️ Computation",
        "🎨 Visualization", 
        "📁 Paths",
        "🔧 Defaults",
        "ℹ️ System Info"
    ])
    
    # Computation settings
    with tab1:
        st.subheader("Computation Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            device_options = ["cpu"]
            if torch.cuda.is_available():
                device_options.append("cuda")
            
            device = st.selectbox(
                "Device",
                device_options,
                index=device_options.index(settings['computation']['device']),
                help="Computation device for PyTorch operations"
            )
            settings['computation']['device'] = device
            
            if device == "cuda":
                st.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
                st.metric("CUDA Version", torch.version.cuda)
        
        with col2:
            num_threads = st.slider(
                "CPU Threads",
                min_value=1,
                max_value=16,
                value=settings['computation']['num_threads'],
                help="Number of threads for CPU operations"
            )
            settings['computation']['num_threads'] = num_threads
            
            cache_enabled = st.checkbox(
                "Enable Caching",
                value=settings['computation']['cache_enabled'],
                help="Cache intermediate results for faster computation"
            )
            settings['computation']['cache_enabled'] = cache_enabled
        
        st.markdown("---")
        st.markdown("**Memory Management**")
        
        col_mem1, col_mem2 = st.columns(2)
        
        with col_mem1:
            if st.button("🧹 Clear Session Cache"):
                st.session_state.clear()
                st.session_state['app_settings'] = settings
                show_success("Session cache cleared!")
        
        with col_mem2:
            if torch.cuda.is_available():
                if st.button("🗑️ Clear GPU Memory"):
                    torch.cuda.empty_cache()
                    show_success("GPU memory cleared!")
    
    # Visualization settings
    with tab2:
        st.subheader("Visualization Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            colormap = st.selectbox(
                "Default Colormap",
                ["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "RdBu_r"],
                index=["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "RdBu_r"].index(
                    settings['visualization']['default_colormap']
                ),
                help="Default colormap for plots"
            )
            settings['visualization']['default_colormap'] = colormap
            
            # Colormap preview
            import numpy as np
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(6, 1))
            gradient = np.linspace(0, 1, 256).reshape(1, -1)
            ax.imshow(gradient, aspect='auto', cmap=colormap)
            ax.set_axis_off()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            dpi = st.slider(
                "Figure DPI",
                min_value=72,
                max_value=300,
                value=settings['visualization']['figure_dpi'],
                step=1,
                help="Resolution for saved figures"
            )
            settings['visualization']['figure_dpi'] = dpi
            
            plot_style = st.selectbox(
                "Plot Style",
                ["default", "seaborn", "ggplot", "bmh", "dark_background"],
                index=["default", "seaborn", "ggplot", "bmh", "dark_background"].index(
                    settings['visualization']['plot_style']
                ),
                help="Matplotlib plot style"
            )
            settings['visualization']['plot_style'] = plot_style
    
    # Path settings
    with tab3:
        st.subheader("Directory Paths")
        
        st.markdown("Configure paths for models, data, and outputs.")
        
        model_dir = st.text_input(
            "Model Directory",
            value=settings['paths']['model_dir'],
            help="Directory for saved models"
        )
        settings['paths']['model_dir'] = model_dir
        
        data_dir = st.text_input(
            "Data Directory",
            value=settings['paths']['data_dir'],
            help="Directory for input data"
        )
        settings['paths']['data_dir'] = data_dir
        
        output_dir = st.text_input(
            "Output Directory",
            value=settings['paths']['output_dir'],
            help="Directory for results and outputs"
        )
        settings['paths']['output_dir'] = output_dir
        
        # Verify paths
        st.markdown("---")
        st.markdown("**Path Verification**")
        
        paths_valid = True
        for name, path in [("Model", model_dir), ("Data", data_dir), ("Output", output_dir)]:
            full_path = project_root / path
            if full_path.exists():
                st.success(f"✅ {name}: `{path}` exists")
            else:
                st.warning(f"⚠️ {name}: `{path}` does not exist")
                if st.button(f"Create {name} Directory", key=f"create_{name}"):
                    full_path.mkdir(parents=True, exist_ok=True)
                    st.rerun()
    
    # Default parameters
    with tab4:
        st.subheader("Default Parameters")
        
        st.markdown("Set default values for common parameters across the application.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            grid_size = st.select_slider(
                "Default Grid Size",
                options=[32, 64, 128, 256],
                value=settings['defaults']['grid_size'],
                help="Default resolution for convergence maps"
            )
            settings['defaults']['grid_size'] = grid_size
        
        with col2:
            batch_size = st.select_slider(
                "Default Batch Size",
                options=[8, 16, 32, 64, 128],
                value=settings['defaults']['batch_size'],
                help="Default batch size for training"
            )
            settings['defaults']['batch_size'] = batch_size
        
        with col3:
            learning_rate = st.select_slider(
                "Default Learning Rate",
                options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                value=settings['defaults']['learning_rate'],
                format_func=lambda x: f"{x:.0e}",
                help="Default learning rate for training"
            )
            settings['defaults']['learning_rate'] = learning_rate
    
    # System information
    with tab5:
        st.subheader("System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Python Environment**")
            st.text(f"Python: {sys.version.split()[0]}")
            
            try:
                import numpy as np
                st.text(f"NumPy: {np.__version__}")
            except ImportError:
                st.text("NumPy: Not available")
            
            try:
                st.text(f"PyTorch: {torch.__version__}")
            except ImportError:
                st.text("PyTorch: Not available")
            
            try:
                import streamlit as st_ver
                st.text(f"Streamlit: {st_ver.__version__}")
            except ImportError:
                st.text("Streamlit: Not available")
        
        with col2:
            st.markdown("**Hardware**")
            
            import platform
            st.text(f"System: {platform.system()}")
            st.text(f"Architecture: {platform.machine()}")
            
            if torch.cuda.is_available():
                st.text(f"GPU: {torch.cuda.get_device_name(0)}")
                st.text(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                st.text("GPU: Not available")
        
        st.markdown("---")
        st.markdown("**Project Information**")
        
        st.text(f"Project Root: {project_root}")
        st.text(f"Config Path: {project_root / 'config' / 'app_settings.json'}")
        
        # Check module availability
        st.markdown("---")
        st.markdown("**Module Availability**")
        
        modules_to_check = [
            ("Core Lensing", "src.lens_models"),
            ("PINN", "src.ml.pinn"),
            ("Validation", "src.validation"),
            ("Uncertainty", "src.ml.uncertainty"),
            ("Multi-Plane", "src.lens_models.multi_plane"),
            ("Real Data", "src.data.real_data_loader")
        ]
        
        for name, module in modules_to_check:
            try:
                __import__(module)
                st.success(f"✅ {name}")
            except ImportError:
                st.error(f"❌ {name}")
    
    # Save settings
    st.markdown("---")
    
    col_save1, col_save2, col_save3 = st.columns([1, 1, 2])
    
    with col_save1:
        if st.button("💾 Save Settings", type="primary", use_container_width=True):
            try:
                save_settings(settings)
                st.session_state['app_settings'] = settings
                show_success("Settings saved successfully!")
            except Exception as e:
                show_error(f"Error saving settings: {e}")
    
    with col_save2:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            default_settings = load_settings()
            st.session_state['app_settings'] = default_settings
            st.rerun()
    
    with col_save3:
        st.info("Settings are saved to `config/app_settings.json`")


if __name__ == "__main__":
    main()
