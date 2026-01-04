"""
PINN Inference - Physics-Informed Neural Network Predictions

Run the trained PINN model on convergence maps to predict lens parameters 
and classify dark matter model type.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import sys
import os

# Add project root to path (needed for src imports)
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from styles import inject_custom_css, render_header, render_card
from utils.session_state import init_session_state

# Configure page
st.set_page_config(
    page_title="PINN Inference - Gravitational Lensing Platform",
    page_icon="🔬",
    layout="wide"
)

init_session_state()
inject_custom_css()

# Import core modules
PINN_AVAILABLE = True
PhysicsInformedNN = None

try:
    from src.ml.pinn import PhysicsInformedNN
except ImportError as e:
    PINN_AVAILABLE = False
    st.error(f"❌ PINN module not available: {e}")

# plotting helper
def plot_classification_probs(class_names, probs, entropy):
    """Plot classification probabilities with custom style."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Dark theme style
    plt.style.use('dark_background')
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    
    colors = ['#6366f1', '#ec4899', '#10b981'] # Primary theme colors
    bars = ax.bar(class_names, probs, color=colors, alpha=0.9)
    
    ax.set_ylabel('Probability', fontsize=10, color='white')
    ax.set_title('Dark Matter Model Probability', fontsize=12, fontweight='bold', color='white')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.1, linestyle='--')
    
    # Hide spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Add values
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{prob*100:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='white')
    
    return fig

def load_pretrained_model(model_path=None):
    """Load pre-trained PINN model."""
    if model_path is None:
        env_path = os.environ.get('PINN_MODEL_PATH')
        if env_path:
            model_path = Path(env_path)
        else:
            model_path = project_root / "results" / "pinn_model_best.pth"
    
    if not isinstance(model_path, Path):
         model_path = Path(model_path)

    if not model_path.exists():
        return None, f"Model file not found at: {model_path}"
    
    try:
        model = PhysicsInformedNN(input_size=64, dropout_rate=0.2)
        
        # Load weights
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        except TypeError:
            checkpoint = torch.load(model_path, map_location='cpu')
            
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, "Model loaded successfully!"
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def create_demo_model():
    """Create a demo PINN model with random weights."""
    model = PhysicsInformedNN(input_size=64, dropout_rate=0.2)
    model.eval()
    return model

def main():
    render_header(
        "PINN Inference",
        "Physics-Informed Neural Network prediction & classification",
        "🧠 AI Core"
    )

    if not PINN_AVAILABLE:
        st.stop()

    # Layout: Sidebar for settings, Main for IO
    col_sidebar, col_main = st.columns([1, 3])

    with col_sidebar:
        st.markdown("### ⚙️ Model Settings")
        
        model_type = st.radio(
            "Model Source",
            ["Pre-trained", "Demo (Random)", "Custom Path"],
            help="Choose 'Demo' if you don't have a trained model file."
        )

        custom_path = None
        if model_type == "Custom Path":
            custom_path = st.text_input("Path to .pth file", "results/pinn_model_best.pth")

        device = st.selectbox(
            "Compute Device",
            ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        )

        if st.button("Load Model", type="primary", use_container_width=True):
            with st.spinner("Initializing neural network..."):
                try:
                    if model_type == "Demo (Random)":
                        model = create_demo_model().to(device)
                        st.session_state['pinn_model'] = model
                        st.session_state['model_device'] = device
                        st.session_state['is_demo'] = True
                        st.toast("✅ Demo model loaded!", icon="🧪")
                    else:
                        path_to_use = custom_path if model_type == "Custom Path" else None
                        model, msg = load_pretrained_model(path_to_use)
                        
                        if model:
                            model = model.to(device)
                            st.session_state['pinn_model'] = model
                            st.session_state['model_device'] = device
                            st.session_state['is_demo'] = False
                            st.toast(f"✅ {msg}", icon="💾")
                        else:
                            st.error(msg)
                except Exception as e:
                    st.error(f"Failed to load model: {e}")

        # Status Indicator
        if 'pinn_model' in st.session_state:
            st.success(f"Active: {model_type} on {st.session_state['model_device']}")
        else:
            st.warning("No model loaded")

    with col_main:
        # Input Data Section
        st.markdown("### 📥 Input Data")
        
        tabs = st.tabs(["From Session", "Upload File", "Generate Random"])
        
        input_data = None
        
        with tabs[0]:
            if 'last_map' in st.session_state:
                input_data = st.session_state['last_map']
                st.info(f"Using map from Simple Lensing simulation ({input_data.shape})")
            else:
                st.info("No data in session. generate one in Simple Lensing page.")
        
        with tabs[1]:
            uploaded = st.file_uploader("Upload .npy", type=['npy'])
            if uploaded:
                try:
                    input_data = np.load(uploaded)
                    st.success("File loaded successfully")
                except Exception as e:
                    st.error(f"Invalid file: {e}")
        
        with tabs[2]:
            if st.button("Generate Random Noise (Test)"):
                input_data = np.random.randn(64, 64) * 0.1 + 0.5
                input_data = np.clip(input_data, 0, 1)
                st.session_state['last_map'] = input_data # Cache it
                st.success("Generated random test map")
            elif 'last_map' in st.session_state:
                 # Allow falling back to session map if tab selected but not clicked
                 pass

        # Visualization of Input
        if input_data is not None:
             col_vis1, col_vis2 = st.columns([1, 2])
             with col_vis1:
                 fig, ax = plt.subplots(figsize=(5, 5))
                 ax.imshow(input_data, cmap='inferno', origin='lower')
                 ax.axis('off')
                 st.pyplot(fig)
             
             with col_vis2:
                 st.markdown("#### Ready for Inference")
                 st.markdown(f"**Shape:** {input_data.shape}")
                 st.markdown(f"**Mean Intensity:** {np.mean(input_data):.3f}")
                 
                 predict_btn = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

                 if predict_btn:
                     if 'pinn_model' not in st.session_state:
                         st.error("Please load a model first (see sidebar).")
                     else:
                        with st.spinner("Processing through Physics-Informed NN..."):
                            try:
                                # Preprocess
                                img = input_data
                                if img.shape != (64, 64):
                                    from scipy.ndimage import zoom
                                    scale = 64 / img.shape[0]
                                    img = zoom(img, scale, order=1)
                                
                                # Normalize
                                img = (img - img.min()) / (img.max() - img.min() + 1e-10)
                                tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
                                tensor = tensor.to(st.session_state['model_device'])
                                
                                # Infer
                                model = st.session_state['pinn_model']
                                with torch.no_grad():
                                    params, classes = model(tensor)
                                
                                params = params.cpu().numpy()[0]
                                probs = torch.softmax(classes, dim=1).cpu().numpy()[0]
                                
                                # Display Results
                                st.markdown("---")
                                st.subheader("📊 Results")
                                
                                res_col1, res_col2 = st.columns(2)
                                
                                with res_col1:
                                    render_card("Predicted Parameters", 
                                        f"""
                                        **Virial Mass:** {params[0]:.2e} M☉<br>
                                        **Scale Radius:** {params[1]:.1f} kpc<br>
                                        **Source X:** {params[2]:.2f}"<br>
                                        **Source Y:** {params[3]:.2f}"<br>
                                        **H₀:** {params[4]:.1f} km/s/Mpc
                                        """, "📐")
                                
                                with res_col2:
                                    # Entroy
                                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                                    fig_cls = plot_classification_probs(['CDM', 'WDM', 'SIDM'], probs, entropy)
                                    st.pyplot(fig_cls)

                            except Exception as e:
                                st.error(f"Inference failed: {e}")
                                st.exception(e)

if __name__ == "__main__":
    main()
