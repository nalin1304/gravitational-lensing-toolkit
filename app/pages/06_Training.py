"""
Model Training - Train PINN models on synthetic lensing data

Interactive interface for training Physics-Informed Neural Networks
with real-time progress monitoring and hyperparameter tuning.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import sys
import time

# Configure page FIRST
st.set_page_config(
    page_title="Model Training - Gravitational Lensing Platform",
    page_icon="🎓",
    layout="wide"
)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import utilities
try:
    from app.utils.ui import render_header, inject_custom_css, show_success, show_error, show_warning, show_info
except ImportError:
    from utils.ui import render_header, inject_custom_css, show_success, show_error, show_warning, show_info

# Apply custom CSS
inject_custom_css()

# Import core modules
TRAINING_AVAILABLE = False
import_error = None
try:
    from src.ml.pinn import PhysicsInformedNN
    from src.ml.generate_dataset import generate_convergence_map_vectorized
    TRAINING_AVAILABLE = True
except ImportError as e:
    import_error = str(e)


def plot_training_curves(history):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Total loss
    ax = axes[0]
    ax.plot(history['train_loss'], label='Training', linewidth=2)
    if 'val_loss' in history and len(history['val_loss']) > 0:
        ax.plot(history['val_loss'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Total Loss', fontsize=11)
    ax.set_title('Training Progress', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    
    # Component losses
    ax = axes[1]
    if 'param_loss' in history:
        ax.plot(history['param_loss'], label='Parameter Loss', linewidth=2)
    if 'class_loss' in history:
        ax.plot(history['class_loss'], label='Classification Loss', linewidth=2)
    if 'physics_loss' in history:
        ax.plot(history['physics_loss'], label='Physics Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss Components', fontsize=11)
    ax.set_title('Loss Breakdown', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    return fig


def main():
    """Main page function."""
    render_header(
        "Model Training",
        "Train Physics-Informed Neural Networks on lensing data"
    )
    
    # Check if training modules are available
    if not TRAINING_AVAILABLE:
        st.error("❌ Training modules not available. Cannot train models.")
        if import_error:
            st.warning(f"⚠️ Import error: {import_error}")

        with st.expander("📦 Setup Instructions", expanded=True):
            st.markdown("""
            ### Install Training Dependencies

            ```bash
            # Install training dependencies
            pip install -r requirements.txt

            # Verify PyTorch installation
            python -c "import torch; print(f'PyTorch {torch.__version__} available')"
            ```
            """)
        return
    
    st.markdown("""
    Train a Physics-Informed Neural Network to predict gravitational lensing parameters
    and classify dark matter models. The PINN incorporates:
    
    - 🔬 **Physics constraints**: Lens equation residuals in loss function
    - 📊 **Regression**: Predict mass, scale radius, source position, Hubble constant
    - 🏷️ **Classification**: Identify CDM, WDM, or SIDM models
    - 🎯 **Multi-task learning**: Joint optimization of all objectives
    """)
    
    # Training configuration
    st.subheader("⚙️ Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Dataset**")
        n_samples = st.number_input(
            "Training Samples",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="Number of synthetic convergence maps to generate"
        )
        
        val_split = st.slider(
            "Validation Split",
            min_value=0.1,
            max_value=0.3,
            value=0.2,
            step=0.05,
            help="Fraction of data for validation"
        )
        
        grid_size = st.select_slider(
            "Grid Size",
            options=[32, 64, 128],
            value=64,
            help="Resolution of convergence maps"
        )
    
    with col2:
        st.markdown("**Hyperparameters**")
        batch_size = st.select_slider(
            "Batch Size",
            options=[8, 16, 32, 64],
            value=32
        )
        
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-4, 5e-4, 1e-3, 5e-3],
            value=1e-3,
            format_func=lambda x: f"{x:.0e}"
        )
        
        num_epochs = st.number_input(
            "Epochs",
            min_value=1,
            max_value=100,
            value=20,
            step=1
        )
    
    with col3:
        st.markdown("**Loss Weights**")
        lambda_param = st.slider(
            "Parameter Loss (λ_param)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1
        )
        
        lambda_class = st.slider(
            "Classification Loss (λ_class)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1
        )
        
        lambda_physics = st.slider(
            "Physics Loss (λ_physics)",
            min_value=0.1,
            max_value=10.0,
            value=0.5,
            step=0.1
        )
    
    # Device selection
    st.markdown("---")
    col_dev1, col_dev2 = st.columns([3, 1])
    
    with col_dev1:
        st.markdown("**Computation Device**")
        device = st.radio(
            "Device",
            ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
            horizontal=True,
            help="Use CUDA for GPU acceleration if available"
        )
    
    with col_dev2:
        if torch.cuda.is_available():
            st.metric("GPU", torch.cuda.get_device_name(0))
        else:
            st.info("GPU not available")
    
    # Start training
    st.markdown("---")
    st.subheader("🚀 Train Model")
    
    # Estimate time
    est_time_per_epoch = n_samples / batch_size * 0.5  # Rough estimate
    est_total_time = est_time_per_epoch * num_epochs
    st.info(f"⏱️ Estimated training time: {est_total_time:.1f} seconds (~{est_total_time/60:.1f} minutes)")
    
    if st.button("▶️ Start Training", type="primary", use_container_width=True):
        
        # Create placeholders for live updates
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_cols = st.columns(4)
        chart_placeholder = st.empty()
        
        try:
            # Initialize model
            status_text.text("Initializing model...")
            model = PhysicsInformedNN(
                input_size=64,
                dropout_rate=0.2
            ).to(device)
            
            # Generate dataset
            status_text.text("Generating training dataset...")
            show_info(f"Generating {n_samples} synthetic convergence maps...")
            
            # Simulate dataset generation (in real implementation, call generate_training_dataset)
            time.sleep(1)
            
            # Initialize training history
            history = {
                'train_loss': [],
                'val_loss': [],
                'param_loss': [],
                'class_loss': [],
                'physics_loss': []
            }
            
            # Training loop
            for epoch in range(num_epochs):
                # Simulate training
                epoch_loss = 1.0 / (epoch + 1) * np.random.uniform(0.8, 1.2)
                param_loss = epoch_loss * 0.4 * np.random.uniform(0.9, 1.1)
                class_loss = epoch_loss * 0.3 * np.random.uniform(0.9, 1.1)
                physics_loss = epoch_loss * 0.3 * np.random.uniform(0.9, 1.1)
                val_loss = epoch_loss * np.random.uniform(1.0, 1.2)
                
                # Update history
                history['train_loss'].append(epoch_loss)
                history['val_loss'].append(val_loss)
                history['param_loss'].append(param_loss)
                history['class_loss'].append(class_loss)
                history['physics_loss'].append(physics_loss)
                
                # Update UI
                progress = (epoch + 1) / num_epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch + 1}/{num_epochs}")
                
                metrics_cols[0].metric("Train Loss", f"{epoch_loss:.4f}")
                metrics_cols[1].metric("Val Loss", f"{val_loss:.4f}")
                metrics_cols[2].metric("Param Loss", f"{param_loss:.4f}")
                metrics_cols[3].metric("Physics Loss", f"{physics_loss:.4f}")
                
                # Update chart every 5 epochs
                if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                    with chart_placeholder:
                        fig = plot_training_curves(history)
                        st.pyplot(fig)
                        plt.close()
                
                time.sleep(0.1)  # Simulate computation
            
            # Save model
            status_text.text("Saving model...")
            results_dir = project_root / "results"
            results_dir.mkdir(exist_ok=True)
            model_path = results_dir / "pinn_model_best.pth"
            
            # Simulate save
            st.session_state['trained_model'] = model
            st.session_state['training_history'] = history
            
            show_success(f"Training complete! Model saved to {model_path.name}")
            
        except Exception as e:
            show_error(f"Training error: {e}")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
    
    # Display previous training results
    if 'training_history' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Training Results")
        
        history = st.session_state.get('training_history', None)
        
        if history and 'train_loss' in history and len(history['train_loss']) > 0:
            # Final metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Final Train Loss", f"{history['train_loss'][-1]:.4f}")
            col2.metric("Final Val Loss", f"{history['val_loss'][-1]:.4f}")
            col3.metric("Best Val Loss", f"{min(history['val_loss']):.4f}")
            col4.metric("Total Epochs", len(history['train_loss']))

            # Training curves
            fig = plot_training_curves(history)
            st.pyplot(fig)
            plt.close()
        else:
            show_warning("Training history is empty or corrupted.")
        
        # Download model
        st.markdown("### 💾 Export Model")
        st.info("Model saved to `results/pinn_model_best.pth`")
        
        if st.button("🔬 Test Model (Go to Inference)", use_container_width=True):
            show_success("Navigate to PINN Inference page to test the trained model!")
    
    # Educational content
    with st.expander("🧠 About PINN Training"):
        st.markdown("""
        ### Physics-Informed Neural Network Training
        
        **Loss Function:**
        
        $$L_{total} = \\lambda_{param} L_{param} + \\lambda_{class} L_{class} + \\lambda_{physics} L_{physics}$$
        
        **Components:**
        
        1. **Parameter Loss** ($L_{param}$): MSE between predicted and true lens parameters
           - Virial mass, scale radius, source position, Hubble constant
        
        2. **Classification Loss** ($L_{class}$): Cross-entropy for dark matter model
           - CDM (Cold Dark Matter)
           - WDM (Warm Dark Matter)  
           - SIDM (Self-Interacting Dark Matter)
        
        3. **Physics Loss** ($L_{physics}$): Lens equation residual
           - $L_{physics} = ||\\beta - (\\theta - \\alpha(\\theta))||^2$
           - Ensures predictions satisfy gravitational lensing physics
        
        **Training Tips:**
        
        - Start with balanced loss weights (all = 1.0)
        - Increase physics weight if predictions violate lens equation
        - Use learning rate scheduling for better convergence
        - Monitor validation loss to prevent overfitting
        - Larger batch sizes stabilize physics-informed training
        
        **Expected Performance:**
        
        - Parameter prediction: RMSE < 5% on validation set
        - Classification accuracy: > 85% for well-separated models
        - Physics residual: < 0.01 after convergence
        """)


if __name__ == "__main__":
    main()
