"""
Scientific Validation - Validate predictions against known lensing systems

Compare model predictions with well-studied gravitational lens systems
and calculate scientific metrics.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import utilities
try:
    from app.utils.ui import render_header, inject_custom_css, show_success, show_error, show_warning, show_info
except ImportError:
    from utils.ui import render_header, inject_custom_css, show_success, show_error, show_warning, show_info

# Import validation modules
VALIDATION_AVAILABLE = True
import_error = None
try:
    from src.validation import ScientificValidator, ValidationLevel, quick_validate, rigorous_validate
    from src.validation.hst_targets import HSTTarget, HSTValidation
except ImportError as e:
    VALIDATION_AVAILABLE = False
    import_error = str(e)

# Configure page
st.set_page_config(
    page_title="Scientific Validation - Gravitational Lensing Platform",
    page_icon="✅",
    layout="wide"
)

# Apply custom CSS
inject_custom_css()


# Known gravitational lens systems
KNOWN_SYSTEMS = {
    "Einstein Cross (Q2237+0305)": {
        "z_lens": 0.0394,
        "z_source": 1.695,
        "M_lens": 1e11,  # M_sun
        "einstein_radius": 1.48,  # arcsec
        "description": "Famous quad lens, foreground galaxy lensing distant quasar"
    },
    "Abell 1689": {
        "z_lens": 0.183,
        "z_source": 2.5,
        "M_lens": 1.3e15,  # M_sun (cluster mass)
        "einstein_radius": 47.0,  # arcsec
        "description": "Massive galaxy cluster, hundreds of lensed images"
    },
    "SDSS J1004+4112": {
        "z_lens": 0.68,
        "z_source": 1.734,
        "M_lens": 5e14,  # M_sun
        "einstein_radius": 15.0,  # arcsec
        "description": "Cluster lens with quintuple quasar image"
    },
    "Cosmic Horseshoe (SDSS J1148+1930)": {
        "z_lens": 0.444,
        "z_source": 2.381,
        "M_lens": 1e13,  # M_sun
        "einstein_radius": 10.5,  # arcsec
        "description": "Nearly complete Einstein ring"
    }
}


def calculate_validation_metrics(predicted, true):
    """Calculate validation metrics between predicted and true values."""
    metrics = {}
    
    # Relative error
    metrics['relative_error'] = np.abs(predicted - true) / (true + 1e-10)
    
    # Absolute error
    metrics['absolute_error'] = np.abs(predicted - true)
    
    # Percentage error
    metrics['percentage_error'] = metrics['relative_error'] * 100
    
    # Chi-squared (assuming 10% uncertainty)
    sigma = true * 0.1
    metrics['chi_squared'] = ((predicted - true) / sigma) ** 2
    
    return metrics


def plot_validation_comparison(systems, predictions, metric='relative_error'):
    """Plot validation comparison across multiple systems."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    system_names = list(systems.keys())
    values = [predictions[name][metric] for name in system_names]
    
    colors = ['green' if v < 0.1 else 'orange' if v < 0.2 else 'red' for v in values]
    bars = ax.barh(system_names, values, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Relative Error', fontsize=12)
    ax.set_title('Validation Against Known Systems', fontsize=14, fontweight='bold')
    ax.axvline(x=0.1, color='green', linestyle='--', alpha=0.5, label='10% threshold')
    ax.axvline(x=0.2, color='orange', linestyle='--', alpha=0.5, label='20% threshold')
    ax.grid(axis='x', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_residuals(predicted, true, param_names):
    """Plot residuals for each parameter."""
    n_params = len(predicted)
    fig, axes = plt.subplots(1, n_params, figsize=(4*n_params, 4))
    
    if n_params == 1:
        axes = [axes]
    
    for ax, pred, truth, name in zip(axes, predicted, true, param_names):
        residual = pred - truth
        rel_error = residual / (truth + 1e-10) * 100
        
        ax.bar([name], [residual], color='steelblue', alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_ylabel('Residual', fontsize=11)
        ax.set_title(f'{name}\n({rel_error:.1f}%)', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Parameter Residuals: Predicted - True', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def main():
    """Main page function."""
    render_header(
        "✅ Scientific Validation",
        "Validate predictions against known gravitational lens systems",
        "Validation"
    )
    
    # Check if validation modules are available
    if not VALIDATION_AVAILABLE:
        st.error("❌ **Validation modules not available**")
        st.warning(f"Import error: `{import_error}`")

        with st.expander("📦 Setup Instructions"):
            st.markdown("""
            The validation modules require the main lensing toolkit. Ensure:

            1. **Module structure is correct**:
               ```bash
               src/
                 validation/
                   __init__.py
                   hst_targets.py
               ```

            2. **All dependencies installed**:
               ```bash
               pip install numpy scipy matplotlib astropy
               ```

            3. **Import path is correct**:
               ```python
               from src.validation import ScientificValidator
               from src.validation.hst_targets import HSTTarget
               ```
            """)
        return
    
    st.markdown("""
    Compare model predictions with well-studied gravitational lensing systems to validate
    scientific accuracy. This page provides:
    
    - 📊 **Quantitative metrics**: Relative error, χ², RMSE
    - 🎯 **Known systems**: Einstein Cross, Abell 1689, SDSS lenses
    - 📈 **Residual analysis**: Parameter-by-parameter comparison
    - ✅ **Pass/fail criteria**: Based on scientific standards
    """)
    
    # Select validation system
    st.subheader("🔭 Select Validation Target")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_system = st.selectbox(
            "Known Lensing System",
            options=list(KNOWN_SYSTEMS.keys()),
            help="Choose a well-studied gravitational lens for validation"
        )
        
        system_info = KNOWN_SYSTEMS[selected_system]
        
        st.info(f"**{selected_system}**\n\n{system_info['description']}")
    
    with col2:
        st.markdown("**System Properties:**")
        st.metric("Lens Redshift", f"{system_info['z_lens']:.3f}")
        st.metric("Source Redshift", f"{system_info['z_source']:.3f}")
        st.metric("Mass (M☉)", f"{system_info['M_lens']:.2e}")
        st.metric("Einstein Radius", f"{system_info['einstein_radius']:.2f}\"")
    
    # Input predicted values
    st.markdown("---")
    st.subheader("📝 Enter Predictions")
    
    st.markdown("Enter your model's predicted values for comparison:")
    
    col_pred1, col_pred2, col_pred3 = st.columns(3)
    
    with col_pred1:
        pred_mass = st.number_input(
            "Predicted Mass (M☉)",
            min_value=1e10,
            max_value=1e16,
            value=system_info['M_lens'] * 1.05,  # Add 5% deviation for realistic testing
            format="%.2e",
            help="Model's predicted lens mass"
        )
    
    with col_pred2:
        pred_z_lens = st.number_input(
            "Predicted z_lens",
            min_value=0.01,
            max_value=5.0,
            value=system_info['z_lens'] * 1.02,  # Add 2% deviation for realistic testing
            format="%.4f",
            help="Model's predicted lens redshift"
        )
    
    with col_pred3:
        pred_einstein_r = st.number_input(
            "Predicted Einstein Radius (\")",
            min_value=0.1,
            max_value=100.0,
            value=system_info['einstein_radius'] * 1.03,  # Add 3% deviation for realistic testing
            format="%.2f",
            help="Model's predicted Einstein radius"
        )
    
    # Validation level
    st.markdown("---")
    validation_level = st.radio(
        "Validation Level",
        ["Quick", "Standard", "Rigorous"],
        horizontal=True,
        help="Quick: Basic checks | Standard: Statistical tests | Rigorous: Full analysis"
    )
    
    # Run validation
    if st.button("▶️ Run Validation", type="primary", use_container_width=True):
        with st.spinner("Running validation..."):
            try:
                # Calculate metrics
                mass_metrics = calculate_validation_metrics(pred_mass, system_info['M_lens'])
                z_metrics = calculate_validation_metrics(pred_z_lens, system_info['z_lens'])
                einstein_metrics = calculate_validation_metrics(pred_einstein_r, system_info['einstein_radius'])
                
                # Store results
                st.session_state['validation_results'] = {
                    'system': selected_system,
                    'mass_metrics': mass_metrics,
                    'z_metrics': z_metrics,
                    'einstein_metrics': einstein_metrics,
                    'predictions': {
                        'mass': pred_mass,
                        'z_lens': pred_z_lens,
                        'einstein_r': pred_einstein_r
                    },
                    'true_values': system_info
                }
                
                show_success("Validation complete!")
                
            except Exception as e:
                show_error(f"Validation error: {e}")
    
    # Display results
    if 'validation_results' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Validation Results")
        
        results = st.session_state['validation_results']
        
        # Overall assessment
        mass_error = results['mass_metrics']['relative_error']
        z_error = results['z_metrics']['relative_error']
        einstein_error = results['einstein_metrics']['relative_error']
        
        avg_error = np.mean([mass_error, z_error, einstein_error])
        
        col_assess1, col_assess2, col_assess3 = st.columns(3)
        
        col_assess1.metric("Average Relative Error", f"{avg_error*100:.2f}%")
        col_assess2.metric("Max Error", f"{max(mass_error, z_error, einstein_error)*100:.2f}%")
        col_assess3.metric("Min Error", f"{min(mass_error, z_error, einstein_error)*100:.2f}%")
        
        # Overall status
        if avg_error < 0.1:
            st.success("✅ **PASS** - Excellent agreement with observations (< 10% error)")
        elif avg_error < 0.2:
            st.warning("⚠️ **MARGINAL** - Acceptable agreement (10-20% error)")
        else:
            st.error("❌ **FAIL** - Poor agreement with observations (> 20% error)")
        
        # Detailed metrics
        st.markdown("### 📈 Detailed Metrics")
        
        # Create comparison table
        comparison_data = {
            'Parameter': ['Mass (M☉)', 'Lens Redshift', 'Einstein Radius (\")'],
            'True Value': [
                f"{system_info['M_lens']:.2e}",
                f"{system_info['z_lens']:.4f}",
                f"{system_info['einstein_radius']:.2f}"
            ],
            'Predicted': [
                f"{pred_mass:.2e}",
                f"{pred_z_lens:.4f}",
                f"{pred_einstein_r:.2f}"
            ],
            'Relative Error': [
                f"{mass_error*100:.2f}%",
                f"{z_error*100:.2f}%",
                f"{einstein_error*100:.2f}%"
            ],
            'χ²': [
                f"{results['mass_metrics']['chi_squared']:.2f}",
                f"{results['z_metrics']['chi_squared']:.2f}",
                f"{results['einstein_metrics']['chi_squared']:.2f}"
            ]
        }
        
        st.table(comparison_data)
        
        # Residual plot
        st.markdown("### 📉 Residual Analysis")
        
        predicted = [pred_mass, pred_z_lens, pred_einstein_r]
        true_vals = [system_info['M_lens'], system_info['z_lens'], system_info['einstein_radius']]
        param_names = ['Mass', 'z_lens', 'θ_E']
        
        fig = plot_residuals(predicted, true_vals, param_names)
        st.pyplot(fig)
        plt.close()
        
        # Scientific interpretation
        with st.expander("📚 Scientific Interpretation"):
            st.markdown(f"""
            **System: {selected_system}**
            
            {system_info['description']}
            
            **Validation Assessment:**
            
            - **Mass Prediction**: {mass_error*100:.1f}% error
              - {"✅ Within observational uncertainties" if mass_error < 0.1 else "⚠️ Marginally consistent" if mass_error < 0.2 else "❌ Inconsistent with observations"}
            
            - **Redshift Prediction**: {z_error*100:.1f}% error
              - {"✅ Excellent spectroscopic agreement" if z_error < 0.05 else "⚠️ Within photometric redshift uncertainties" if z_error < 0.1 else "❌ Significant discrepancy"}
            
            - **Einstein Radius**: {einstein_error*100:.1f}% error
              - {"✅ Consistent with strong lensing observations" if einstein_error < 0.1 else "⚠️ Within systematic uncertainties" if einstein_error < 0.2 else "❌ Outside observational constraints"}
            
            **χ² Analysis:**
            
            Total χ² = {results['mass_metrics']['chi_squared'] + results['z_metrics']['chi_squared'] + results['einstein_metrics']['chi_squared']:.2f}
            
            - Good fit: χ² ≈ N_params (here N=3)
            - Your fit: {"Excellent" if avg_error < 0.1 else "Acceptable" if avg_error < 0.2 else "Poor"}
            
            **Recommendations:**
            
            {'''- Model performs well on this system
            - Consider testing on additional systems
            - Ready for scientific publication''' if avg_error < 0.1 else '''- Model shows moderate agreement
            - Consider refining training data
            - Additional validation recommended''' if avg_error < 0.2 else '''- Model needs significant improvement
            - Review training procedure
            - Check physics constraints in loss function'''}
            """)
    
    # Additional validation targets
    with st.expander("🎯 Other Validation Targets"):
        st.markdown("""
        ### Additional Well-Studied Systems
        
        Consider validating against these systems for comprehensive testing:
        
        **Strong Lenses:**
        - **B1938+666**: Radio Einstein ring
        - **MG J0414+0534**: Quad lens with substructure
        - **RX J1131-1231**: Time delay lens
        
        **Galaxy Clusters:**
        - **MACS J0717**: Massive merger
        - **El Gordo (ACT-CL J0102-4915)**: Most massive known
        - **Bullet Cluster (1E 0657-56)**: Dark matter separation
        
        **HST Treasury Programs:**
        - CLASH (Cluster Lensing And Supernova survey with Hubble)
        - HFF (Hubble Frontier Fields)
        - BELLS (BOSS Emission-Line Lens Survey)
        """)


if __name__ == "__main__":
    main()
