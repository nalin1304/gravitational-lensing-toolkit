"""
Bayesian Uncertainty Quantification - Estimate prediction uncertainties

Use Bayesian methods to quantify epistemic and aleatoric uncertainty
in PINN predictions.
"""

import streamlit as st

# Configure page (must be first Streamlit command)
st.set_page_config(
    page_title="Bayesian UQ - Gravitational Lensing Platform",
    page_icon="📊",
    layout="wide"
)

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import utilities
try:
    from app.utils.ui import render_header, inject_custom_css, show_success, show_error, show_warning, show_info
except ImportError:
    from utils.ui import render_header, inject_custom_css, show_success, show_error, show_warning, show_info

# Apply custom CSS
inject_custom_css()

# Import uncertainty modules
UQ_AVAILABLE = True
import_error = None
try:
    from src.ml.uncertainty import BayesianPINN, UncertaintyCalibrator, visualize_uncertainty
except ImportError as e:
    UQ_AVAILABLE = False
    import_error = str(e)


def plot_uncertainty_distribution(samples, param_name, true_value=None):
    """Plot uncertainty distribution for a parameter."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Histogram
    n, bins, patches = ax.hist(samples, bins=30, alpha=0.7, color='steelblue', 
                                edgecolor='black', density=True)
    
    # Statistics
    mean = np.mean(samples)
    std = np.std(samples)
    median = np.median(samples)
    
    # Plot lines
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.3f}')
    ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.3f}')
    
    if true_value is not None:
        ax.axvline(true_value, color='orange', linestyle='--', linewidth=2, 
                  label=f'True: {true_value:.3f}')
    
    # Confidence intervals
    ci_lower, ci_upper = np.percentile(samples, [2.5, 97.5])
    ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='green', label='95% CI')
    
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(f'Posterior Distribution: {param_name}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_credible_intervals(params, uncertainties, param_names):
    """Plot credible intervals for all parameters."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(param_names))
    
    # Plot error bars
    ax.errorbar(params, y_pos, xerr=uncertainties*1.96, fmt='o', 
                markersize=8, capsize=5, capthick=2, 
                color='steelblue', ecolor='gray', linewidth=2)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names)
    ax.set_xlabel('Parameter Value', fontsize=12)
    ax.set_title('95% Credible Intervals', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(samples, param_names):
    """Plot parameter correlation matrix."""
    # Calculate correlation
    corr_matrix = np.corrcoef(samples.T)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(param_names)))
    ax.set_yticks(np.arange(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_yticklabels(param_names)
    
    # Add correlation values
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title('Parameter Correlation Matrix', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Correlation Coefficient')
    
    plt.tight_layout()
    return fig


def main():
    """Main page function."""
    render_header(
        "📊 Bayesian Uncertainty Quantification",
        "Quantify epistemic and aleatoric uncertainties in predictions",
        "Advanced"
    )
    
    # Check if uncertainty quantification modules are available
    if not UQ_AVAILABLE:
        st.error("❌ **Uncertainty quantification modules not available**")
        st.warning(f"Import error: `{import_error}`")

        with st.expander("📦 Setup Instructions"):
            st.markdown("""
            Bayesian UQ requires PyTorch and the ML toolkit:

            1. **Install PyTorch**:
               ```bash
               pip install torch torchvision
               ```

            2. **Install ML dependencies**:
               ```bash
               pip install scikit-learn scipy
               ```

            3. **Verify installation**:
               ```bash
               python -c "import torch; from src.ml.uncertainty import BayesianPINN; print('✅ OK')"
               ```

            4. **Check module structure**:
               ```bash
               src/
                 ml/
                   __init__.py
                   uncertainty.py
               ```
            """)
        return
    
    st.markdown("""
    Bayesian uncertainty quantification provides rigorous statistical estimates of prediction
    confidence. This is crucial for:
    
    - 🎯 **Confidence intervals**: 95% credible regions for parameters
    - 🔬 **Epistemic uncertainty**: Model uncertainty (reducible with more data)
    - 🎲 **Aleatoric uncertainty**: Inherent data noise (irreducible)
    - 📊 **Calibration**: Are confidence intervals well-calibrated?
    """)
    
    # Configuration
    st.subheader("⚙️ Uncertainty Estimation Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        method = st.selectbox(
            "Bayesian Method",
            ["Monte Carlo Dropout", "Deep Ensembles", "Variational Inference"],
            help="Method for uncertainty quantification"
        )
        
        n_samples = st.slider(
            "Number of Samples",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="More samples = better uncertainty estimates"
        )
    
    with col2:
        confidence_level = st.slider(
            "Confidence Level",
            min_value=0.80,
            max_value=0.99,
            value=0.95,
            step=0.01,
            format="%.2f",
            help="Credible interval coverage (0.95 = 95%)"
        )
        
        calibration_check = st.checkbox(
            "Run Calibration Analysis",
            value=True,
            help="Verify that confidence intervals are well-calibrated"
        )
    
    # Input data
    st.markdown("---")
    st.subheader("📥 Input Data")
    
    data_source = st.radio(
        "Data Source",
        ["Use Session Data"],
        horizontal=True
    )
    
    input_data = None
    
    if data_source == "Use Session Data":
        if 'convergence_map' in st.session_state:
            input_data = st.session_state['convergence_map']
            if input_data is not None and hasattr(input_data, 'shape'):
                show_info(f"Using convergence map: {input_data.shape}")
            else:
                input_data = None
                show_warning("No valid data in session. Please generate data first.")
        else:
            input_data = None
            show_warning("No data in session. Please generate data first.")
    # No synthetic fallback; require valid session data
    
    # Run uncertainty quantification
    if input_data is not None:
        st.markdown("---")
        
        if st.button("▶️ Estimate Uncertainties", type="primary", use_container_width=True):
            with st.spinner(f"Running {method} with {n_samples} samples..."):
                try:
                    # Run Bayesian UQ pipeline with actual implementation
                    uq = BayesianPINN(method=method)
                    samples = uq.sample(input_data, n_samples=n_samples)
                    means = np.mean(samples, axis=0)
                    stds = np.std(samples, axis=0)
                    medians = np.median(samples, axis=0)
                    ci_lower = np.percentile(samples, (1 - confidence_level) / 2 * 100, axis=0)
                    ci_upper = np.percentile(samples, (1 + confidence_level) / 2 * 100, axis=0)

                    st.session_state['uq_results'] = {
                        'samples': samples,
                        'means': means,
                        'stds': stds,
                        'medians': medians,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'true_params': None,
                        'method': method,
                        'n_samples': n_samples
                    }
                    show_success("Uncertainty quantification complete!")
                except FileNotFoundError as e:
                    show_error(f"Required model or data file missing: {e}")
                except ValueError as e:
                    show_error(f"Invalid input data: {e}")
                except Exception as e:
                    show_error(f"Error in uncertainty estimation: {e}")
    
    # Display results
    if 'uq_results' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Uncertainty Quantification Results")
        
        results = st.session_state['uq_results']
        param_names = ['M_vir (M☉)', 'r_s (kpc)', 'β_x (\")', 'β_y (\")', 'H₀ (km/s/Mpc)']
        
        # Summary statistics
        st.markdown("### 📈 Summary Statistics")
        
        cols = st.columns(5)
        for i, (col, name, mean, std) in enumerate(zip(cols, param_names, results['means'], results['stds'])):
            relative_unc = (std / mean) * 100
            col.metric(
                name.split('(')[0].strip(),
                f"{mean:.2e}" if mean > 1e6 else f"{mean:.2f}",
                f"±{relative_unc:.1f}%"
            )
        
        # Credible intervals plot
        st.markdown("### 🎯 Credible Intervals")
        
        fig = plot_credible_intervals(results['means'], results['stds'], param_names)
        st.pyplot(fig)
        plt.close()
        
        # Individual parameter distributions
        st.markdown("### 📊 Posterior Distributions")
        
        selected_param = st.selectbox(
            "Select Parameter",
            range(len(param_names)),
            format_func=lambda i: param_names[i]
        )
        
        fig = plot_uncertainty_distribution(
            results['samples'][:, selected_param],
            param_names[selected_param],
            results['true_params'][selected_param]
        )
        st.pyplot(fig)
        plt.close()
        
        # Correlation analysis
        st.markdown("### 🔗 Parameter Correlations")
        
        fig = plot_correlation_matrix(results['samples'], param_names)
        st.pyplot(fig)
        plt.close()
        
        # Uncertainty decomposition
        with st.expander("🔬 Uncertainty Decomposition"):
            st.markdown(f"""
            **Method:** {results['method']}  
            **Samples:** {results['n_samples']}
            
            ### Types of Uncertainty
            
            **Epistemic Uncertainty (Model Uncertainty):**
            - Arises from lack of knowledge about the model
            - Can be reduced with more training data
            - Estimated via: {results['method']}
            
            **Aleatoric Uncertainty (Data Uncertainty):**
            - Inherent noise in observations
            - Cannot be reduced (fundamental limit)
            - Estimated from data likelihood
            
            ### Relative Uncertainties
            
            | Parameter | Mean | Std | Relative Unc. | Coverage |
            |-----------|------|-----|---------------|----------|
            """)
            
            for i, name in enumerate(param_names):
                mean = results['means'][i]
                std = results['stds'][i]
                rel_unc = (std / mean) * 100
                coverage = 95  # Placeholder
                st.markdown(f"| {name} | {mean:.2e} | {std:.2e} | {rel_unc:.1f}% | {coverage}% |")
            
            st.markdown("""
            ### Interpretation Guidelines
            
            - **< 5% relative uncertainty**: Excellent precision
            - **5-10%**: Good precision, typical for strong lenses
            - **10-20%**: Moderate precision, acceptable for surveys
            - **> 20%**: High uncertainty, need more data or better model
            """)
        
        # Calibration analysis
        if calibration_check:
            with st.expander("📏 Calibration Analysis"):
                st.markdown("""
                ### Uncertainty Calibration
                
                Well-calibrated uncertainties mean:
                - 95% credible intervals contain the true value 95% of the time
                - Underconfident: intervals too wide
                - Overconfident: intervals too narrow
                
                **Calibration Metrics:**
                """)
                
                # Check coverage
                in_interval = []
                for i in range(len(param_names)):
                    true_val = results['true_params'][i]
                    lower = results['ci_lower'][i]
                    upper = results['ci_upper'][i]
                    in_interval.append(lower <= true_val <= upper)
                
                coverage = np.mean(in_interval) * 100
                
                st.metric("Empirical Coverage", f"{coverage:.1f}%")
                st.metric("Target Coverage", f"{confidence_level*100:.1f}%")
                
                if abs(coverage - confidence_level*100) < 5:
                    st.success("✅ Well-calibrated uncertainties!")
                elif coverage < confidence_level*100:
                    st.warning("⚠️ Overconfident - intervals too narrow")
                else:
                    st.info("ℹ️ Underconfident - intervals too wide")
    
    # Educational content
    with st.expander("🧠 Learn More: Bayesian Uncertainty Quantification"):
        st.markdown("""
        ### Bayesian Inference for Neural Networks
        
        **Goal:** Estimate posterior distribution over parameters
        
        $$p(\\theta | D) = \\frac{p(D | \\theta) p(\\theta)}{p(D)}$$
        
        Where:
        - $\\theta$ = model parameters
        - $D$ = observed data
        - $p(\\theta | D)$ = posterior (what we want)
        - $p(D | \\theta)$ = likelihood
        - $p(\\theta)$ = prior
        
        ### Methods Implemented
        
        **1. Monte Carlo Dropout**
        - Apply dropout at test time
        - Run multiple forward passes
        - Variation gives uncertainty estimate
        - Fast, but approximate
        
        **2. Deep Ensembles**
        - Train multiple models independently
        - Disagreement between models = uncertainty
        - More accurate, but computationally expensive
        
        **3. Variational Inference**
        - Learn approximate posterior distribution
        - Optimize KL divergence to true posterior
        - Principled Bayesian approach
        
        ### Applications
        
        - **Parameter constraints**: Full posterior distributions
        - **Model selection**: Compare evidence between models
        - **Experimental design**: Optimize future observations
        - **Risk assessment**: Quantify decision uncertainties
        """)


if __name__ == "__main__":
    main()
