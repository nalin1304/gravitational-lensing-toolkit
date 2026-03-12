# Gravitational Lensing Toolkit: Complete Feature Catalog

> **Exhaustive inventory of every feature, capability, class, and function in the codebase.**
> Auto-generated from full source-code audit. Last updated: March 2026.

---

## Table of Contents

1. [Lens Models (`src/lens_models/`)](#1-lens-models)
2. [Ray Optics (`src/optics/`)](#2-ray-optics)
3. [Machine Learning (`src/ml/`)](#3-machine-learning)
4. [Time Delay & Cosmography (`src/time_delay/`)](#4-time-delay--cosmography)
5. [Dark Matter Analysis (`src/dark_matter/`)](#5-dark-matter-analysis)
6. [Validation & Calibration (`src/validation/`)](#6-validation--calibration)
7. [Observational Data (`src/data/`)](#7-observational-data)
8. [Utilities (`src/utils/`)](#8-utilities)
9. [REST API (`api/`)](#9-rest-api)
10. [Streamlit UI (`app/`)](#10-streamlit-ui)
11. [Test Suite (`tests/`)](#11-test-suite)
12. [Research References](#12-research-references)

---

## 1. Lens Models

**Directory:** `src/lens_models/`

### 1.1 Lens System (`lens_system.py`)

| Feature | Description |
|---------|-------------|
| `LensSystem` class | Manages cosmological configuration: lens redshift `z_l`, source redshift `z_s`, Hubble constant `H0`, matter density `Om0` |
| Angular diameter distances | Computes `D_L`, `D_S`, `D_LS` using `astropy.cosmology.FlatLambdaCDM` |
| Critical surface density | Calculates `Sigma_crit = (c^2 / 4piG) * (D_S / D_L D_LS)` |
| Einstein radius | Derives Einstein radius for any mass profile |
| Redshift validation | Enforces `z_s > z_l` |

### 1.2 Mass Profiles (`mass_profiles.py`)

#### 1.2.1 Point Mass Profile

| Feature | Description |
|---------|-------------|
| `PointMassProfile` class | Single point-mass lens (Schwarzschild lens) |
| Deflection angle | `alpha = theta_E^2 / theta` (exact analytic) |
| Convergence | Delta-function convergence (Dirac delta at origin) |
| Surface density | `Sigma(r) = M * delta(r) / (pi * r^2)` |
| Einstein radius | `theta_E = sqrt(4GM D_LS / c^2 D_L D_S)` |
| Magnification | Analytic `mu = u^2 + 2 / (u * sqrt(u^2 + 4))` where `u = beta / theta_E` |

#### 1.2.2 NFW Profile (Navarro-Frenk-White)

| Feature | Description |
|---------|-------------|
| `NFWProfile` class | Standard CDM halo profile |
| Virial mass & concentration | Parameterized by `M_vir` and `c` |
| Characteristic density | `rho_s = M_vir / (4pi r_s^3 f(c))` |
| Convergence function | Three-regime piecewise: `x < 1`, `x = 1`, `x > 1` using `arctanh`/`arctan` |
| Deflection angle | NFW lensing function `g(x)` with `ln(x/2) + arctanh/arctan` branches |
| Surface density | Projected NFW density `Sigma(R) = 2 rho_s r_s f_NFW(x)` |
| Ellipticity support | Optional ellipticity parameter for non-circular halos |
| Vectorized computation | All methods accept array inputs for batch computation |

#### 1.2.3 Warm Dark Matter Profile

| Feature | Description |
|---------|-------------|
| `WarmDarkMatterProfile` class | WDM-suppressed NFW profile |
| WDM particle mass | `m_wdm` parameter (keV) |
| Transfer function suppression | `T(k) = [1 + (alpha*k)^(2*nu)]^(-5/nu)` (Viel et al. 2005) |
| Half-mode mass | `M_hm` computation for free-streaming cutoff |
| Modified concentration | WDM concentration-mass relation with suppression factor |
| Core radius | WDM-induced core from thermal velocities |

#### 1.2.4 Self-Interacting Dark Matter Profile

| Feature | Description |
|---------|-------------|
| `SIDMProfile` class | SIDM with isothermal core |
| Cross-section | `sigma_SIDM` parameter (cm^2/g) |
| Core-NFW transition | Matched isothermal core + NFW envelope at `r_1` |
| Core radius | `r_core = r_s * (sigma_SIDM / sigma_0)^alpha` scaling |
| Isothermal core | `rho_core(r) = rho_0 / (1 + (r/r_c)^2)` |
| Gravothermal evolution | Accounts for self-interaction rate-dependent core formation |

#### 1.2.5 Elliptical NFW Profile

| Feature | Description |
|---------|-------------|
| `EllipticalNFWProfile` class | Elliptical generalization of NFW |
| Elliptical radius | `xi = sqrt(x^2/(1-e) + y^2*(1-e))` (Golse & Kneib 2002) |
| Position angle | Rotation by angle `theta` applied to coordinates |
| Elliptical convergence | NFW convergence evaluated on elliptical radius |
| Elliptical deflection | Modified deflection with ellipticity corrections |

#### 1.2.6 Sersic Profile

| Feature | Description |
|---------|-------------|
| `SersicProfile` class | Stellar light/mass distribution |
| Sersic index | `n` parameter (n=1 exponential, n=4 de Vaucouleurs) |
| Effective radius | `R_eff` half-light radius |
| `b_n` coefficient | Approximation `b_n ~ 2n - 1/3 + 4/(405n)` |
| Surface brightness | `I(R) = I_e * exp(-b_n * [(R/R_e)^(1/n) - 1])` |
| Convergence | Mass-to-light ratio applied to Sersic surface brightness |
| Deflection angle | Numerical integration of projected mass |

#### 1.2.7 Composite Galaxy Model

| Feature | Description |
|---------|-------------|
| `CompositeGalaxyModel` class | Combined dark matter halo + stellar component |
| Dual profile | NFW (dark halo) + Sersic (stellar) superposition |
| Combined deflection | Vector sum of halo and stellar deflections |
| Combined convergence | Sum of halo and stellar convergences |
| Mass decomposition | Separate dark/stellar mass contributions |

### 1.3 Multi-Plane Lensing (`multi_plane.py`)

| Feature | Description |
|---------|-------------|
| `MultiPlaneLensing` class | Multi-plane ray tracing through N lens planes |
| Recursive lens equation | `beta_j = theta - sum_i D_ij/D_j alpha_i(theta_i)` |
| Distance ratios | Proper cosmological `D_ij` for each plane pair |
| Arbitrary lens count | Supports any number of intermediate lens planes |
| Coupled deflections | Each plane's deflection depends on positions at prior planes |

### 1.4 Cosmology (`cosmology.py`)

| Feature | Description |
|---------|-------------|
| Flat LCDM cosmology | `FlatLambdaCDM` from astropy |
| Comoving distance | `d_C(z)` numerical integration |
| Angular diameter distance | `d_A(z) = d_C / (1+z)` |
| Luminosity distance | `d_L(z) = d_C * (1+z)` |
| Critical density | `rho_crit(z)` at arbitrary redshift |
| Distance ratio | `D_LS / D_S` geometric factor for lensing |

---

## 2. Ray Optics

**Directory:** `src/optics/`

### 2.1 Ray Tracing (`ray_tracing.py`)

| Feature | Description |
|---------|-------------|
| Geometric ray tracing | Standard lens equation: `beta = theta - alpha(theta)` |
| Image finding | Grid search + refinement for multiple image positions |
| Magnification | Jacobian-based `mu = 1/det(A)` where `A = I - d_alpha/d_theta` |
| Critical curves | Contour where `det(A) = 0` |
| Caustics | Source-plane mapping of critical curves |
| Convergence maps | 2D grid evaluation of `kappa(x, y)` |

### 2.2 Ray Tracing Backends (`ray_tracing_backends.py`)

| Feature | Description |
|---------|-------------|
| `RayTracingMode` enum | `THIN_LENS` vs `SCHWARZSCHILD` mode selection |
| **Thin-lens ray trace** | Born approximation on FLRW background for galaxy-scale lensing |
| Image plane grid | Configurable resolution and extent |
| Connected-component labeling | `scipy.ndimage.label` for separating multiple images |
| Weighted centroid | Precision image position from inverse-distance weighting |
| Magnification via Jacobian | Central-difference Jacobian `A = I - d_alpha/d_theta` |
| **Schwarzschild geodesic** | Full GR null geodesic integration in Schwarzschild metric |
| `solve_ivp` DOP853 | 8th-order Runge-Kutta ODE solver for geodesic equations |
| Event detection | Horizon crossing and infinity-return terminal events |
| Christoffel symbols | Explicit `Gamma^r`, `Gamma^phi` for Schwarzschild |
| Effective potential | Photon effective potential `V_eff = (1 - r_s/r) L^2/r^2` |
| Weak-field fallback | Analytic `alpha = 4GM/c^2b` for `b >> r_s` |
| Method validation | `validate_method_compatibility()` enforces z_l <= 0.05 for Schwarzschild |
| Unified interface | `ray_trace()` dispatches to appropriate backend |
| Weak-field comparison | `compare_methods_weak_field()` cross-validates both backends |

### 2.3 Wave Optics (`wave_optics.py`)

| Feature | Description |
|---------|-------------|
| `WaveOpticsEngine` class | Fresnel-Kirchhoff diffraction integral for lensing |
| Amplification factor | `F(w) = (w/2pi i) int d^2theta exp[iw Phi(theta)]` |
| FFT-based computation | Fast Fourier transform for Fresnel integral |
| Interference patterns | Wave interference between multiple images |
| Dimensionless frequency | `w = 2pi f G M / c^3` |

### 2.4 Advanced Wave Optics (`advanced_wave_optics.py`)

| Feature | Description |
|---------|-------------|
| `LefschetzWaveOptics` class | Lefschetz thimble method for oscillatory integrals (Shi 2024) |
| Saddle point finder | Gradient-based detection of stationary points of Fermat potential |
| Thimble contribution | Hessian eigenvalue decomposition + Maslov index phase shift |
| Stationary phase approximation | `F ~ exp(iw Phi_0) / sqrt(det H)` at each saddle |
| Gaussian envelope | Local contribution spread via quadratic approximation |
| `ImprovedWaveOptics` class | Born approximation with higher-order corrections (Yarimoto & Oguri 2024) |
| Born-Fresnel integral | FFT of `exp(iw Phi)` with normalization |
| Correction factors | `F_corrected = F_Born * (1 + C1/w + C2/w^2)` for `w > 0.1` |
| Regime classifier | `check_wave_regime()`: geometric, wave, interference, diffraction |
| `compute_wave_amplification()` | Convenience function with automatic method selection |

### 2.5 GR Module (`gr_module.py`)

| Feature | Description |
|---------|-------------|
| EinsteinPy integration | Full GR geodesic integration using `einsteinpy.geodesic.Geodesic` |
| Schwarzschild geodesics | Null geodesic trajectories around point masses |
| Kerr geodesics | Spinning black hole spacetime (if EinsteinPy supports) |
| Deflection extraction | Total angular deviation from asymptotic trajectory |

---

## 3. Machine Learning

**Directory:** `src/ml/`

### 3.1 Base PINN (`pinn_model.py`)

| Feature | Description |
|---------|-------------|
| `PhysicsInformedNN` class | CNN-based PINN for lensing parameter estimation |
| Multi-head architecture | Shared encoder + parameter regression head + classification head |
| Convolutional encoder | `Conv2d` → `BatchNorm` → `ReLU` → `MaxPool` feature extraction |
| Parameter regression | Predicts `[M_vir, r_s, beta_x, beta_y, H0]` |
| DM classification | 3-class output: CDM, WDM, SIDM |
| Dropout regularization | Configurable dropout rate for uncertainty estimation |

### 3.2 Advanced PINN (`pinn_advanced.py`)

| Feature | Description |
|---------|-------------|
| `AdvancedPINN` class | Enhanced architecture with modern deep learning components |
| **Residual blocks** | `ResidualBlock` with skip connections, BN, optional downsampling |
| **Self-attention** | `SelfAttention` module: `Q`, `K`, `V` projections with softmax attention |
| **Multi-scale features** | `MultiScaleFeatureExtractor`: parallel 1x1, 3x3, 5x5, pool branches (Inception-style) |
| **Adaptive activation** | `AdaptiveActivation` with learnable `alpha * swish + (1-alpha) * tanh` |
| **Physics-constrained layer** | `PhysicsConstrainedLayer`: enforces `M_vir > 0`, `c > 0`, `0 <= e < 1`, `0 <= theta < pi` |
| Feature pyramid | Multi-resolution feature maps for scale-invariant detection |
| Gradient clipping | Built-in gradient norm clipping during training |

### 3.3 Coordinate-Based PINN (`coordinate_pinn.py`)

| Feature | Description |
|---------|-------------|
| `CoordinatePINN` class | Takes `(x, y)` coordinates, predicts lensing potential `psi(x,y)` |
| Fourier feature encoding | Random Fourier features: `gamma(x) = [cos(Bx), sin(Bx)]` for high-freq learning |
| `torch.autograd` derivatives | Exact `d_psi/dx`, `d_psi/dy` via automatic differentiation |
| Second-order derivatives | `d2_psi/dx2`, `d2_psi/dy2` for Laplacian computation |
| **Physics losses** | Poisson: `||nabla^2 psi - 2 kappa||^2`, Deflection: `||alpha - nabla psi||^2` |
| Convergence from potential | `kappa = 0.5 * nabla^2 psi` derived quantity |
| Shear computation | `gamma_1 = 0.5*(psi_xx - psi_yy)`, `gamma_2 = psi_xy` |
| Boundary conditions | Asymptotic `psi -> 0` enforcement at large radii |

### 3.4 Score-Based Lensing (`score_based_lensing.py`)

| Feature | Description |
|---------|-------------|
| `ScoreBasedLensing` class | Score-based generative model for blind lens inversion (Barco et al. 2025) |
| U-Net backbone | Encoder-decoder with skip connections and time embedding |
| Time embedding | Sinusoidal positional encoding for diffusion timestep |
| Score function | `s_theta(x, t) ~ nabla_x log p_t(x)` |
| Forward SDE | `dx = f(x,t)dt + g(t)dW` progressive noise addition |
| Reverse SDE | `dx = [f - g^2 s_theta] dt + g dW_bar` denoising |
| Langevin dynamics | Corrector step with step-size-controlled MCMC |
| Joint inference | Simultaneously recovers source brightness and lens mass distribution |
| Multi-channel support | 2-channel output: source image + convergence map |

### 3.5 Neural Posterior Estimation (`neural_posterior_estimation.py`)

| Feature | Description |
|---------|-------------|
| `NeuralPosteriorEstimator` class | Amortized Bayesian inference replacing MCMC (Venkatraman et al. 2025) |
| CNN summary network | ResNet-style encoder compresses image to feature vector |
| **Normalizing flows** | `RealNVP`-style affine coupling layers for flexible posterior |
| Scale & translation nets | Learned `s(x)` and `t(x)` transformations per coupling layer |
| Log-determinant tracking | Exact `log |det J|` for density evaluation |
| Posterior sampling | Direct sampling from `q_phi(theta|x)` without MCMC |
| KL divergence objective | Sequential Neural Posterior Estimation (SNPE-C) training |
| Amortized inference | Single forward pass for new observations (no retraining) |

### 3.6 JWST Lens Detection (`jwst_lens_detection.py`)

| Feature | Description |
|---------|-------------|
| `JWSTLensFinder` class | Two-stage lens detection for JWST data (Silver et al. 2025) |
| **ResNet detector** | 18-layer residual network for conventional strong lens detection |
| **U-Net segmenter** | Encoder-decoder for pixel-level weak/dwarf-galaxy lens segmentation |
| Low-mass lens detection | Targets `theta_E < 0.5"`, `M_halo < 10^11 M_sun` |
| Multi-filter support | Handles JWST NIRCam multi-band input |
| Confidence scoring | Sigmoid probability output for detection confidence |
| Two-stage pipeline | Classification (is it a lens?) then segmentation (where is the lens?) |

### 3.7 Transfer Learning (`transfer_learning.py`)

| Feature | Description |
|---------|-------------|
| `TransferLearningTrainer` class | Sim-to-real domain adaptation |
| **DANN** | Domain-Adversarial Neural Network with gradient reversal layer |
| **MMD** | Maximum Mean Discrepancy with Gaussian RBF kernel |
| **CORAL** | Correlation Alignment: minimizes covariance matrix difference |
| `GradientReversalLayer` | Custom autograd function: identity forward, negated gradient backward |
| Domain discriminator | MLP to classify source vs. target domain |
| Feature extractor freezing | Selective layer freezing for fine-tuning |
| Bayesian uncertainty | MC Dropout for epistemic uncertainty quantification |
| Progressive unfreezing | Gradual layer unfreezing schedule |

### 3.8 Data Augmentation (`augmentation.py`)

| Feature | Description |
|---------|-------------|
| `RandomRotation90` | Random 0/90/180/270 degree rotations |
| `RandomFlip` | Horizontal and vertical random flips |
| `RandomBrightness` | Multiplicative brightness jitter `[1-factor, 1+factor]` |
| `RandomNoise` | Additive Gaussian noise with configurable std |
| `get_training_transforms()` | Composed pipeline of all augmentations |
| `get_validation_transforms()` | Minimal transforms for validation (normalize only) |
| Array/tensor support | Works with both NumPy arrays and PyTorch tensors |

### 3.9 Physics-Constrained Loss (`physics_constrained_loss.py`)

| Feature | Description |
|---------|-------------|
| `PhysicsConstrainedPINNLoss` class | Multi-component loss enforcing lensing physics |
| **Poisson loss** | `||nabla^2 psi - 2 kappa||^2` via 5-point Laplacian stencil |
| **Gradient consistency** | `||alpha_pred - nabla psi||^2` via Sobel operator |
| **Mass conservation** | Penalizes negative or extremely large total convergence |
| **Parameter regularization** | L2 + soft bounds penalty for physical parameter ranges |
| **Classification loss** | Cross-entropy for DM type classification |
| Finite difference Laplacian | `Conv2d` with `[0,1,0; 1,-4,1; 0,1,0]` kernel |
| Sobel gradient | `Conv2d` with Sobel-x and Sobel-y kernels |
| `create_coordinate_grid()` | Meshgrid utility with `requires_grad=True` for autograd |
| `validate_poisson_equation()` | Post-hoc validation: reports max/mean/relative error |
| `validate_gradient_consistency()` | Post-hoc validation of `alpha = nabla psi` |

### 3.10 Unit-Safe Physics (`physics_unit_safe.py`)

| Feature | Description |
|---------|-------------|
| `compute_nfw_deflection_unit_safe()` | NFW deflection with full `astropy.units` dimensional analysis |
| 10-step pipeline | Constants -> input conversion -> distances -> coordinates -> Sigma_crit -> rho_s -> kappa_s -> x=r/r_s -> f(x) -> alpha(arcsec) |
| Runtime unit assertions | `assert G.unit == kpc^3/(Msun*s^2)` at every intermediate step |
| Wright & Brainerd (2000) | Follows ApJ 534, 34 for NFW deflection function |
| `test_unit_correctness()` | Self-test that validates dimensional correctness |

### 3.11 Dataset Generation (`generate_dataset.py`)

| Feature | Description |
|---------|-------------|
| `generate_convergence_map_vectorized()` | Fully vectorized convergence map generation (10-100x speedup) |
| `generate_convergence_map()` | Legacy wrapper (deprecated, delegates to vectorized) |
| `add_noise()` | Gaussian + Poisson noise model for realistic observations |
| `generate_single_sample()` | Single training sample: random cosmology + lens + noise |
| `generate_synthetic_convergence()` | API wrapper for convergence map generation |
| `generate_training_data()` | Full HDF5 dataset: train/val/test split, balanced CDM/WDM/SIDM |
| `LensDataset` class | PyTorch-compatible dataset for HDF5 data with channel dim |
| Random parameter ranges | `z_lens ~ U[0.3, 0.8]`, `M_vir ~ U[1e11, 5e12]`, `c ~ U[5, 15]` |

### 3.12 Model Evaluation (`evaluate.py`)

| Feature | Description |
|---------|-------------|
| `compute_metrics()` | MAE, RMSE, MAPE per parameter + classification accuracy |
| `evaluate_model()` | Full model evaluation on DataLoader with predictions |
| `plot_confusion_matrix()` | Annotated heatmap for CDM/WDM/SIDM classification |
| `plot_parameter_errors()` | Absolute + relative error histograms for each parameter |
| `plot_calibration_curve()` | Per-class probability calibration curves |
| `plot_parameter_scatter()` | Predicted vs. true scatter with R^2 annotation |
| `print_evaluation_summary()` | Formatted console report with all metrics |

### 3.13 Bayesian Uncertainty Quantification (`bayesian_uq.py`)

| Feature | Description |
|---------|-------------|
| MC Dropout | Multiple forward passes with dropout enabled for epistemic UQ |
| Deep Ensemble | N independent models trained, predictions averaged |
| Predictive entropy | `-sum p log p` for classification uncertainty |
| Mutual information | `H[y|x] - E[H[y|x, theta]]` for model disagreement |
| Calibration metrics | Expected Calibration Error (ECE) computation |

---

## 4. Time Delay & Cosmography

**Directory:** `src/time_delay/`

### 4.1 Cosmography (`cosmography.py`)

| Feature | Description |
|---------|-------------|
| `TimeDelayCosmography` class | H0 inference from lensed quasar time delays (Suyu et al. 2010) |
| Fermat potential | `phi(theta) = 0.5*(theta - beta)^2 - psi(theta)` |
| Time delay | `Delta t = (1+z_l)/c * D_l D_s / D_ls * Delta phi` |
| H0 posterior | Bayesian inference of Hubble constant from measured time delays |
| MCMC sampling | Posterior sampling for H0 with lens model uncertainties |
| Distance-redshift integration | `np.trapezoid` for comoving distance integrals (NumPy 2.0 compatible) |
| Cosmological parameters | Joint constraints on H0, Omega_m from time-delay distances |

### 4.2 LSTM Time Delay Estimator (`lstm_timedelay.py`)

| Feature | Description |
|---------|-------------|
| `LSTMTimeDelayEstimator` class | LSTM-FCNN for time delay from light curves (HOLISMOKES XII) |
| Bidirectional LSTM | Captures forward and backward temporal dependencies |
| Fully-connected head | Maps LSTM hidden states to time delay prediction |
| Light curve preprocessing | Normalization, resampling, gap handling |
| Uncertainty output | Predicts `(delta_t, sigma_t)` for each quasar pair |
| Multi-pair support | Handles doubly/quadruply lensed quasars with multiple delays |
| Training pipeline | End-to-end training with MSE + NLL loss |

---

## 5. Dark Matter Analysis

**Directory:** `src/dark_matter/`

### 5.1 Substructure Detection (`substructure.py`)

| Feature | Description |
|---------|-------------|
| `SubstructureDetector` class | Detection of dark matter subhalos in lensing residuals |
| Residual analysis | Subtracts smooth lens model to reveal substructure |
| Power spectrum | Fourier analysis of convergence residuals |
| Subhalo mass function | Expected `dN/dM ~ M^(-1.9)` CDM prediction (Dalal & Kochanek 2002) |
| Anomalous flux ratios | Detects substructure from image flux anomalies |
| Sensitivity maps | Maps minimum detectable subhalo mass across image plane |

---

## 6. Validation & Calibration

**Directory:** `src/validation/`

### 6.1 Scientific Validator (`scientific_validator.py`)

| Feature | Description |
|---------|-------------|
| `ScientificValidator` class | Comprehensive physics validation framework |
| Energy conservation | Verifies mass/energy conservation in lensing |
| Convergence bounds | Checks `kappa >= 0` (non-negative mass) |
| Magnification theorem | Validates `sum(signed magnifications) = 1` (odd-number theorem) |
| Profile-specific checks | NFW concentration-mass relation, WDM free-streaming, SIDM core size |
| Deflection symmetry | Validates radial symmetry for spherical lenses |
| Metric collection | Aggregates pass/fail/warning counts |

### 6.2 Synthetic Data Calibration (`calibration.py`)

| Feature | Description |
|---------|-------------|
| `SyntheticDataCalibrator` class | Calibrate synthetic data against real HST/JWST observations |
| Literature values database | Einstein Cross, Twin Quasar, Abell 2218 with published parameters |
| SIS analog generation | Synthetic `kappa(theta) = theta_E / (2|theta|)` for known systems |
| Einstein radius recovery | Radial profile analysis to find `kappa = 1` crossing |
| Mass estimation | Critical surface density + enclosed mass calculation |
| Holdout calibration | Cross-system calibration factors (not self-calibration) |
| Calibration table | Markdown report with errors for each system |
| JSON export | `save_calibration()` for persistent results |

### 6.3 HST Validation (`hst_targets.py`)

| Feature | Description |
|---------|-------------|
| `HSTTarget` dataclass | Target metadata: name, RA/Dec, redshifts, filter, dataset ID |
| `HSTValidation` class | Validation pipeline against real HST observations |
| Target catalog | Einstein Cross, Abell 1689, SDSS J1004+4112 |
| Chi-squared comparison | `chi^2 = sum((obs - sim)^2 / sigma^2)` with Poisson + readnoise |
| Reduced chi-squared | `chi^2_red = chi^2 / N_dof` with quality interpretation |
| Residual maps | Pixel-by-pixel `sim - obs` difference images |
| Report generation | Automated text report with quality assessment |
| MAST integration | Placeholder for HST archive data download |

### 6.4 Comparisons (`comparisons.py`)

| Feature | Description |
|---------|-------------|
| Benchmark against literature | Compare NFW profiles to Bartelmann (1996) tables |
| PyAutoLens comparison | Cross-validate with PyAutoLens results |
| Lenstronomy comparison | Cross-validate with lenstronomy results |
| Profile comparison plots | Side-by-side radial profile visualization |

---

## 7. Observational Data

**Directory:** `src/data/`

### 7.1 Real Data Loader (`real_data_loader.py`)

| Feature | Description |
|---------|-------------|
| `FITSDataLoader` class | Load and parse FITS files from HST/JWST |
| `ObservationMetadata` dataclass | Telescope, instrument, filter, exposure time, pixel scale, RA/Dec |
| Multi-extension FITS | `list_extensions()` for complex FITS files |
| Pixel scale extraction | CD matrix, CDELT, PIXSCALE keyword fallback chain |
| Instrument defaults | HST ACS (0.05"/px), WFC3/UVIS (0.04"/px), JWST NIRCam (0.031"/px) |
| `PSFModel` class | PSF generation with Gaussian, Moffat, and Airy disk models |
| Gaussian PSF | `exp(-r^2 / 2 sigma^2)` with FWHM-to-sigma conversion |
| Moffat PSF | `[1 + (r/alpha)^2]^(-beta)` with `beta = 4.765` |
| Airy disk PSF | `[2 J_1(x) / x]^2` diffraction-limited pattern |
| PSF convolution | FFT-based `fftconvolve` for efficiency |
| `preprocess_real_data()` | NaN handling (zero/median/interpolate), resize, normalize |
| `load_real_data()` | Convenience function: load + preprocess in one call |

---

## 8. Utilities

**Directory:** `src/utils/`

### 8.1 Reproducibility (`reproducibility.py`)

| Feature | Description |
|---------|-------------|
| `set_seed()` | Sets Python `random`, NumPy, PyTorch (CPU + CUDA) seeds |
| `get_random_state()` | Captures complete RNG state across all libraries |
| `set_random_state()` | Restores previously saved RNG state |
| `save_random_state()` | Pickle-based state persistence to file |
| `load_random_state()` | Load and restore state from file |
| `DeterministicContext` | Context manager: saves state, sets seed, restores on exit |
| `hash_config()` | SHA-256 deterministic hash of experiment configuration |
| `get_deterministic_hash()` | Hash arbitrary list of objects for experiment tracking |
| cuDNN determinism | Sets `cudnn.deterministic=True`, `benchmark=False` |

### 8.2 Visualization (`visualization.py`)

| Feature | Description |
|---------|-------------|
| `plot_lens_system()` | Complete lens visualization: convergence background, source, images, Einstein ring |
| Dark astronomy style | `plt.style.use('dark_background')` for publication plots |
| Image labeling | A, B, C, D labels with magnification-proportional marker size |
| Einstein radius overlay | Dashed circle at `theta_E` |
| Info box | Lens/source redshift and image count annotation |
| `plot_radial_profile()` | Log-log surface density and convergence vs. radius |
| `plot_deflection_field()` | Quiver plot of `(alpha_x, alpha_y)` with magnitude coloring |
| `plot_magnification_map()` | Magnification with SymLogNorm and critical curve contours |
| `plot_source_plane_mapping()` | Source plane grid distortion showing caustic structure |

### 8.3 Constants (`constants.py`)

| Feature | Description |
|---------|-------------|
| `G_CONST` | Gravitational constant (SI) |
| `C_LIGHT` | Speed of light (m/s) |
| `M_SUN_KG` | Solar mass (kg) |
| `ARCSEC_TO_RAD` | Arcsecond to radian conversion |
| `RAD_TO_ARCSEC` | Radian to arcsecond conversion |
| `KPC_TO_M` | Kiloparsec to meters |

---

## 9. REST API

**Directory:** `api/`

### 9.1 FastAPI Application (`main.py`)

| Feature | Description |
|---------|-------------|
| `/health` | Health check endpoint with database/model status |
| `/api/v1/analyze` | Submit lensing analysis job (convergence map generation) |
| `/api/v1/pinn/infer` | PINN model inference: parameter estimation + DM classification |
| `/api/v1/models` | List available trained models with metadata |
| `/api/v1/stats` | System statistics: total jobs, uptime, model count |
| `/api/v1/compare` | Compare results between different lens models |
| Async job queue | Background task processing with job ID tracking |
| Model caching | `MODEL_CACHE` for loaded PyTorch models |
| CORS middleware | Configurable cross-origin resource sharing |

### 9.2 Data Models (`models.py`)

| Feature | Description |
|---------|-------------|
| `AnalysisRequest` | Pydantic model: profile type, mass, scale radius, ellipticity, redshifts |
| `PINNInferenceRequest` | Image data + model selection for PINN inference |
| `AnalysisResponse` | Convergence map, metadata, execution time |
| `PINNInferenceResponse` | Predicted parameters, DM class probabilities, uncertainties |
| Input validation | Pydantic validators for physical parameter ranges |

### 9.3 Authentication (`auth.py`)

| Feature | Description |
|---------|-------------|
| API key authentication | Header-based API key validation |
| Rate limiting | Request rate limiting per API key |
| JWT tokens | JSON Web Token support for session management |

### 9.4 Error Handling (`error_handler.py`)

| Feature | Description |
|---------|-------------|
| Exception middleware | Global exception catching and JSON error responses |
| Validation errors | Pydantic validation error formatting |
| HTTP error codes | Proper 400/401/404/500 status code mapping |

---

## 10. Streamlit UI

**Directory:** `app/`

### 10.1 Home Page (`Home.py`)

| Feature | Description |
|---------|-------------|
| Dashboard overview | Project description, quick links, system status |
| Navigation | Multi-page Streamlit app with sidebar navigation |

### 10.2 Simple Lensing (`pages/02_Simple_Lensing.py`)

| Feature | Description |
|---------|-------------|
| Interactive lens simulation | Point mass and SIS lens equation visualization |
| Parameter sliders | Mass, source position, redshifts |
| Image position display | Computed image positions with magnifications |
| Convergence map | Real-time convergence map rendering |

### 10.3 PINN Inference (`pages/03_PINN_Inference.py`)

| Feature | Description |
|---------|-------------|
| Model inference UI | Upload or generate image, run PINN inference |
| Parameter display | Predicted M_vir, concentration, source position, H0 |
| DM classification | CDM/WDM/SIDM probability bar chart |
| Uncertainty visualization | Error bars from Bayesian UQ |

### 10.4 Results (`pages/03_Results.py`)

| Feature | Description |
|---------|-------------|
| Results viewer | Display saved analysis results |
| Comparison plots | Side-by-side parameter comparisons |

### 10.5 Multi-Plane Lensing (`pages/04_Multi_Plane.py`)

| Feature | Description |
|---------|-------------|
| Multi-plane setup | Configure multiple lens planes with different profiles |
| Recursive ray tracing | Visualization of multi-plane deflection |
| Intermediate planes | Display convergence at each lens plane |

### 10.6 Real Data (`pages/05_Real_Data.py`)

| Feature | Description |
|---------|-------------|
| FITS file upload | Load real HST/JWST observations |
| Data preprocessing | Noise removal, normalization, PSF deconvolution |
| Metadata display | Telescope, instrument, filter, pixel scale info |

### 10.7 Training (`pages/06_Training.py`)

| Feature | Description |
|---------|-------------|
| Model training UI | Configure and launch PINN training |
| Loss curves | Real-time training loss visualization |
| Hyperparameter config | Learning rate, batch size, epochs, architecture |
| Checkpoint management | Save/load model checkpoints |

### 10.8 Validation (`pages/07_Validation.py`)

| Feature | Description |
|---------|-------------|
| Scientific validation | Run validation suite with pass/fail reporting |
| Calibration UI | Calibrate against known lens systems |
| Comparison tables | Literature vs. recovered parameter tables |

### 10.9 Bayesian UQ (`pages/08_Bayesian_UQ.py`)

| Feature | Description |
|---------|-------------|
| Uncertainty visualization | MC Dropout and ensemble uncertainty maps |
| Epistemic vs. aleatoric | Decomposed uncertainty display |
| Calibration curves | Predicted vs. observed confidence |

### 10.10 Settings (`pages/09_Settings.py`)

| Feature | Description |
|---------|-------------|
| Configuration | API endpoint, model paths, compute device |
| Theme settings | Dark/light mode, color scheme |
| Cache management | Clear cached data and models |

### 10.11 UI Utilities (`app/utils/`)

| Feature | Description |
|---------|-------------|
| `helpers.py` | Common Streamlit helper functions |
| `plotting.py` | Streamlit-compatible plotting wrappers |
| `session_state.py` | Session state management |
| `demo_helpers.py` | Demo data generation for UI testing |
| `ui.py` | Reusable UI components (cards, metrics, etc.) |
| `styles.py` | Custom CSS styling for the Streamlit app |

---

## 11. Test Suite

**Directory:** `tests/`

| Test File | Coverage |
|-----------|----------|
| `test_lens_models.py` | NFW, PointMass, WDM, SIDM convergence, deflection, Einstein radius |
| `test_ray_tracing.py` | Image finding, magnification, critical curves |
| `test_wave_optics.py` | Fresnel integral, interference patterns |
| `test_pinn.py` | Forward pass, loss computation, parameter shapes |
| `test_time_delay.py` | Time delay computation, H0 recovery, LSTM predictions |
| `test_dark_matter.py` | Substructure detection, power spectrum |
| `test_scientific_validation.py` | Validator checks, convergence bounds, magnification theorem |
| `test_api.py` | API endpoints, request/response validation, error handling |
| `test_security.py` | Authentication, rate limiting, injection prevention |
| `test_integration.py` | End-to-end pipeline tests |

---

## 12. Research References

### Core Gravitational Lensing
| Reference | Used In |
|-----------|---------|
| Schneider, Ehlers & Falco (1992), "Gravitational Lenses" | Ray tracing, thin lens formalism, Poisson equation |
| Bartelmann & Schneider (2001), Phys. Rep. 340, 291 | Convergence, shear, magnification theory |
| Misner, Thorne & Wheeler (1973), "Gravitation" | Schwarzschild geodesics, GR module |

### Mass Profiles
| Reference | Used In |
|-----------|---------|
| Navarro, Frenk & White (1997), ApJ 490, 493 | NFW profile |
| Wright & Brainerd (2000), ApJ 534, 34 | NFW deflection function |
| Golse & Kneib (2002), A&A 390, 781 | Elliptical NFW |
| Viel et al. (2005), Phys. Rev. D 71, 063534 | WDM transfer function |

### Machine Learning
| Reference | Used In |
|-----------|---------|
| Raissi et al. (2019), J. Comp. Phys. 378, 686 | PINN methodology |
| Lu et al. (2021), Nat. Mach. Intell. 3, 218 | Physics-constrained ML |
| Barco et al. (2025) | Score-based blind lens inversion |
| Venkatraman et al. (2025) | Neural Posterior Estimation for lensing |
| Silver et al. (2025) | JWST low-mass lens detection |

### Wave Optics
| Reference | Used In |
|-----------|---------|
| Shi (2024), MNRAS 534, 3269 | Lefschetz thimble method |
| Yarimoto & Oguri (2024), PRD 110, 103506 | Born approximation corrections |
| Yeung et al. (2024), arXiv:2410.19804 | wolensing GPU wave optics |

### Time Delays & Cosmography
| Reference | Used In |
|-----------|---------|
| Suyu et al. (2010), ApJ 711, 201 | Time-delay cosmography, H0 inference |
| Huber & Suyu (2024), HOLISMOKES XII | LSTM time delay estimation |
| Kundic et al. (1997), ApJ 482, 75 | Twin Quasar time delay measurement |

### Dark Matter
| Reference | Used In |
|-----------|---------|
| Dalal & Kochanek (2002), ApJ 572, 25 | Substructure detection, flux anomalies |
| Springel et al. (2008), MNRAS 391, 1685 | Composite galaxy models |

### Calibration & Validation
| Reference | Used In |
|-----------|---------|
| Schmidt et al. (1998), ApJ 507, 46 | Einstein Cross parameters |
| Kneib et al. (1996), ApJ 471, 643 | Abell 2218 cluster lensing |
| Huchra et al. (1985) | Twin Quasar discovery |
| Kochanek et al. (2006) | CASTLES gravitational lens database |
| Treu & Marshall (2016) | Strong lensing review |

---

> **Total:** 200+ distinct features across 12 major components, 45+ source files, implementing 7 mass profile types, 4 ray-tracing methods, 7 ML architectures, 3 domain adaptation techniques, 2 time-delay estimation methods, and a full validation/calibration pipeline.
