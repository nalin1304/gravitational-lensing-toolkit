# Gravitational Lensing Frontend

A professional Streamlit-based frontend for gravitational lensing analysis with physics-informed neural networks.

## Features

### 🏠 Home Page
- Overview of the platform
- Quick start guide
- System status dashboard
- Feature highlights

### 🔬 Lens Model Builder
- **Lens Models Supported:**
  - Point Mass
  - NFW (Navarro-Frenk-White)
  - Sersic
  - Composite (DM + Baryons)
  
- **Real-time Parameters:**
  - Cosmological parameters (z_lens, z_source, H0, Ωₘ)
  - Model-specific parameters with interactive sliders
  - Subhalo inclusion options

- **Live Visualizations:**
  - Convergence maps (κ)
  - Deflection angle fields
  - 3D lensing potential surfaces
  
### 📊 Visualizations Page
- Interactive convergence maps
- Deflection angle vector fields
- Lensing potential heatmaps
- Critical curves and caustics
- Multi-model comparison plots

### 🌊 Wave Optics
- Wave vs geometric optics comparison
- Interactive wavelength slider (100-2000 nm)
- Interference pattern detection
- Einstein ring formation animation
- Fringe analysis and statistics

### 🧠 PINN Training
- Training data upload (HDF5 format)
- Hyperparameter configuration:
  - Learning rate
  - Batch size
  - Epochs
  - Physics loss weight (λ)
  - Dropout rate
- Real-time training monitor with loss curves
- Model inference testing

### ✅ Validation Tests
- Run test suite (unit, integration, physics)
- View test results with detailed breakdown
- Compare with literature references
- Generate validation reports

## Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the Streamlit app
streamlit run app.py

# Or with custom port
streamlit run app.py --server.port 8501
```

The application will be available at `http://localhost:8501`

## Project Structure

```
frontend/
├── __init__.py          # Package initialization with exports
├── app.py               # Main Streamlit application
├── components.py        # Reusable UI components
├── utils.py             # Helper functions and utilities
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Key Components

### Custom CSS Styling
- Dark theme with gradient backgrounds
- Animated cards and hover effects
- Custom progress bars and metrics
- Responsive layout

### Session State Management
Persistent data across page navigation:
- Current lens model
- Convergence grids
- Training history
- Test results

### Cached Computations
Expensive calculations are cached using `@st.cache_data`:
- Lens model creation
- Convergence map computation
- Deflection field calculation
- Wave optics simulations

## Pages

### 1. Home
Entry point with feature overview and quick start guide.

### 2. Lens Model Builder
Create and configure lens models with:
- Cosmology settings
- Model type selection
- Real-time parameter adjustment
- 3D potential visualization

### 3. Visualizations
Generate and explore:
- κ maps with custom colormaps
- Deflection fields
- Critical curves and caustics
- Model comparisons

### 4. Wave Optics
Compare wave and geometric optics:
- Wavelength-dependent effects
- Interference fringes
- Einstein ring animations
- Fringe statistics

### 5. PINN Training
Train physics-informed neural networks:
- Upload training data
- Configure hyperparameters
- Monitor training progress
- Test predictions

### 6. Validation Tests
Run and analyze test suite:
- Execute tests by category
- View results and statistics
- Compare with literature
- Generate reports

## Integration with Backend

The frontend integrates with:
- `src.lens_models` - Mass profile classes
- `src.optics` - Ray tracing and wave optics
- `src.ml` - PINN models
- `tests/` - Validation test suite

## Customization

### Adding New Lens Models

1. Update `get_lens_model()` in `utils.py`
2. Add form fields in `components.py`
3. Update model selection in `app.py`

### Adding New Visualizations

1. Create computation function in `utils.py`
2. Add visualization option in `app.py`
3. Create Plotly figure in the visualization function

### Modifying Styling

Edit `apply_custom_css()` in `components.py`:
- Colors
- Fonts
- Layouts
- Animations

## Performance Tips

1. **Use caching**: `@st.cache_data` for expensive computations
2. **Optimize grid size**: Smaller grids for development
3. **GPU acceleration**: Enable CUDA for PINN training
4. **Session state**: Store frequently accessed data

## Troubleshooting

### Common Issues

**Import errors:**
```bash
# Ensure you're in the project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**Plotly not displaying:**
```bash
# Update plotly
pip install --upgrade plotly
```

**Slow performance:**
- Reduce grid size in settings
- Enable caching
- Use smaller batch sizes for training

## License

This project is part of the Gravitational Lensing Analysis Platform.

## Support

For issues and feature requests, please refer to the main project documentation.
