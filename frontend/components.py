"""
Reusable UI Components for Gravitational Lensing Frontend

This module provides reusable Streamlit components for building
a professional gravitational lensing analysis interface.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown(
        """
    <style>
        /* Main app styling */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #f8fafc !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #1e293b;
        }
        
        /* Cards */
        .card {
            background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid rgba(102, 126, 234, 0.2);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }
        
        /* Info boxes */
        .info-box {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%);
            border-left: 4px solid #3b82f6;
            border-radius: 0 8px 8px 0;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* Warning boxes */
        .warning-box {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%);
            border-left: 4px solid #f59e0b;
            border-radius: 0 8px 8px 0;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* Success boxes */
        .success-box {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
            border-left: 4px solid #10b981;
            border-radius: 0 8px 8px 0;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        /* Sliders */
        .stSlider > div > div > div {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Select boxes */
        .stSelectbox > div > div {
            background-color: #1e293b;
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 8px;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #1e293b;
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
            border: 1px solid rgba(102, 126, 234, 0.2);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        /* Metrics */
        .stMetric {
            background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid rgba(102, 126, 234, 0.2);
        }
        
        .stMetric label {
            color: #94a3b8;
        }
        
        .stMetric div {
            color: #f8fafc;
            font-weight: 600;
        }
        
        /* Progress bars */
        .stProgress > div > div {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: #1e293b;
            border-radius: 8px;
            border: 1px solid rgba(102, 126, 234, 0.2);
        }
        
        /* File uploader */
        .stFileUploader > div > div {
            background-color: #1e293b;
            border: 2px dashed rgba(102, 126, 234, 0.4);
            border-radius: 12px;
        }
        
        /* Dataframe */
        .stDataFrame {
            background-color: #1e293b;
            border-radius: 8px;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #0f172a;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 4px;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .fade-in {
            animation: fadeIn 0.5s ease-out;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str, badge: str = ""):
    """
    Render a professional page header.

    Parameters
    ----------
    title : str
        Main page title
    subtitle : str
        Subtitle description
    badge : str, optional
        Badge text to display
    """
    # Build badge HTML
    if badge:
        badge_html = f'<span style="display: inline-block; background: rgba(255, 255, 255, 0.2); padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.875rem; margin-top: 0.5rem; color: white;">{badge}</span>'
    else:
        badge_html = ""

    st.markdown(
        f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    ">
        <h1 style="
            color: white;
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        ">{title}</h1>
        <p style="
            color: rgba(255, 255, 255, 0.9);
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
        ">{subtitle}</p>
        {badge_html}
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_card(title: str, description: str, icon: str = ""):
    """
    Render a styled card component.

    Parameters
    ----------
    title : str
        Card title
    description : str
        Card description text
    icon : str, optional
        Emoji icon to display
    """
    st.markdown(
        f"""
    <div class="card fade-in">
        <h4 style="margin: 0 0 0.5rem 0; color: #f8fafc;">
            {f'<span style="margin-right: 0.5rem;">{icon}</span>' if icon else ""}{title}
        </h4>
        <p style="margin: 0; color: #94a3b8; line-height: 1.5;">{description}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_info_box(message: str):
    """Render an info box with the given message."""
    st.markdown(
        f"""
    <div class="info-box">
        <p style="margin: 0; color: #60a5fa;">ℹ️ {message}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_warning_box(message: str):
    """Render a warning box with the given message."""
    st.markdown(
        f"""
    <div class="warning-box">
        <p style="margin: 0; color: #fbbf24;">⚠️ {message}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_success_box(message: str):
    """Render a success box with the given message."""
    st.markdown(
        f"""
    <div class="success-box">
        <p style="margin: 0; color: #34d399;">✅ {message}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def sidebar_navigation():
    """
    Render sidebar navigation menu.

    Returns
    -------
    str
        Selected page name
    """
    with st.sidebar:
        st.markdown(
            """
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: white; margin: 0;">🔭 LensLab</h2>
            <p style="color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 0.875rem;">
                Gravitational Lensing Analysis
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        page = st.radio(
            "Navigation",
            [
                "Home",
                "Lens Model Builder",
                "Visualizations",
                "Wave Optics",
                "PINN Training",
                "Validation Tests",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Quick settings in sidebar
        st.markdown("### ⚙️ Quick Settings")

        theme = st.selectbox(
            "Theme", ["Dark (Default)", "Light"], index=0, label_visibility="collapsed"
        )

        st.markdown("---")

        # Footer
        st.markdown(
            """
        <div style="text-align: center; padding: 1rem 0; color: #64748b; font-size: 0.75rem;">
            <p>v2.0.0 • Production Ready</p>
            <p>Built with Streamlit & PyTorch</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    return page


def lens_model_form(model_type: str):
    """
    Render a form for configuring lens model parameters.

    Parameters
    ----------
    model_type : str
        Type of lens model

    Returns
    -------
    dict
        Dictionary of parameter values
    """
    params = {}

    if model_type == "Point Mass":
        params["mass"] = st.number_input(
            "Mass (M☉)", min_value=1e10, max_value=1e15, value=1e12, format="%.2e"
        )

    elif model_type == "NFW":
        col1, col2 = st.columns(2)
        with col1:
            params["M_vir"] = st.number_input(
                "Virial Mass (M☉)",
                min_value=1e11,
                max_value=1e15,
                value=1e13,
                format="%.2e",
            )
            params["concentration"] = st.slider("Concentration", 1.0, 20.0, 5.0, 0.5)

        with col2:
            params["ellipticity"] = st.slider("Ellipticity", 0.0, 0.9, 0.0, 0.05)
            params["include_subhalos"] = st.checkbox("Include Subhalos", False)

    elif model_type == "Sersic":
        col1, col2 = st.columns(2)
        with col1:
            params["mass"] = st.number_input(
                "Total Mass (M☉)",
                min_value=1e9,
                max_value=1e13,
                value=1e11,
                format="%.2e",
            )
            params["n_sersic"] = st.slider("Sersic Index", 0.5, 8.0, 4.0, 0.1)

        with col2:
            params["R_eff"] = st.slider("Effective Radius (kpc)", 0.1, 50.0, 5.0, 0.5)

    return params


def training_monitor():
    """Render a real-time training monitor with loss curves."""
    # Create placeholder for dynamic updates
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Total Loss")
        chart_placeholder_1 = st.empty()

    with col2:
        st.markdown("#### Component Losses")
        chart_placeholder_2 = st.empty()

    # Initialize loss history
    if "loss_history" not in st.session_state:
        st.session_state["loss_history"] = {
            "total": [],
            "mse": [],
            "ce": [],
            "physics": [],
        }

    # Create plots
    epochs = list(range(len(st.session_state["loss_history"]["total"])))

    # Total loss plot
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=epochs,
            y=st.session_state["loss_history"]["total"],
            mode="lines",
            name="Total Loss",
            line=dict(color="#667eea", width=2),
        )
    )
    fig1.update_layout(
        height=300,
        xaxis_title="Epoch",
        yaxis_title="Loss",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    chart_placeholder_1.plotly_chart(fig1, use_container_width=True)

    # Component losses plot
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=epochs,
            y=st.session_state["loss_history"]["mse"],
            mode="lines",
            name="MSE",
            line=dict(color="#10b981", width=2),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=epochs,
            y=st.session_state["loss_history"]["ce"],
            mode="lines",
            name="Cross-Entropy",
            line=dict(color="#f59e0b", width=2),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=epochs,
            y=st.session_state["loss_history"]["physics"],
            mode="lines",
            name="Physics",
            line=dict(color="#3b82f6", width=2),
        )
    )
    fig2.update_layout(
        height=300,
        xaxis_title="Epoch",
        yaxis_title="Loss",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
    )
    chart_placeholder_2.plotly_chart(fig2, use_container_width=True)


def plot_3d_surface(lens_model, grid_size: int = 100, grid_extent: float = 5.0):
    """
    Create a 3D surface plot of the lensing potential.

    Parameters
    ----------
    lens_model : MassProfile
        The lens model
    grid_size : int
        Grid resolution
    grid_extent : float
        Physical extent in arcseconds

    Returns
    -------
    plotly.graph_objects.Figure
        3D surface plot
    """
    # Create grid
    x = np.linspace(-grid_extent, grid_extent, grid_size)
    y = np.linspace(-grid_extent, grid_extent, grid_size)
    xx, yy = np.meshgrid(x, y)

    # Compute potential
    psi = lens_model.lensing_potential(xx.ravel(), yy.ravel())
    psi = psi.reshape(xx.shape)

    # Create 3D surface
    fig = go.Figure(
        data=[
            go.Surface(
                x=xx,
                y=yy,
                z=psi,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="ψ (arcsec²)"),
            )
        ]
    )

    fig.update_layout(
        title="Lensing Potential ψ(θ)",
        scene=dict(
            xaxis_title="θ_x (arcsec)",
            yaxis_title="θ_y (arcsec)",
            zaxis_title="ψ (arcsec²)",
            bgcolor="rgba(0,0,0,0)",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=600,
    )

    return fig


def plot_comparison(
    model_names: list, lens_system, grid_size: int = 100, grid_extent: float = 5.0
):
    """
    Create a comparison plot of multiple lens models.

    Parameters
    ----------
    model_names : list
        List of model names to compare
    lens_system : LensSystem
        The lens system
    grid_size : int
        Grid resolution
    grid_extent : float
        Physical extent in arcseconds

    Returns
    -------
    plotly.graph_objects.Figure
        Comparison plot
    """
    from src.lens_models.mass_profiles import PointMassProfile, NFWProfile

    # Create grid
    x = np.linspace(-grid_extent, grid_extent, grid_size)
    y = np.linspace(-grid_extent, grid_extent, grid_size)
    xx, yy = np.meshgrid(x, y)

    # Create models
    models = {}
    if "Point Mass" in model_names:
        models["Point Mass"] = PointMassProfile(1e12, lens_system)
    if "NFW" in model_names:
        models["NFW"] = NFWProfile(1e13, 5, lens_system)

    # Create subplots
    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=list(models.keys()),
        specs=[[{"type": "heatmap"}] * n_cols for _ in range(n_rows)],
    )

    # Add each model
    for idx, (name, model) in enumerate(models.items()):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        kappa = model.convergence(xx.ravel(), yy.ravel())
        kappa = kappa.reshape(xx.shape)

        fig.add_trace(
            go.Heatmap(
                z=kappa, x=x, y=y, colorscale="Hot", showscale=(idx == 0), name=name
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        height=400 * n_rows,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )

    return fig
