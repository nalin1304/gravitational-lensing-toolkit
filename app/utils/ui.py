"""
UI Components and Styling Utilities

Provides reusable UI components for Streamlit application including:
- Custom CSS injection
- Header rendering
- Card components
- Progress indicators
- Styled messages

Author: Refactored from app/styles.py
"""

import streamlit as st
from typing import Optional

# Custom CSS styles
CUSTOM_CSS = """
<style>
    /* ========================================
       GLOBAL VARIABLES & THEME
       ======================================== */
    :root {
        --primary-blue: #667eea;
        --secondary-purple: #764ba2;
        --success-green: #10b981;
        --warning-yellow: #f59e0b;
        --danger-red: #ef4444;
        --info-cyan: #06b6d4;
        
        --bg-dark: #0e1117;
        --bg-card: #1e2130;
        --bg-hover: #262837;
        
        --text-primary: #ffffff;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        
        --border-color: rgba(102, 126, 234, 0.2);
        --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
        --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.3);
        --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.4);
    }
    
    /* ========================================
       MAIN LAYOUT
       ======================================== */
    .main {
        padding: 1rem 2rem;
    }
    
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Better spacing for sections */
    .stMarkdown {
        margin-bottom: 1rem;
    }
    
    /* Cleaner headers */
    h1, h2, h3 {
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Better button spacing */
    .stButton {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Expander improvements */
    .streamlit-expanderHeader {
        background-color: var(--bg-card) !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: var(--bg-hover) !important;
    }
    
    /* Info boxes - less intrusive */
    .stInfo, .stWarning, .stSuccess, .stError {
        padding: 0.75rem 1rem !important;
        border-radius: 8px !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Metrics improvements */
    [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        border-radius: 8px !important;
        background-color: var(--bg-card) !important;
    }
    
    /* ========================================
       HEADER COMPONENTS
       ======================================== */
    .app-header {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-purple) 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }
    
    .app-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.3;
        z-index: 0;
    }
    
    .app-header-content {
        position: relative;
        z-index: 1;
    }
    
    .app-header h1 {
        color: white !important;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        letter-spacing: -0.5px;
    }
    
    .app-header-subtitle {
        color: rgba(255, 255, 255, 0.95) !important;
        font-size: 1.15rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    .app-header-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 0.375rem 0.875rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        color: white;
        margin-top: 1rem;
        backdrop-filter: blur(10px);
    }
    
    /* ========================================
       CARDS & CONTAINERS
       ======================================== */
    .custom-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
        border-color: var(--primary-blue);
    }
    
    .card-header {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .card-body {
        color: var(--text-secondary);
        line-height: 1.6;
    }
    
    /* ========================================
       METRIC CARDS
       ======================================== */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid var(--border-color);
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: var(--shadow-sm);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        border-color: var(--primary-blue);
    }
    
    div[data-testid="metric-container"] label {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }
    
    /* Additional styles for buttons, sidebars, inputs, etc. */
    /* (Full CSS available in original styles.py) */
    
    /* ========================================
       ANIMATIONS
       ======================================== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fadeIn {
        animation: fadeIn 0.5s ease-out;
    }
</style>
"""


def inject_custom_css() -> None:
    """
    Inject custom CSS into Streamlit app.
    
    Call this function at the top of every page to apply consistent styling.
    
    Example:
        >>> from app.utils.ui import inject_custom_css
        >>> inject_custom_css()
    """
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str = "", badge: str = "") -> None:
    """
    Render professional application header with gradient background.
    
    Args:
        title: Main header title
        subtitle: Optional subtitle text
        badge: Optional badge text (e.g., "Beta", "v2.0")
    
    Example:
        >>> render_header(
        ...     title="Gravitational Lensing Simulator",
        ...     subtitle="Physics-Informed Neural Network Analysis",
        ...     badge="v2.0"
        ... )
    """
    header_html = f"""
    <div class="app-header animate-fadeIn">
        <div class="app-header-content">
            <h1>🔭 {title}</h1>
            {f'<p class="app-header-subtitle">{subtitle}</p>' if subtitle else ''}
            {f'<span class="app-header-badge">{badge}</span>' if badge else ''}
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_card(title: str, content: str, icon: str = "📊") -> None:
    """
    Render custom card component with title, content, and icon.
    
    Args:
        title: Card title
        content: Card content (supports HTML)
        icon: Emoji icon for card header (default: "📊")
    
    Example:
        >>> render_card(
        ...     title="Convergence Map",
        ...     content="<p>Generated from NFW profile with M=1e14 M_sun</p>",
        ...     icon="🌌"
        ... )
    """
    card_html = f"""
    <div class="custom-card animate-fadeIn">
        <div class="card-header">
            <span>{icon}</span>
            <span>{title}</span>
        </div>
        <div class="card-body">
            {content}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def show_success(message: str) -> None:
    """Show success message with consistent styling."""
    st.success(f"✅ {message}")


def show_warning(message: str) -> None:
    """Show warning message with consistent styling."""
    st.warning(f"⚠️ {message}")


def show_info(message: str) -> None:
    """Show info message with consistent styling."""
    st.info(f"ℹ️ {message}")


def show_error(message: str) -> None:
    """Show error message with consistent styling."""
    st.error(f"❌ {message}")


def create_download_button(
    data,
    filename: str,
    button_text: str = "📥 Download",
    mime_type: str = "text/plain"
) -> None:
    """
    Create a styled download button.
    
    Args:
        data: Data to download (bytes, str, or file-like object)
        filename: Name of the downloaded file
        button_text: Button label text (default: "📥 Download")
        mime_type: MIME type of the file (default: "text/plain")
    
    Example:
        >>> import json
        >>> data = json.dumps({"param1": 1.5, "param2": 2.0})
        >>> create_download_button(data, "parameters.json", "Download Parameters", "application/json")
    """
    try:
        st.download_button(
            label=button_text,
            data=data,
            file_name=filename,
            mime=mime_type
        )
    except Exception as e:
        show_error(f"Failed to create download button: {str(e)}")


def create_parameter_summary(params: dict) -> str:
    """
    Create a formatted Markdown summary of parameters.
    
    Args:
        params: Dictionary of parameters to summarize
    
    Returns:
        Formatted Markdown string
    
    Example:
        >>> params = {"M_vir": 1e14, "r_s": 150.0, "z_lens": 0.5}
        >>> summary = create_parameter_summary(params)
        >>> st.markdown(summary)
    """
    summary = "### 📋 Parameter Summary\n\n"
    for key, value in params.items():
        if isinstance(value, float):
            summary += f"- **{key.replace('_', ' ').title()}**: {value:.4f}\n"
        else:
            summary += f"- **{key.replace('_', ' ').title()}**: {value}\n"
    return summary


def render_footer(version: str = "2.0.0") -> None:
    """
    Render application footer with version and links.
    
    Args:
        version: Application version string
    
    Example:
        >>> render_footer("2.0.0")
    """
    footer_html = f"""
    <div style="
        margin-top: 4rem;
        padding: 2rem 0;
        border-top: 1px solid rgba(102, 126, 234, 0.2);
        text-align: center;
        color: var(--text-muted, #64748b);
        font-size: 0.875rem;
    ">
        <p style="margin: 0;">
            <strong>Gravitational Lensing Analysis Platform</strong> v{version}
        </p>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">
            Built with ❤️ using Streamlit & PyTorch | 
            <a href="https://github.com/your-repo/gravitational-lensing-toolkit" 
               style="color: #6366f1; text-decoration: none;">GitHub</a>
        </p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)


def render_sidebar_nav() -> None:
    """
    Render enhanced sidebar navigation.
    
    Example:
        >>> render_sidebar_nav()
    """
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <span style="font-size: 2.5rem;">🔭</span>
            <h2 style="margin: 0.5rem 0 0 0; font-size: 1.2rem; 
                       background: linear-gradient(135deg, #6366f1, #ec4899); 
                       -webkit-background-clip: text; 
                       -webkit-text-fill-color: transparent;">
                Lensing Platform
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick links
        st.markdown("### 🚀 Quick Actions")
        st.markdown("""
        <div style="font-size: 0.9rem; color: var(--text-secondary, #94a3b8);">
            <p>• Run a demo from Home</p>
            <p>• Upload FITS in Real Data</p>
            <p>• Train models in Training</p>
        </div>
        """, unsafe_allow_html=True)
