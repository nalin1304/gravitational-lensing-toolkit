"""
🌟 ULTRA-MODERN STYLING FOR STREAMLIT APPLICATION 🌟
Featuring: Animations, Particle Effects, Glassmorphism, Smooth Transitions
100x Better UI with Professional Design
"""

import streamlit as st

CUSTOM_CSS = """
<style>
    /* ========================================
       🎨 GLOBAL VARIABLES & THEME
       ======================================== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800&display=swap');

    :root {
        --primary-blue: #6366f1; /* Indigo 500 */
        --primary-glow: rgba(99, 102, 241, 0.5);
        --secondary-purple: #8b5cf6; /* Violet 500 */
        --accent-pink: #ec4899; /* Pink 500 */
        --accent-cyan: #06b6d4; /* Cyan 500 */
        
        --success-green: #10b981;
        --warning-yellow: #f59e0b;
        --danger-red: #ef4444;
        
        --bg-dark: #0f172a; /* Slate 900 */
        --bg-card: rgba(30, 41, 59, 0.7); /* Slate 800 */
        --bg-glass: rgba(15, 23, 42, 0.6);
        --bg-hover: rgba(51, 65, 85, 0.5);
        
        --text-primary: #f8fafc; /* Slate 50 */
        --text-secondary: #cbd5e1; /* Slate 300 */
        --text-muted: #94a3b8; /* Slate 400 */
        
        --border-color: rgba(148, 163, 184, 0.2);
        --border-highlight: rgba(99, 102, 241, 0.5);
        
        --shadow-sm: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-md: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        --shadow-glow: 0 0 20px var(--primary-glow);
        
        --gradient-cosmic: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        --gradient-surface: linear-gradient(180deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%);
    }

    /* ========================================
       🌌 BASE STYLES & TYPOGRAPHY
       ======================================== */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
        background-color: var(--bg-dark);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.025em;
    }
    
    /* Background Animation */
    .stApp {
        background: 
            radial-gradient(circle at 15% 50%, rgba(99, 102, 241, 0.15) 0%, transparent 25%),
            radial-gradient(circle at 85% 30%, rgba(236, 72, 153, 0.15) 0%, transparent 25%),
            var(--bg-dark);
        background-attachment: fixed;
    }
    
    /* Hide Streamlit Chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ========================================
       ✨ ANIMATIONS
       ======================================== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 10px var(--primary-glow); }
        50% { box-shadow: 0 0 20px var(--primary-glow), 0 0 10px var(--primary-blue); }
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ========================================
       🚀 COMPONENT STYLING
       ======================================== */
    
    /* Buttons */
    .stButton button {
        background: var(--gradient-cosmic);
        background-size: 200% 200%;
        border: none;
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        transition: all 0.3s ease;
        animation: gradient-shift 5s ease infinite;
        box-shadow: var(--shadow-md);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg), var(--shadow-glow);
    }
    
    .stButton button:active {
        transform: translateY(0);
    }

    /* Secondary Button */
    button[kind="secondary"] {
        background: transparent;
        border: 1px solid var(--border-color);
        color: var(--text-secondary);
    }
    
    button[kind="secondary"]:hover {
        border-color: var(--primary-blue);
        color: var(--primary-blue);
    }

    /* Cards (Metrics) */
    div[data-testid="metric-container"] {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-color: var(--border-highlight);
        box-shadow: var(--shadow-glow);
    }
    
    div[data-testid="metric-container"] label {
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        background: var(--gradient-cosmic);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        font-weight: 800;
    }

    /* Inputs */
    .stTextInput input, .stNumberInput input, .stSelectbox select, .stBoxLayout {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: var(--primary-blue) !important;
        box-shadow: 0 0 0 2px var(--primary-glow) !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] h1 {
        background: var(--gradient-cosmic);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
        margin-bottom: 2rem;
    }

    /* Custom Header Container */
    .app-header {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        animation: fadeIn 0.8s ease-out;
    }
    
    .app-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 4px;
        background: var(--gradient-cosmic);
    }
    
    .app-header h1 {
        margin: 0;
        font-size: 2.5rem;
        color: var(--text-primary);
    }
    
    .app-header p {
        color: var(--text-secondary);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Custom Card */
    .custom-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        animation: fadeIn 0.8s ease-out;
        height: 100%;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        border-color: var(--border-highlight);
        box-shadow: var(--shadow-md);
    }
    
    .card-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        display: inline-block;
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: var(--text-primary);
    }
    
    .card-content {
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Progress Info */
    .stInfo {
        background-color: rgba(6, 182, 212, 0.1) !important;
        border: 1px solid rgba(6, 182, 212, 0.3) !important;
        color: var(--text-primary) !important;
    }
    
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
    }
    
    .stWarning {
        background-color: rgba(245, 158, 11, 0.1) !important;
        border: 1px solid rgba(245, 158, 11, 0.3) !important;
    }
    
    .stError {
        background-color: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
    }

</style>
"""


def inject_custom_css():
    """Inject custom CSS into Streamlit app"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str = "", badge: str = ""):
    """Render professional application header"""
    badge_html = f'<span style="background: var(--primary-blue); padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.8rem; vertical-align: middle; margin-left: 1rem;">{badge}</span>' if badge else ""
    
    header_html = f"""
    <div class="app-header">
        <h1>{title}{badge_html}</h1>
        {f'<p>{subtitle}</p>' if subtitle else ''}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_card(title: str, content: str, icon: str = "📊", key: str = None):
    """Render custom card component"""
    card_html = f"""
    <div class="custom-card">
        <div class="card-icon">{icon}</div>
        <div class="card-title">{title}</div>
        <div class="card-content">{content}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
