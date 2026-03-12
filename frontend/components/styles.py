# ================================================================
# frontend/components/styles.py — Custom CSS Theme & Styling
# ================================================================

MAIN_CSS = """
<style>
/* ===== IMPORTS ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ===== ROOT VARIABLES ===== */
:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-card: rgba(17, 24, 39, 0.7);
    --bg-card-hover: rgba(17, 24, 39, 0.9);
    --border-color: rgba(99, 102, 241, 0.15);
    --border-glow: rgba(99, 102, 241, 0.3);
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-blue: #6366f1;
    --accent-cyan: #22d3ee;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --accent-orange: #f59e0b;
    --accent-purple: #a855f7;
    --gradient-primary: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
    --gradient-green: linear-gradient(135deg, #10b981 0%, #34d399 100%);
    --gradient-red: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
    --gradient-blue: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.4);
    --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.5);
    --shadow-glow: 0 0 20px rgba(99, 102, 241, 0.15);
}

/* ===== GLOBAL ===== */
.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ===== HIDE DEFAULT STREAMLIT ELEMENTS ===== */
#MainMenu {visibility: hidden;}
.stAppHeader {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar {width: 6px; height: 6px;}
::-webkit-scrollbar-track {background: var(--bg-primary);}
::-webkit-scrollbar-thumb {background: var(--accent-blue); border-radius: 3px;}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #111827 100%) !important;
    border-right: 1px solid var(--border-color) !important;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown label,
section[data-testid="stSidebar"] label {
    color: var(--text-secondary) !important;
    font-weight: 500;
}

section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(99, 102, 241, 0.08) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
}

section[data-testid="stSidebar"] .stTextInput input:focus,
section[data-testid="stSidebar"] .stSelectbox > div > div:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
}

section[data-testid="stSidebar"] .stSlider > div > div > div {
    background: var(--accent-blue) !important;
}

section[data-testid="stSidebar"] hr {
    border-color: var(--border-color) !important;
    margin: 0.8rem 0 !important;
}

/* ===== METRIC CARDS ===== */
.metric-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--gradient-primary);
    border-radius: 16px 16px 0 0;
}

.metric-card:hover {
    border-color: var(--border-glow);
    box-shadow: var(--shadow-glow);
    transform: translateY(-2px);
}

.metric-card .metric-label {
    color: var(--text-muted);
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}

.metric-card .metric-value {
    color: var(--text-primary);
    font-size: 1.65rem;
    font-weight: 700;
    line-height: 1.2;
}

.metric-card .metric-delta {
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 0.3rem;
    display: flex;
    align-items: center;
    gap: 4px;
}

.metric-card .metric-delta.positive {color: var(--accent-green);}
.metric-card .metric-delta.negative {color: var(--accent-red);}
.metric-card .metric-delta.neutral {color: var(--accent-orange);}

/* ===== SIGNAL BADGES ===== */
.signal-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 16px;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.signal-buy {
    background: rgba(16, 185, 129, 0.15);
    color: var(--accent-green);
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.signal-sell {
    background: rgba(239, 68, 68, 0.15);
    color: var(--accent-red);
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.signal-hold {
    background: rgba(245, 158, 11, 0.15);
    color: var(--accent-orange);
    border: 1px solid rgba(245, 158, 11, 0.3);
}

/* ===== PREDICTION CARD ===== */
.prediction-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(168, 85, 247, 0.08) 100%);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    padding: 2rem;
    position: relative;
    overflow: hidden;
}

.prediction-card::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 100%;
    height: 100%;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.05) 0%, transparent 70%);
    pointer-events: none;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--bg-card);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid var(--border-color);
}

.stTabs [data-baseweb="tab"] {
    color: var(--text-muted) !important;
    border-radius: 10px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}

.stTabs [aria-selected="true"] {
    background: var(--gradient-primary) !important;
    color: white !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}

.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* ===== SECTION HEADERS ===== */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.5rem 0 1rem;
}

.section-header .icon {
    width: 36px;
    height: 36px;
    background: var(--gradient-primary);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
}

.section-header h3 {
    color: var(--text-primary);
    font-weight: 700;
    font-size: 1.15rem;
    margin: 0;
}

/* ===== GLASSMORPHISM PANEL ===== */
.glass-panel {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 1.5rem;
}

/* ===== INDICATOR GAUGES ===== */
.gauge-container {
    text-align: center;
    padding: 1rem;
}

.gauge-value {
    font-size: 2rem;
    font-weight: 800;
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.gauge-label {
    color: var(--text-muted);
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

/* ===== BUTTONS ===== */
.stButton > button {
    background: var(--gradient-primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 2rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(99, 102, 241, 0.45) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ===== EXPANDERS ===== */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* ===== INFO/SUCCESS/WARNING/ERROR BOXES ===== */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: none !important;
    backdrop-filter: blur(8px) !important;
}

/* ===== HERO HEADER ===== */
.hero-header {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(168, 85, 247, 0.08) 50%, rgba(34, 211, 238, 0.06) 100%);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: -100px; right: -100px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.08) 0%, transparent 70%);
    pointer-events: none;
}

.hero-header .hero-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f1f5f9 0%, #cbd5e1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.3;
}

.hero-header .hero-subtitle {
    color: var(--text-muted);
    font-size: 0.9rem;
    font-weight: 400;
    margin-top: 0.4rem;
}

.hero-header .hero-status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.12);
    color: var(--accent-green);
    padding: 4px 12px;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 0.8rem;
}

.hero-header .hero-status .dot {
    width: 7px; height: 7px;
    background: var(--accent-green);
    border-radius: 50%;
    animation: pulse-dot 2s ease-in-out infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ===== VOLUME BAR ===== */
.volume-bar-container {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 4px 0;
}

.volume-bar {
    flex: 1;
    height: 6px;
    background: rgba(99, 102, 241, 0.1);
    border-radius: 3px;
    overflow: hidden;
}

.volume-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s ease;
}

/* ===== TREND TABLE ===== */
.trend-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.65rem 1rem;
    border-radius: 10px;
    margin: 4px 0;
    background: rgba(99, 102, 241, 0.04);
    border: 1px solid transparent;
    transition: all 0.2s ease;
}

.trend-row:hover {
    border-color: var(--border-color);
    background: rgba(99, 102, 241, 0.08);
}

/* ===== FOOTER ===== */
.dashboard-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: var(--text-muted);
    font-size: 0.75rem;
    border-top: 1px solid var(--border-color);
    margin-top: 3rem;
}

/* ===== ANIMATION UTILITIES ===== */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

.animate-in {
    animation: fadeInUp 0.5s ease forwards;
}

/* ===== RESPONSIVE COLUMNS ===== */
[data-testid="stHorizontalBlock"] {
    gap: 1rem !important;
}

/* ===== EMPTY PLOTLY CHART BACKGROUND ===== */
.js-plotly-plot .plotly .main-svg {
    border-radius: 16px;
}

/* ===== START PAGE ===== */
.start-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 82vh;
    text-align: center;
    animation: fadeInUp 0.8s ease forwards;
}

.start-logo {
    font-size: 4.5rem;
    margin-bottom: 0.2rem;
    filter: drop-shadow(0 0 25px rgba(99, 102, 241, 0.5));
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

.start-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 40%, #22d3ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin-bottom: 0.6rem;
}

.start-subtitle {
    color: var(--text-muted);
    font-size: 1.1rem;
    font-weight: 400;
    max-width: 520px;
    line-height: 1.6;
    margin: 0 auto 2.5rem;
}

.start-features {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-bottom: 2.5rem;
    flex-wrap: wrap;
}

.start-feature-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 50px;
    padding: 8px 18px;
    color: var(--text-secondary);
    font-size: 0.82rem;
    font-weight: 500;
    transition: all 0.3s ease;
}

.start-feature-pill:hover {
    border-color: var(--accent-blue);
    background: rgba(99, 102, 241, 0.14);
    transform: translateY(-2px);
}

.start-search-wrapper {
    max-width: 520px;
    margin: 0 auto 1.5rem;
    width: 100%;
}

.start-popular {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 2rem;
}

.start-popular-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 10px 18px;
    color: var(--text-primary);
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    text-decoration: none;
}

.start-popular-chip:hover {
    border-color: var(--accent-blue);
    box-shadow: var(--shadow-glow);
    transform: translateY(-3px);
}

.start-popular-chip .chip-icon {
    font-size: 1.1rem;
}

.start-popular-chip .chip-name {
    color: var(--text-muted);
    font-size: 0.72rem;
    font-weight: 400;
}

.start-footer-text {
    color: var(--text-muted);
    font-size: 0.72rem;
    margin-top: 2rem;
    opacity: 0.7;
}
</style>
"""


def inject_css():
    """Inject the custom CSS theme into the Streamlit app."""
    import streamlit as st
    st.markdown(MAIN_CSS, unsafe_allow_html=True)
