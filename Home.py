"""
╔══════════════════════════════════════════════════════════════════╗
║         AI-Powered Data Science Toolkit — Home Page             ║
║         Production-grade Streamlit dashboard homepage            ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the first Streamlit call
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Science Toolkit",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────
# GLOBAL CSS
# Aesthetic direction: refined dark-industrial with sharp amber /
# cyan / emerald accents. Uses "Syne" (display) + "DM Sans" (body)
# from Google Fonts. Cards have a glassy frosted border, not the
# dull solid border from the old version.
# ─────────────────────────────────────────────────────────────────
def inject_global_css() -> None:
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">

        <style>
        /* ── Reset & base ───────────────────────────────────────── */
        *, *::before, *::after { box-sizing: border-box; }

        html, body, [class*="css"], .stApp {
            background-color: #080c14 !important;
            font-family: 'DM Sans', sans-serif;
            color: #c9d1e0;
        }

        /* Remove Streamlit chrome clutter */
        #MainMenu, footer, header { visibility: hidden !important; }
        .block-container {
            padding: 2rem 3rem 4rem 3rem !important;
            max-width: 1280px;
        }

        /* ── Sidebar ────────────────────────────────────────────── */
        section[data-testid="stSidebar"] {
            background: #0b1120 !important;
            border-right: 1px solid #1c2740;
        }
        section[data-testid="stSidebar"] * {
            font-family: 'DM Sans', sans-serif !important;
        }

        /* ── Hero Banner ────────────────────────────────────────── */
        .hero-wrapper {
            position: relative;
            overflow: hidden;
            border-radius: 20px;
            padding: 4rem 3.5rem 3.5rem;
            margin-bottom: 3rem;
            background:
                radial-gradient(ellipse 80% 60% at 70% 0%,  rgba(255,180,0,0.10) 0%, transparent 60%),
                radial-gradient(ellipse 60% 80% at 0%  80%,  rgba(0,210,200,0.08) 0%, transparent 55%),
                linear-gradient(160deg, #0d1526 0%, #0a1020 100%);
            border: 1px solid #1c2d4a;
        }
        /* Subtle grid overlay */
        .hero-wrapper::before {
            content: "";
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
            background-size: 48px 48px;
            pointer-events: none;
        }
        .hero-eyebrow {
            display: inline-block;
            font-family: 'DM Sans', sans-serif;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #ffb400;
            background: rgba(255,180,0,0.10);
            border: 1px solid rgba(255,180,0,0.25);
            padding: 4px 14px;
            border-radius: 30px;
            margin-bottom: 1.2rem;
        }
        .hero-title {
            font-family: 'Syne', sans-serif;
            font-size: clamp(2.2rem, 4vw, 3.6rem);
            font-weight: 800;
            line-height: 1.1;
            color: #eaf0ff;
            margin: 0 0 1rem 0;
            letter-spacing: -0.03em;
        }
        .hero-title em {
            font-style: normal;
            background: linear-gradient(90deg, #ffb400, #ff7c1f);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            font-weight: 400;
            color: #7d8fa8;
            max-width: 560px;
            line-height: 1.75;
            margin: 0 0 2rem 0;
        }
        .hero-tagline-pills {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .pill {
            font-size: 0.78rem;
            font-weight: 500;
            padding: 5px 15px;
            border-radius: 30px;
            border: 1px solid;
        }
        .pill-amber  { color: #ffb400; border-color: rgba(255,180,0,0.35);  background: rgba(255,180,0,0.06);  }
        .pill-cyan   { color: #00d2c8; border-color: rgba(0,210,200,0.35);  background: rgba(0,210,200,0.06);  }
        .pill-emerald{ color: #00c97d; border-color: rgba(0,201,125,0.35);  background: rgba(0,201,125,0.06);  }

        /* ── Section heading ────────────────────────────────────── */
        .section-label {
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #3d5278;
            margin-bottom: 0.4rem;
        }
        .section-title {
            font-family: 'Syne', sans-serif;
            font-size: 1.65rem;
            font-weight: 700;
            color: #dce6f7;
            margin: 0 0 0.5rem 0;
        }
        .section-divider {
            height: 2px;
            width: 48px;
            border-radius: 2px;
            margin: 0.5rem 0 2.2rem 0;
        }
        .divider-amber   { background: linear-gradient(90deg, #ffb400, transparent); }
        .divider-cyan    { background: linear-gradient(90deg, #00d2c8, transparent); }
        .divider-emerald { background: linear-gradient(90deg, #00c97d, transparent); }

        /* ── Module Cards ───────────────────────────────────────── */
        .mod-card {
            position: relative;
            border-radius: 16px;
            padding: 2rem 1.8rem 1.6rem;
            background: #0d1526;
            border: 1px solid #1c2d4a;
            transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
            overflow: hidden;
            height: 100%;
        }
        .mod-card::after {
            content: "";
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 3px;
            border-radius: 16px 16px 0 0;
        }
        .mod-card-amber::after   { background: linear-gradient(90deg, #ffb400, #ff7c1f); }
        .mod-card-cyan::after    { background: linear-gradient(90deg, #00d2c8, #0085ff); }
        .mod-card-emerald::after { background: linear-gradient(90deg, #00c97d, #00d4aa); }

        .mod-card:hover {
            transform: translateY(-5px);
            box-shadow:
                0 16px 48px rgba(0,0,0,0.45),
                0  0  0 1px rgba(255,255,255,0.06);
        }
        .mod-card-amber:hover   { border-color: rgba(255,180,0,0.35);  box-shadow: 0 16px 48px rgba(255,180,0,0.12), 0 0 0 1px rgba(255,180,0,0.2); }
        .mod-card-cyan:hover    { border-color: rgba(0,210,200,0.35);  box-shadow: 0 16px 48px rgba(0,210,200,0.10), 0 0 0 1px rgba(0,210,200,0.2); }
        .mod-card-emerald:hover { border-color: rgba(0,201,125,0.35);  box-shadow: 0 16px 48px rgba(0,201,125,0.10), 0 0 0 1px rgba(0,201,125,0.2); }

        /* Card glow orb */
        .card-orb {
            position: absolute;
            top: -40px; right: -40px;
            width: 140px; height: 140px;
            border-radius: 50%;
            filter: blur(55px);
            opacity: 0.18;
            pointer-events: none;
        }
        .orb-amber   { background: #ffb400; }
        .orb-cyan    { background: #00d2c8; }
        .orb-emerald { background: #00c97d; }

        .card-icon-wrap {
            width: 52px; height: 52px;
            border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.5rem;
            margin-bottom: 1.1rem;
        }
        .icon-amber   { background: rgba(255,180,0,0.12);  border: 1px solid rgba(255,180,0,0.25); }
        .icon-cyan    { background: rgba(0,210,200,0.10);  border: 1px solid rgba(0,210,200,0.25); }
        .icon-emerald { background: rgba(0,201,125,0.10);  border: 1px solid rgba(0,201,125,0.25); }

        .card-title {
            font-family: 'Syne', sans-serif;
            font-size: 1.15rem;
            font-weight: 700;
            color: #dce6f7;
            margin: 0 0 0.5rem 0;
        }
        .card-desc {
            font-size: 0.875rem;
            color: #6a7d9a;
            line-height: 1.7;
            margin: 0 0 1.3rem 0;
        }
        .feature-list {
            list-style: none;
            padding: 0; margin: 0 0 1.5rem 0;
        }
        .feature-list li {
            font-size: 0.82rem;
            color: #8496b4;
            padding: 5px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .feat-dot {
            width: 5px; height: 5px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .dot-amber   { background: #ffb400; }
        .dot-cyan    { background: #00d2c8; }
        .dot-emerald { background: #00c97d; }

        .card-badge {
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 3px 12px;
            border-radius: 20px;
            margin-bottom: 1.2rem;
        }
        .badge-amber   { color: #ffb400; background: rgba(255,180,0,0.10);  border: 1px solid rgba(255,180,0,0.2); }
        .badge-cyan    { color: #00d2c8; background: rgba(0,210,200,0.08);  border: 1px solid rgba(0,210,200,0.2); }
        .badge-emerald { color: #00c97d; background: rgba(0,201,125,0.08);  border: 1px solid rgba(0,201,125,0.2); }

        /* ── CTA Buttons (Streamlit page_link override) ─────────── */
        /* Target the anchor inside st.page_link */
        div[data-testid="stPageLink"] a {
            display: inline-flex !important;
            align-items: center !important;
            gap: 6px !important;
            font-family: 'DM Sans', sans-serif !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            padding: 10px 22px !important;
            border-radius: 8px !important;
            text-decoration: none !important;
            transition: all 0.2s ease !important;
            letter-spacing: 0.01em !important;
            width: 100% !important;
            justify-content: center !important;
        }

        /* Per-column button colors via nth-child targeting */
        .btn-amber a {
            background: rgba(255,180,0,0.12)  !important;
            border: 1px solid rgba(255,180,0,0.35) !important;
            color: #ffb400 !important;
        }
        .btn-amber a:hover {
            background: rgba(255,180,0,0.22) !important;
            border-color: #ffb400 !important;
        }
        .btn-cyan a {
            background: rgba(0,210,200,0.10) !important;
            border: 1px solid rgba(0,210,200,0.35) !important;
            color: #00d2c8 !important;
        }
        .btn-cyan a:hover {
            background: rgba(0,210,200,0.20) !important;
            border-color: #00d2c8 !important;
        }
        .btn-emerald a {
            background: rgba(0,201,125,0.10) !important;
            border: 1px solid rgba(0,201,125,0.35) !important;
            color: #00c97d !important;
        }
        .btn-emerald a:hover {
            background: rgba(0,201,125,0.20) !important;
            border-color: #00c97d !important;
        }

        /* ── Stats Strip ────────────────────────────────────────── */
        .stats-strip {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1px;
            background: #1c2d4a;
            border-radius: 14px;
            overflow: hidden;
            margin: 3rem 0 0 0;
            border: 1px solid #1c2d4a;
        }
        .stat-cell {
            background: #0d1526;
            padding: 1.4rem 1.2rem;
            text-align: center;
        }
        .stat-val {
            font-family: 'Syne', sans-serif;
            font-size: 1.9rem;
            font-weight: 800;
            color: #dce6f7;
            line-height: 1;
        }
        .stat-key {
            font-size: 0.75rem;
            font-weight: 500;
            color: #3d5278;
            margin-top: 5px;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        /* ── Footer ─────────────────────────────────────────────── */
        .footer {
            margin-top: 4rem;
            padding-top: 1.5rem;
            border-top: 1px solid #1c2d4a;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        .footer-left {
            font-size: 0.78rem;
            color: #3d5278;
        }
        .footer-right {
            font-size: 0.78rem;
            color: #2a3d5c;
        }

        /* ── Sidebar inner styles ───────────────────────────────── */
        .sidebar-logo {
            font-family: 'Syne', sans-serif;
            font-size: 1.1rem;
            font-weight: 800;
            color: #dce6f7;
            margin-bottom: 0.2rem;
        }
        .sidebar-tagline {
            font-size: 0.72rem;
            color: #3d5278;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 1.2rem;
        }
        .sidebar-nav-label {
            font-size: 0.65rem;
            font-weight: 600;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            color: #2a3d5c;
            margin: 1.2rem 0 0.5rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        # Logo / brand
        st.markdown(
            """
            <div style="padding: 0.5rem 0 0.8rem;">
                <div class="sidebar-logo">⚡ AI DS Toolkit</div>
                <div class="sidebar-tagline">Query · Learn · Clean · Automate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-nav-label">Navigation</div>', unsafe_allow_html=True)

        # Page links — filenames match the existing project structure
        st.page_link("Home.py",                label="🏠  Home",            )
        st.page_link("pages/1_DataSphere.py",  label="🧠  SQL Assistant",   )
        st.page_link("pages/2_IntelliML.py",   label="🤖  ML Advisor",      )
        st.page_link("pages/3_DataClean.py",   label="🧹  Data Cleaner",    )

        st.markdown('<div class="sidebar-nav-label" style="margin-top:2rem;">Platform Info</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size:0.78rem; color:#3d5278; line-height:1.8;">
                Version &nbsp;<strong style="color:#556882;">1.0.0</strong><br>
                Status &nbsp;&nbsp;<span style="color:#00c97d;">● Live</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────────────────────────
def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-wrapper">
            <div class="hero-eyebrow">⚡ AI-Powered Platform</div>
            <h1 class="hero-title">
                The <em>intelligent</em> toolkit<br>for data science teams.
            </h1>
            <p class="hero-subtitle">
                A unified platform that turns natural language into SQL,
                recommends the right ML model for your data, and auto-cleans
                messy datasets — all without leaving your browser.
            </p>
            <div class="hero-tagline-pills">
                <span class="pill pill-amber">🧠 NLP → SQL</span>
                <span class="pill pill-cyan">🤖 Auto ML Advisor</span>
                <span class="pill pill-emerald">🧹 Smart Data Cleaning</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
# MODULE CARDS — one function per card keeps it readable
# ─────────────────────────────────────────────────────────────────
def card_sql_assistant() -> None:
    """Card for the Natural Language → SQL module."""
    st.markdown(
        """
        <div class="mod-card mod-card-amber">
            <div class="card-orb orb-amber"></div>
            <div class="card-icon-wrap icon-amber">🧠</div>
            <span class="card-badge badge-amber">NLP → SQL</span>
            <h3 class="card-title">Natural Language to SQL Assistant</h3>
            <p class="card-desc">
                Transform plain-English questions into executable SQL queries
                and connect seamlessly to relational databases.
            </p>
            <ul class="feature-list">
                <li><span class="feat-dot dot-amber"></span>Multi-database support — MySQL, PostgreSQL, SQLite</li>
                <li><span class="feat-dot dot-amber"></span>Natural language query interpretation</li>
                <li><span class="feat-dot dot-amber"></span>Instant SQL query generation</li>
                <li><span class="feat-dot dot-amber"></span>Live query execution &amp; result preview</li>
                <li><span class="feat-dot dot-amber"></span>Interactive result visualisation</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Spacer between card and button
    st.write("")
    st.markdown('<div class="btn-amber">', unsafe_allow_html=True)
    st.page_link("pages/1_DataSphere.py", label="Launch SQL Assistant →", icon="🧠")
    st.markdown("</div>", unsafe_allow_html=True)


def card_ml_advisor() -> None:
    """Card for the ML Advisor module."""
    st.markdown(
        """
        <div class="mod-card mod-card-cyan">
            <div class="card-orb orb-cyan"></div>
            <div class="card-icon-wrap icon-cyan">🤖</div>
            <span class="card-badge badge-cyan">Auto ML</span>
            <h3 class="card-title">Intelligent Machine Learning Advisor</h3>
            <p class="card-desc">
                Analyse your dataset and receive smart recommendations
                for the best ML model based on your data's characteristics.
            </p>
            <ul class="feature-list">
                <li><span class="feat-dot dot-cyan"></span>Automatic dataset profiling</li>
                <li><span class="feat-dot dot-cyan"></span>Problem-type detection — Classification / Regression / Clustering</li>
                <li><span class="feat-dot dot-cyan"></span>Ranked algorithm recommendations</li>
                <li><span class="feat-dot dot-cyan"></span>Model performance insights</li>
                <li><span class="feat-dot dot-cyan"></span>Feature importance guidance</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown('<div class="btn-cyan">', unsafe_allow_html=True)
    st.page_link("pages/2_IntelliML.py", label="Launch ML Advisor →", icon="🤖")
    st.markdown("</div>", unsafe_allow_html=True)


def card_data_cleaner() -> None:
    """Card for the Automatic Data Cleaning module."""
    st.markdown(
        """
        <div class="mod-card mod-card-emerald">
            <div class="card-orb orb-emerald"></div>
            <div class="card-icon-wrap icon-emerald">🧹</div>
            <span class="card-badge badge-emerald">Auto Clean</span>
            <h3 class="card-title">Automatic Data Cleaning System</h3>
            <p class="card-desc">
                Resolve inconsistencies, missing values, and formatting issues
                automatically — get a model-ready dataset in seconds.
            </p>
            <ul class="feature-list">
                <li><span class="feat-dot dot-emerald"></span>Missing value detection &amp; smart imputation</li>
                <li><span class="feat-dot dot-emerald"></span>Duplicate row removal</li>
                <li><span class="feat-dot dot-emerald"></span>Data type detection &amp; correction</li>
                <li><span class="feat-dot dot-emerald"></span>Outlier detection &amp; treatment</li>
                <li><span class="feat-dot dot-emerald"></span>Normalisation &amp; standardisation</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown('<div class="btn-emerald">', unsafe_allow_html=True)
    st.page_link("pages/3_DataClean.py", label="Launch Data Cleaner →", icon="🧹")
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# MODULES SECTION (section header + 3-column card grid)
# ─────────────────────────────────────────────────────────────────
def render_modules() -> None:
    # Section heading
    st.markdown(
        """
        <div class="section-label">Core modules</div>
        <h2 class="section-title">Three tools, one platform.</h2>
        <div class="section-divider divider-amber"></div>
        """,
        unsafe_allow_html=True,
    )

    # Three equal columns, slightly gapped
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        card_sql_assistant()
    with col2:
        card_ml_advisor()
    with col3:
        card_data_cleaner()


# ─────────────────────────────────────────────────────────────────
# STATS STRIP
# ─────────────────────────────────────────────────────────────────
def render_stats() -> None:
    st.markdown(
        """
        <div class="stats-strip">
            <div class="stat-cell">
                <div class="stat-val">3</div>
                <div class="stat-key">Integrated Apps</div>
            </div>
            <div class="stat-cell">
                <div class="stat-val" style="color:#ffb400;">NLP</div>
                <div class="stat-key">SQL Generation</div>
            </div>
            <div class="stat-cell">
                <div class="stat-val" style="color:#00d2c8;">Auto</div>
                <div class="stat-key">Model Selection</div>
            </div>
            <div class="stat-cell">
                <div class="stat-val" style="color:#00c97d;">AI</div>
                <div class="stat-key">Data Cleaning</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
def render_footer() -> None:
    st.markdown(
        """
        <div class="footer">
            <span class="footer-left">Built with <strong style="color:#556882;">Streamlit</strong></span>
            <span class="footer-right">AI Data Science Toolkit &copy; 2026</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    inject_global_css()
    render_sidebar()
    render_hero()
    render_modules()
    render_stats()
    render_footer()


if __name__ == "__main__" or True:
    # `or True` ensures main() runs when Streamlit imports this file
    main()