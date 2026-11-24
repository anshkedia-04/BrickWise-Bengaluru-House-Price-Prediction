"""Custom CSS styling for BrickWise Bengaluru Property Valuation UI."""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

:root {
    --bw-ink: #0F172A;
    --bw-muted: #475569;
    --bw-subtle: #64748B;
    --bw-surface: #FFFFFF;
    --bw-bg: #F8FAFC;
    --bw-border: #E2E8F0;
    --bw-accent: #B84A27;
    --bw-accent-soft: #FFF4EF;
    --bw-accent-border: #F4C7B8;
    --bw-emerald: #047857;
    --bw-emerald-soft: #ECFDF5;
}

/* Base layout */
.stApp {
    background-color: var(--bw-bg);
    color: var(--bw-ink);
}

.block-container {
    padding-top: 1.75rem !important;
    padding-bottom: 3rem !important;
    max-width: 1240px !important;
}

h1, h2, h3, h4 {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--bw-ink);
    letter-spacing: -0.02em;
}

p, label, li, td, th, .stMarkdown, .stCaption {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hide default footer and deploy noise cleanly */
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

/* Brand Header */
.bw-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 68%, #292524 100%);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.35rem;
    color: #F8FAFC;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1.25rem;
    box-shadow: 0 10px 30px -12px rgba(15, 23, 42, 0.28);
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.bw-brand-row {
    display: flex;
    align-items: center;
    gap: 0.85rem;
}

.bw-logo-mark {
    width: 44px;
    height: 44px;
    border-radius: 11px;
    background: linear-gradient(135deg, #C2410C 0%, #EA580C 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 1.15rem;
    color: #FFFFFF;
    font-family: 'Plus Jakarta Sans', sans-serif;
    box-shadow: 0 6px 16px -4px rgba(194, 65, 12, 0.55);
    flex-shrink: 0;
}

.bw-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.55rem;
    font-weight: 800;
    line-height: 1.15;
    margin: 0;
    color: #FFFFFF;
}

.bw-subtitle {
    font-size: 0.9rem;
    color: #CBD5E1;
    margin: 0.25rem 0 0 0;
    font-weight: 400;
}

.bw-header-pills {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
}

.bw-pill {
    background: rgba(255, 255, 255, 0.09);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 999px;
    padding: 0.42rem 0.85rem;
    font-size: 0.78rem;
    color: #E2E8F0;
    font-weight: 500;
    white-space: nowrap;
}

.bw-pill strong {
    color: #FFFFFF;
    font-family: 'JetBrains Mono', monospace;
}

/* Section Cards */
.bw-card {
    background: var(--bw-surface);
    border: 1px solid var(--bw-border);
    border-radius: 14px;
    padding: 1.35rem 1.5rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    margin-bottom: 1rem;
}

.bw-card-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--bw-ink);
    margin: 0 0 0.25rem 0;
}

.bw-card-caption {
    font-size: 0.82rem;
    color: var(--bw-subtle);
    margin: 0 0 0.9rem 0;
}

/* Hero Valuation Result */
.bw-valuation-hero {
    background: linear-gradient(135deg, #FFFFFF 0%, #FFF7F4 100%);
    border: 1px solid var(--bw-accent-border);
    border-left: 5px solid var(--bw-accent);
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1rem;
    box-shadow: 0 6px 22px -10px rgba(184, 74, 39, 0.18);
}

.bw-eyebrow {
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--bw-accent);
    margin-bottom: 0.35rem;
}

.bw-price-main {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.35rem;
    font-weight: 800;
    color: var(--bw-ink);
    line-height: 1.1;
    font-variant-numeric: tabular-nums;
    margin: 0;
}

.bw-price-sub {
    margin-top: 0.45rem;
    font-size: 0.92rem;
    color: var(--bw-muted);
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    align-items: center;
}

.bw-mono {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}

.bw-badge-neutral {
    display: inline-block;
    background: #F1F5F9;
    color: #334155;
    border-radius: 6px;
    padding: 0.2rem 0.55rem;
    font-size: 0.78rem;
    font-weight: 600;
}

.bw-badge-up {
    display: inline-block;
    background: #FEF2F2;
    color: #B91C1C;
    border-radius: 6px;
    padding: 0.2rem 0.55rem;
    font-size: 0.78rem;
    font-weight: 600;
}

.bw-badge-down {
    display: inline-block;
    background: var(--bw-emerald-soft);
    color: var(--bw-emerald);
    border-radius: 6px;
    padding: 0.2rem 0.55rem;
    font-size: 0.78rem;
    font-weight: 600;
}

/* Metric Mini Grid */
.bw-kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.75rem;
    margin-bottom: 1rem;
}

.bw-kpi-box {
    background: var(--bw-surface);
    border: 1px solid var(--bw-border);
    border-radius: 12px;
    padding: 0.95rem 1.05rem;
}

.bw-kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--bw-subtle);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.3rem;
}

.bw-kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--bw-ink);
}

.bw-kpi-note {
    font-size: 0.76rem;
    color: var(--bw-subtle);
    margin-top: 0.2rem;
}

/* Tabs refinement */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: #FFFFFF;
    padding: 0.35rem;
    border-radius: 12px;
    border: 1px solid var(--bw-border);
    margin-bottom: 1.15rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.5rem 1.05rem;
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--bw-muted);
}

.stTabs [aria-selected="true"] {
    background-color: var(--bw-ink) !important;
    color: #FFFFFF !important;
}

/* Primary button styling */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #B84A27 0%, #C2410C 100%);
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.62rem 1.15rem;
    box-shadow: 0 6px 16px -6px rgba(184, 74, 39, 0.55);
}

div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #9A3B1D 0%, #B4380A 100%);
}

/* Advisory box */
.bw-advisory {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 4px solid #D97706;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 0.84rem;
    color: #78350F;
    margin-bottom: 0.85rem;
}

@media (max-width: 768px) {
    .bw-header {
        padding: 1.2rem 1.25rem;
    }
    .bw-price-main {
        font-size: 1.85rem;
    }
    .bw-kpi-grid {
        grid-template-columns: 1fr;
    }
}
</style>
"""
