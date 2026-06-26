import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from extractor import extract_text, chunk_text
from analyzer import extract_kpis
from greenwash import detect_greenwashing, calculate_risk_score, summarize_greenwashing
from scorer import get_all_scores

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="ESG Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CUSTOM CSS — clean, professional, data-forward
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Override Streamlit primary button → small, pill-shaped, subtle */
div.stButton > button[kind="primary"] {
    background: #0A4D3C !important;
    color: white !important;
    border: none !important;
    border-radius: 100px !important;
    padding: 8px 28px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    letter-spacing: 0.02em !important;
    width: auto !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    box-shadow: 0 2px 8px rgba(10,77,60,0.25) !important;
    transition: all 0.2s !important;
}
div.stButton > button[kind="primary"]:hover {
    background: #0D6B52 !important;
    box-shadow: 0 4px 14px rgba(10,77,60,0.35) !important;
    transform: translateY(-1px) !important;
}

/* Center the button */
div.stButton {
    display: flex;
    justify-content: center;
    margin: 16px 0;
}

/* Page background */
.stApp {
    background: #F7F8FA;
}

/* Hide default streamlit header */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0A4D3C 0%, #1A7A5E 60%, #2AAD82 100%);
    border-radius: 16px;
    padding: 44px 48px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "ESG";
    position: absolute;
    right: 40px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 120px;
    font-weight: 700;
    color: rgba(255,255,255,0.06);
    letter-spacing: -4px;
    pointer-events: none;
}
.hero h1 {
    color: #FFFFFF;
    font-size: 32px;
    font-weight: 700;
    margin: 0 0 6px;
    letter-spacing: -0.5px;
}
.hero p {
    color: rgba(255,255,255,0.75);
    font-size: 15px;
    margin: 0;
    line-height: 1.6;
}
.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    color: rgba(255,255,255,0.9);
    font-size: 11px;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 0 4px 12px 0;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* Score cards */
.score-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 24px;
}
.score-card {
    background: white;
    border-radius: 12px;
    padding: 20px 22px;
    border: 1px solid #EAECF0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.score-card .label {
    font-size: 11px;
    font-weight: 600;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 8px;
}
.score-card .value {
    font-size: 34px;
    font-weight: 700;
    color: #111827;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
}
.score-card .value span {
    font-size: 16px;
    font-weight: 500;
    color: #9CA3AF;
}
.score-card .pill {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    margin-top: 8px;
}
.pill-green { background: #D1FAE5; color: #065F46; }
.pill-yellow { background: #FEF3C7; color: #92400E; }
.pill-red { background: #FEE2E2; color: #991B1B; }
.pill-blue { background: #DBEAFE; color: #1E40AF; }

/* Category score strip */
.cat-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 24px;
}
.cat-card {
    background: white;
    border-radius: 12px;
    padding: 18px 22px;
    border: 1px solid #EAECF0;
    display: flex;
    align-items: center;
    gap: 16px;
}
.cat-icon {
    width: 44px;
    height: 44px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}
.cat-e { background: #D1FAE5; }
.cat-s { background: #DBEAFE; }
.cat-g { background: #EDE9FE; }
.cat-label { font-size: 12px; color: #6B7280; font-weight: 500; margin-bottom: 2px; }
.cat-val { font-size: 24px; font-weight: 700; color: #111827; font-family: 'JetBrains Mono', monospace; }

/* Section header */
.section-header {
    font-size: 18px;
    font-weight: 600;
    color: #111827;
    margin: 32px 0 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Insight panel */
.insight-panel {
    background: white;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid #EAECF0;
    margin-bottom: 14px;
}
.insight-row {
    display: flex;
    gap: 24px;
}
.insight-col { flex: 1; }
.insight-col h4 {
    font-size: 13px;
    font-weight: 600;
    color: #374151;
    margin: 0 0 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.insight-item {
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #4B5563;
    line-height: 1.5;
}
.insight-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
    margin-top: 6px;
}
.dot-green { background: #10B981; }
.dot-amber { background: #F59E0B; }
.dot-blue { background: #3B82F6; }

/* KPI table */
.kpi-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
.kpi-table th {
    background: #F9FAFB;
    color: #6B7280;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid #EAECF0;
}
.kpi-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #F3F4F6;
    color: #374151;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px;
}
.kpi-table tr:last-child td { border-bottom: none; }
.kpi-table tr:hover td { background: #F9FAFB; }

/* Greenwashing flag cards */
.flag-high {
    background: #FFF5F5;
    border: 1px solid #FECACA;
    border-left: 4px solid #EF4444;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.flag-medium {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 4px solid #F59E0B;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.flag-low {
    background: #F0F9FF;
    border: 1px solid #BAE6FD;
    border-left: 4px solid #38BDF8;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.flag-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 20px;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.badge-high { background: #FEE2E2; color: #991B1B; }
.badge-medium { background: #FEF3C7; color: #92400E; }
.badge-low { background: #E0F2FE; color: #0369A1; }
.flag-claim { font-size: 13px; font-weight: 500; color: #1F2937; margin-bottom: 4px; }
.flag-reason { font-size: 12px; color: #6B7280; line-height: 1.5; }

/* Upload zone */
.upload-zone {
    background: white;
    border-radius: 12px;
    padding: 28px;
    border: 1.5px dashed #D1D5DB;
    text-align: center;
    margin-bottom: 20px;
}

/* Progress bar */
.progress-bar {
    height: 4px;
    background: #E5E7EB;
    border-radius: 4px;
    overflow: hidden;
    margin: 8px 0;
}
.progress-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
}

/* Divider */
.my-divider {
    border: none;
    border-top: 1px solid #EAECF0;
    margin: 28px 0;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HERO
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <div>
        <span class="hero-tag">AI-Powered</span>
        <span class="hero-tag">Groq LLM</span>
        <span class="hero-tag">ESG Analytics</span>
    </div>
    <h1>🌿 ESG Intelligence Dashboard</h1>
    <p>Upload any ESG or Sustainability Report — extract KPIs, score disclosure quality,<br>detect greenwashing, and generate AI-powered insights in minutes.</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload ESG Report (PDF)",
    type=["pdf"],
    help="Upload any corporate ESG, sustainability, or annual report"
)

if uploaded_file:
    st.success(f"✅ Uploaded: **{uploaded_file.name}** ({round(uploaded_file.size/1024/1024, 1)} MB)")

    if st.button("🔍  Analyze Report", type="primary"):

        col_p1, col_p2, col_p3 = st.columns(3)

        with st.spinner("📄 Extracting text from PDF..."):
            text = extract_text(uploaded_file)
            chunks = chunk_text(text)

        st.info(f"📊 Extracted **{len(text):,} characters** across **{len(chunks)} chunks**")

        with st.spinner("🤖 AI extracting ESG KPIs — this takes ~60 seconds..."):
            kpis = extract_kpis(chunks)

        with st.spinner("🔎 Scanning for greenwashing..."):
            flags = detect_greenwashing(chunks)
            risk_score = calculate_risk_score(flags)
            summary = summarize_greenwashing(flags)

        with st.spinner("📈 Generating ESG rating..."):
            scores = get_all_scores(kpis)

        st.markdown("<hr class='my-divider'>", unsafe_allow_html=True)

        # ============================================================
        # SCORE CARDS
        # ============================================================

        grade = scores.get("grade", "N/A")
        grade_color = "pill-green" if grade in ["A","A+","Excellent"] else "pill-yellow" if grade in ["B","B+","Good"] else "pill-amber"

        st.markdown(f"""
        <div class="score-grid">
            <div class="score-card">
                <div class="label">Overall ESG</div>
                <div class="value">{scores['overall']}<span>/100</span></div>
                <div class="pill pill-blue">{scores.get('maturity','N/A')}</div>
            </div>
            <div class="score-card">
                <div class="label">Grade</div>
                <div class="value">{grade}</div>
                <div class="pill pill-green">Disclosure Quality</div>
            </div>
            <div class="score-card">
                <div class="label">Confidence</div>
                <div class="value">{scores.get('confidence',0)}<span>%</span></div>
                <div class="pill pill-yellow">AI Confidence</div>
            </div>
            <div class="score-card">
                <div class="label">Greenwash Risk</div>
                <div class="value">{risk_score}<span>/100</span></div>
                <div class="pill {'pill-green' if risk_score < 30 else 'pill-yellow' if risk_score < 60 else 'pill-red'}">{'Low Risk' if risk_score < 30 else 'Moderate' if risk_score < 60 else 'High Risk'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ============================================================
        # CATEGORY STRIP
        # ============================================================

        st.markdown(f"""
        <div class="cat-strip">
            <div class="cat-card">
                <div class="cat-icon cat-e">🌱</div>
                <div>
                    <div class="cat-label">Environmental</div>
                    <div class="cat-val">{scores['E']}/100</div>
                </div>
            </div>
            <div class="cat-card">
                <div class="cat-icon cat-s">👥</div>
                <div>
                    <div class="cat-label">Social</div>
                    <div class="cat-val">{scores['S']}/100</div>
                </div>
            </div>
            <div class="cat-card">
                <div class="cat-icon cat-g">🏛</div>
                <div>
                    <div class="cat-label">Governance</div>
                    <div class="cat-val">{scores['G']}/100</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ============================================================
        # CHARTS — side by side
        # ============================================================

        ch1, ch2 = st.columns(2)

        with ch1:
            fig_bar = go.Figure(go.Bar(
                x=["Environmental", "Social", "Governance"],
                y=[scores["E"], scores["S"], scores["G"]],
                marker_color=["#10B981", "#3B82F6", "#8B5CF6"],
                text=[f"{scores['E']}", f"{scores['S']}", f"{scores['G']}"],
                textposition="outside",
                width=0.5
            ))
            fig_bar.update_layout(
                title=dict(text="ESG Category Scores", font=dict(size=14, color="#374151")),
                height=340,
                margin=dict(t=40, b=20, l=10, r=10),
                yaxis=dict(range=[0, 110], showgrid=True, gridcolor="#F3F4F6", title=""),
                xaxis=dict(showgrid=False),
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Plus Jakarta Sans", color="#374151")
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with ch2:
            radar = go.Figure(go.Scatterpolar(
                r=[scores["E"], scores["S"], scores["G"], scores["E"]],
                theta=["Environmental", "Social", "Governance", "Environmental"],
                fill="toself",
                fillcolor="rgba(16,185,129,0.15)",
                line=dict(color="#10B981", width=2),
                name="ESG"
            ))
            radar.update_layout(
                title=dict(text="ESG Radar", font=dict(size=14, color="#374151")),
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
                    bgcolor="white"
                ),
                showlegend=False,
                height=340,
                margin=dict(t=40, b=20, l=30, r=30),
                paper_bgcolor="white",
                font=dict(family="Plus Jakarta Sans")
            )
            st.plotly_chart(radar, use_container_width=True)

        # ============================================================
        # AI INSIGHTS
        # ============================================================

        st.markdown("<div class='section-header'>🤖 AI Executive Summary</div>", unsafe_allow_html=True)

        strengths = scores.get("strengths", [])
        weaknesses = scores.get("weaknesses", [])
        recs = scores.get("recommendations", [])

        strengths_html = "".join([f"<div class='insight-item'><div class='insight-dot dot-green'></div>{s}</div>" for s in strengths]) or "<div class='insight-item'><div class='insight-dot dot-green'></div>No specific strengths identified.</div>"
        weaknesses_html = "".join([f"<div class='insight-item'><div class='insight-dot dot-amber'></div>{w}</div>" for w in weaknesses]) or "<div class='insight-item'><div class='insight-dot dot-amber'></div>No specific weaknesses identified.</div>"

        st.markdown(f"""
        <div class="insight-panel">
            <div class="insight-row">
                <div class="insight-col">
                    <h4>✅ Strengths</h4>
                    {strengths_html}
                </div>
                <div style="width:1px;background:#EAECF0;flex-shrink:0;"></div>
                <div class="insight-col">
                    <h4>⚠️ Weaknesses</h4>
                    {weaknesses_html}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if recs:
            recs_html = "".join([f"<div class='insight-item'><div class='insight-dot dot-blue'></div>{r}</div>" for r in recs])
            st.markdown(f"""
            <div class="insight-panel">
                <h4 style="font-size:13px;font-weight:600;color:#374151;margin:0 0 12px;text-transform:uppercase;letter-spacing:0.05em;">💡 AI Recommendations</h4>
                {recs_html}
            </div>
            """, unsafe_allow_html=True)

        # ============================================================
        # KPI TABLES
        # ============================================================

        st.markdown("<div class='section-header'>📑 Extracted ESG KPIs</div>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["🌱 Environmental", "👥 Social", "🏛 Governance"])

        def render_kpi_table(data_dict):
            if not data_dict:
                st.warning("No KPIs extracted for this category.")
                return
            rows = "".join([
                f"<tr><td>{k.replace('_',' ').title()}</td><td>{v}</td></tr>"
                for k, v in data_dict.items()
                if v not in [None, "null", "N/A", ""]
            ])
            if not rows:
                st.warning("No KPIs extracted for this category.")
                return
            st.markdown(f"""
            <div style="background:white;border-radius:12px;border:1px solid #EAECF0;overflow:hidden;">
            <table class="kpi-table">
                <thead><tr><th>KPI</th><th>Value</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
            </div>
            """, unsafe_allow_html=True)

        with tab1:
            render_kpi_table(kpis.get("environmental", {}))
        with tab2:
            render_kpi_table(kpis.get("social", {}))
        with tab3:
            render_kpi_table(kpis.get("governance", {}))

        # ============================================================
        # GREENWASHING
        # ============================================================

        st.markdown("<div class='section-header'>⚠️ Greenwashing Analysis</div>", unsafe_allow_html=True)

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Total Flags", summary["total_flags"])
        g2.metric("High Risk", summary["high"], delta=None)
        g3.metric("Medium Risk", summary["medium"])
        g4.metric("Low Risk", summary["low"])

        st.info(f"**Verdict:** {summary['verdict']}")

        if flags:
            st.markdown("<br>", unsafe_allow_html=True)
            for flag in flags:
                risk = flag.get("risk", "LOW").upper()
                claim = flag.get("claim", "")[:200]
                reason = flag.get("reason", "")
                css_class = f"flag-{risk.lower()}"
                badge_class = f"badge-{risk.lower()}"
                st.markdown(f"""
                <div class="{css_class}">
                    <span class="flag-badge {badge_class}">{risk} RISK</span>
                    <div class="flag-claim">"{claim}"</div>
                    <div class="flag-reason">{reason}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No significant greenwashing indicators detected.")

        # ============================================================
        # DOWNLOAD
        # ============================================================

        st.markdown("<hr class='my-divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📥 Export Analysis</div>", unsafe_allow_html=True)

        dl1, dl2 = st.columns(2)

        with dl1:
            st.download_button(
                label="📄 Download Full JSON",
                data=json.dumps({"scores": scores, "kpis": kpis, "greenwashing": flags}, indent=4),
                file_name="esg_analysis.json",
                mime="application/json",
                use_container_width=True
            )

        with dl2:
            csv_rows = [{"Category": cat, "KPI": k, "Value": v}
                        for cat, vals in kpis.items()
                        for k, v in vals.items()]
            st.download_button(
                label="📊 Download KPI CSV",
                data=pd.DataFrame(csv_rows).to_csv(index=False),
                file_name="esg_kpis.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("""
        <div style="text-align:center;padding:24px 0 8px;color:#9CA3AF;font-size:12px;">
        Powered by Groq LLM · ESG KPI Extraction · Greenwashing Detection · AI Scoring
        </div>
        """, unsafe_allow_html=True)