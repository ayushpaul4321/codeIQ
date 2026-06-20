"""
SprintGuard Dashboard — Streamlit frontend
Three pages: Sprint Overview | Bug Triage | Re-planner

Dark theme, animated Plotly charts, full API integration.
Requirements: 10.1 – 10.7

Dependencies: streamlit, requests, plotly
"""

from __future__ import annotations

import os
import time
import random
from typing import Optional

import requests
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE: str = os.getenv("API_BASE", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Demo mode mock functions
# ---------------------------------------------------------------------------

def _mock_sprint_risk(sprint_id: str) -> dict:
    return {
        "sprint_id": sprint_id,
        "risk_level": "HIGH",
        "risk_score": 0.78,
        "days_remaining": 3,
        "velocity_trend": -4.5,
        "bug_hours_added_today": 7.2,
        "factors": [
            "High bug hours added today (7.2h) relative to sprint capacity",
            "Velocity declining — team is below target by 4.5 story points",
            "Only 3 days remaining with significant unresolved work",
        ],
    }


def _mock_bug_analyze(title: str, description: str) -> dict:
    import uuid
    devs = ["john.smith", "alice.wong", "bob.martinez", "sara.chen", "dev.patel"]
    probs = sorted([random.uniform(0.1, 0.9) for _ in range(3)], reverse=True)
    total = sum(probs)
    probs = [round(p / total, 3) for p in probs]
    assigned = devs[0]
    hours = round(random.uniform(2.5, 18.0), 1)
    return {
        "bug_id": str(uuid.uuid4()),
        "assigned_dev": assigned,
        "assignment_confidence": probs[0],
        "top3_devs": [
            {"dev": devs[0], "probability": probs[0]},
            {"dev": devs[1], "probability": probs[1]},
            {"dev": devs[2], "probability": probs[2]},
        ],
        "effort_estimate": {
            "hours": hours,
            "confidence_interval": [round(hours * 0.7, 1), round(hours * 1.4, 1)],
        },
        "sprint_impact": {
            "risk_before": "MEDIUM",
            "risk_after": "HIGH",
            "risk_score": 0.74,
            "factors": ["New bug adds " + str(hours) + "h to already tight sprint"],
            "replan_suggested": True,
        },
    }


def _mock_replan(sprint_id: str) -> dict:
    return {
        "sprint_id": sprint_id,
        "current_risk": "HIGH",
        "suggestions": [
            {
                "id": "OPT-A",
                "action": "Defer STORY-108 (Dashboard Analytics) and STORY-112 (Export CSV) to next sprint",
                "story_points_removed": 8,
                "projected_risk": "LOW",
                "projected_risk_score": 0.28,
                "note": None,
            },
            {
                "id": "OPT-B",
                "action": "Defer STORY-112 (Export CSV) only — keep analytics in scope",
                "story_points_removed": 3,
                "projected_risk": "MEDIUM",
                "projected_risk_score": 0.51,
                "note": None,
            },
        ],
        "recommended": "OPT-A",
    }

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SprintGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — dark theme + animations
# ---------------------------------------------------------------------------

GLOBAL_CSS = """
<style>
/* ── Base ─────────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: #e2e8f0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: #12141a;
    border-right: 1px solid #2d2f3d;
}

/* ── Buttons ──────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9);
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    letter-spacing: 0.025em;
    transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.2s ease;
    cursor: pointer;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.45);
    background: linear-gradient(135deg, #8b5cf6, #7c3aed);
}
.stButton > button:active {
    transform: translateY(0);
    box-shadow: none;
}

/* ── Cards ────────────────────────────────────────────────────────────── */
.sg-card {
    background: #1c1e24;
    border: 1px solid #2d2f3d;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    animation: fadeInUp 0.4s ease both;
}
.sg-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}

/* ── Metric card ──────────────────────────────────────────────────────── */
.metric-card {
    background: #1c1e24;
    border: 1px solid #2d2f3d;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    animation: glowIn 0.6s ease both;
}
.metric-card .metric-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 0.4rem;
}
.metric-card .metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1.1;
}
.metric-card .metric-delta {
    font-size: 0.78rem;
    margin-top: 0.3rem;
}

/* ── Badges ───────────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-low    { background: #14532d; color: #22c55e; border: 1px solid #22c55e; }
.badge-medium { background: #451a03; color: #f59e0b; border: 1px solid #f59e0b; }
.badge-high   { background: #450a0a; color: #ef4444; border: 1px solid #ef4444; animation: pulse 1.6s infinite; }
.badge-recommended { background: #1e1b4b; color: #a5b4fc; border: 1px solid #7c3aed; }

/* ── Suggestion card ──────────────────────────────────────────────────── */
.suggestion-card {
    background: #1c1e24;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    animation: fadeInUp 0.5s ease both;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.suggestion-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.35);
}

/* ── Section headings ─────────────────────────────────────────────────── */
.sg-section-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #c4b5fd;
    border-left: 3px solid #7c3aed;
    padding-left: 0.75rem;
    margin-bottom: 1rem;
}

/* ── Divider ──────────────────────────────────────────────────────────── */
.sg-divider {
    border: none;
    border-top: 1px solid #2d2f3d;
    margin: 1.5rem 0;
}

/* ── Keyframes ────────────────────────────────────────────────────────── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes glowIn {
    0%   { opacity: 0; box-shadow: 0 0 0px rgba(124,58,237,0); }
    60%  { box-shadow: 0 0 18px rgba(124,58,237,0.35); }
    100% { opacity: 1; box-shadow: 0 0 6px rgba(124,58,237,0.15); }
}

@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.5); }
    50%       { box-shadow: 0 0 0 6px rgba(239,68,68,0); }
}

@keyframes arrowBounce {
    0%, 100% { transform: translateX(0); }
    50%       { transform: translateX(5px); }
}
.arrow-anim {
    display: inline-block;
    animation: arrowBounce 0.9s ease infinite;
}

/* ── Input boxes ──────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input {
    background: #12141a !important;
    color: #e2e8f0 !important;
    border: 1px solid #2d2f3d !important;
    border-radius: 8px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.25) !important;
}

/* ── Dataframe ────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}

/* ── Hide default Streamlit branding ──────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def get_risk_color(risk_level: Optional[str]) -> str:
    """Return hex color for a risk level string."""
    mapping = {
        "LOW":    "#22c55e",
        "MEDIUM": "#f59e0b",
        "HIGH":   "#ef4444",
    }
    return mapping.get((risk_level or "").upper(), "#94a3b8")


def render_risk_badge(risk_level: Optional[str]) -> str:
    """Return an HTML string with a styled risk badge."""
    level = (risk_level or "UNKNOWN").upper()
    css_class = {
        "LOW":    "badge-low",
        "MEDIUM": "badge-medium",
        "HIGH":   "badge-high",
    }.get(level, "badge")
    return f'<span class="badge {css_class}">{level}</span>'


def format_confidence_interval(ci: Optional[list]) -> str:
    """Format a [lower, upper] confidence interval list nicely."""
    if not ci or len(ci) < 2:
        return "N/A"
    try:
        lower, upper = float(ci[0]), float(ci[1])
        return f"{lower:.1f} \u2013 {upper:.1f} hrs"
    except (TypeError, ValueError):
        return "N/A"


def _plotly_dark_layout(**kwargs) -> dict:
    """Base Plotly layout dict for dark theme."""
    base = dict(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        font=dict(color="#e2e8f0", family="Inter, Segoe UI, sans-serif"),
        xaxis=dict(
            gridcolor="#2d2f3d",
            zerolinecolor="#2d2f3d",
            color="#94a3b8",
        ),
        yaxis=dict(
            gridcolor="#2d2f3d",
            zerolinecolor="#2d2f3d",
            color="#94a3b8",
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            bgcolor="rgba(28,30,36,0.8)",
            bordercolor="#2d2f3d",
            borderwidth=1,
        ),
    )
    base.update(kwargs)
    return base


def _safe_get(url: str, **kwargs):
    """GET with ConnectionError handling. Returns (data_dict_or_None, error_str_or_None)."""
    try:
        resp = requests.get(url, timeout=10, **kwargs)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "API unavailable - check that the backend is running"
    except requests.exceptions.HTTPError as exc:
        return None, f"API error {exc.response.status_code}: {exc.response.text}"
    except Exception as exc:
        return None, f"Unexpected error: {exc}"


def _safe_post(url: str, payload: dict, **kwargs):
    """POST with ConnectionError handling. Returns (data_dict_or_None, error_str_or_None)."""
    try:
        resp = requests.post(url, json=payload, timeout=15, **kwargs)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "API unavailable - check that the backend is running"
    except requests.exceptions.HTTPError as exc:
        return None, f"API error {exc.response.status_code}: {exc.response.text}"
    except Exception as exc:
        return None, f"Unexpected error: {exc}"


# ---------------------------------------------------------------------------
# PAGE 1 — Sprint Overview
# ---------------------------------------------------------------------------

def _generate_burndown_mock(days_total: int = 14, total_points: int = 42) -> tuple[list, list, list]:
    """Generate realistic planned vs actual burndown data."""
    random.seed(days_total + total_points)  # reproducible per sprint shape
    planned = [round(total_points - total_points * (i / days_total), 1) for i in range(days_total + 1)]
    actual: list[float] = [float(total_points)]
    for i in range(1, days_total + 1):
        # Actual has some variance — sometimes slower, sometimes faster
        ideal_delta = total_points / days_total
        noise = random.uniform(-ideal_delta * 0.6, ideal_delta * 0.3)
        prev = actual[-1]
        new_val = max(0.0, prev - ideal_delta + noise)
        actual.append(round(new_val, 1))
    days_axis = list(range(days_total + 1))
    return days_axis, planned, actual


def render_sprint_overview() -> None:
    """Page 1: Sprint health overview with burndown chart and risk gauge."""
    st.markdown('<div class="sg-section-title">Sprint Overview</div>', unsafe_allow_html=True)

    demo_mode = st.session_state.get("demo_mode", True)

    # ── Sprint ID input ────────────────────────────────────────────────────
    col_input, col_btn, col_demo, _ = st.columns([2, 1, 1, 2])
    with col_input:
        sprint_id = st.text_input(
            "Sprint ID",
            value=st.session_state.get("overview_sprint_id", ""),
            placeholder="e.g. SPRINT-42",
            key="overview_sprint_id_input",
            label_visibility="collapsed",
        )
    with col_btn:
        load_clicked = st.button("\U0001f50d Load Sprint Data", key="load_sprint_btn")
    with col_demo:
        demo_load_clicked = st.button("\U0001f3ad Load Demo Data", key="load_demo_btn")

    # Handle "Load Demo Data" button — auto-fills SPRINT-42 and loads mock data immediately
    if demo_load_clicked and demo_mode:
        st.session_state["overview_sprint_id"] = "SPRINT-42"
        st.session_state["overview_data"] = _mock_sprint_risk("SPRINT-42")
        st.rerun()
    elif demo_load_clicked and not demo_mode:
        st.info("Enable \U0001f3ad Demo Mode in the sidebar to use Load Demo Data.")

    if load_clicked and sprint_id:
        st.session_state["overview_sprint_id"] = sprint_id
        if demo_mode:
            st.session_state["overview_data"] = _mock_sprint_risk(sprint_id)
        else:
            with st.spinner("Fetching sprint data\u2026"):
                data, err = _safe_get(f"{API_BASE}/api/v1/sprint/{sprint_id}/risk")
            if err:
                st.error(err)
                st.session_state.pop("overview_data", None)
            else:
                st.session_state["overview_data"] = data
    elif load_clicked and not sprint_id:
        st.warning("Please enter a Sprint ID first.")

    data = st.session_state.get("overview_data")

    if data is None:
        st.markdown(
            '<div class="sg-card" style="text-align:center;padding:3rem 1rem;color:#475569;">'
            "Enter a Sprint ID and click <strong>Load Sprint Data</strong> to begin."
            " In Demo Mode, click <strong>\U0001f3ad Load Demo Data</strong> for instant results."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # Show demo banner when using mock data
    if demo_mode:
        st.info("\U0001f3ad Demo Mode \u2014 showing mock data")

    # ── Metric cards ───────────────────────────────────────────────────────
    days_remaining    = data.get("days_remaining", "\u2014")
    velocity_trend    = data.get("velocity_trend", None)
    bug_hours_today   = data.get("bug_hours_added_today", "\u2014")
    risk_level        = data.get("risk_level", "UNKNOWN")
    risk_score        = data.get("risk_score", 0.0)
    factors           = data.get("factors", [])

    velocity_display = f"{velocity_trend:+.1f}" if velocity_trend is not None else "\u2014"
    velocity_color   = "#22c55e" if (velocity_trend or 0) >= 0 else "#ef4444"

    st.markdown("<hr class='sg-divider'>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Days Remaining</div>
                <div class="metric-value">{days_remaining}</div>
                <div class="metric-delta" style="color:#94a3b8;">calendar days left</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        velocity_direction = "\u2191 above average" if (velocity_trend or 0) >= 0 else "\u2193 below average"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Velocity Trend</div>
                <div class="metric-value" style="color:{velocity_color};">{velocity_display}</div>
                <div class="metric-delta" style="color:{velocity_color};">
                    {velocity_direction}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        bh_color = "#ef4444" if float(bug_hours_today or 0) > 4 else "#f59e0b" if float(bug_hours_today or 0) > 1 else "#22c55e"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Bug Hours Today</div>
                <div class="metric-value" style="color:{bh_color};">{float(bug_hours_today or 0):.1f}h</div>
                <div class="metric-delta" style="color:#94a3b8;">hours added by new bugs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='sg-divider'>", unsafe_allow_html=True)

    # ── Burndown + Risk gauge side by side ─────────────────────────────────
    chart_col, gauge_col = st.columns([3, 2])

    with chart_col:
        st.markdown('<div class="sg-section-title">Burndown Chart</div>', unsafe_allow_html=True)
        days_total = max(int(days_remaining) + 3, 14) if str(days_remaining).isdigit() else 14
        days_axis, planned, actual = _generate_burndown_mock(days_total=days_total)

        fig_burndown = go.Figure()
        fig_burndown.add_trace(go.Scatter(
            x=days_axis, y=planned,
            mode="lines",
            name="Planned",
            line=dict(color="#7c3aed", width=2.5, dash="dot"),
            hovertemplate="Day %{x}: %{y} pts planned<extra></extra>",
        ))
        fig_burndown.add_trace(go.Scatter(
            x=days_axis, y=actual,
            mode="lines+markers",
            name="Actual",
            line=dict(color="#06b6d4", width=2.5),
            marker=dict(size=5, color="#06b6d4"),
            hovertemplate="Day %{x}: %{y} pts remaining<extra></extra>",
        ))
        # Shade the ideal zone
        fig_burndown.add_trace(go.Scatter(
            x=days_axis + days_axis[::-1],
            y=planned + [0] * len(days_axis),
            fill="toself",
            fillcolor="rgba(124,58,237,0.06)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig_burndown.update_layout(
            **_plotly_dark_layout(
                title=dict(text="Sprint Burndown \u2014 Planned vs Actual", font=dict(size=14, color="#c4b5fd")),
                xaxis_title="Sprint Day",
                yaxis_title="Story Points Remaining",
            )
        )
        st.plotly_chart(fig_burndown, use_container_width=True)

    with gauge_col:
        st.markdown('<div class="sg-section-title">Risk Score</div>', unsafe_allow_html=True)
        risk_color = get_risk_color(risk_level)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(float(risk_score), 3),
            number=dict(suffix="", font=dict(size=36, color=risk_color)),
            title=dict(
                text=f"<b>{risk_level}</b><br><span style='font-size:0.8em;color:#94a3b8;'>Risk Level</span>",
                font=dict(size=16, color=risk_color),
            ),
            gauge=dict(
                axis=dict(
                    range=[0, 1],
                    tickwidth=1,
                    tickcolor="#94a3b8",
                    dtick=0.2,
                ),
                bar=dict(color=risk_color, thickness=0.25),
                bgcolor="#161b22",
                borderwidth=1,
                bordercolor="#2d2f3d",
                steps=[
                    dict(range=[0.0, 0.4], color="rgba(34,197,94,0.18)"),
                    dict(range=[0.4, 0.7], color="rgba(245,158,11,0.18)"),
                    dict(range=[0.7, 1.0], color="rgba(239,68,68,0.18)"),
                ],
                threshold=dict(
                    line=dict(color=risk_color, width=3),
                    thickness=0.75,
                    value=float(risk_score),
                ),
            ),
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0e1117",
            font=dict(color="#e2e8f0"),
            height=300,
            margin=dict(l=20, r=20, t=60, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Risk factors ───────────────────────────────────────────────────────
    if factors:
        st.markdown("<hr class='sg-divider'>", unsafe_allow_html=True)
        st.markdown('<div class="sg-section-title">Risk Factors</div>', unsafe_allow_html=True)
        factor_html = "".join(
            f"<li style='margin-bottom:0.4rem;color:#cbd5e1;'>{f}</li>"
            for f in factors
        )
        st.markdown(
            f"""
            <div class="sg-card">
                <ul style="margin:0;padding-left:1.4rem;line-height:1.7;">{factor_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# PAGE 2 — Bug Triage
# ---------------------------------------------------------------------------

def render_bug_triage() -> None:
    """Page 2: Submit a bug and view assignment, effort, and sprint impact."""
    st.markdown('<div class="sg-section-title">Bug Triage</div>', unsafe_allow_html=True)

    demo_mode = st.session_state.get("demo_mode", True)

    # ── Input form ─────────────────────────────────────────────────────────
    st.markdown(
        '<div class="sg-card" style="margin-bottom:1.5rem;">',
        unsafe_allow_html=True,
    )

    title       = st.text_input("Bug Title *", placeholder="e.g. NPE when saving empty form", key="triage_title")
    description = st.text_area("Description *", height=150, placeholder="Full reproduction steps and observed behaviour\u2026", key="triage_desc")

    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        reporter = st.text_input("Reporter (optional)", placeholder="e.g. jane.doe", key="triage_reporter")
    with opt_col2:
        sprint_id = st.text_input("Sprint ID (optional)", placeholder="e.g. SPRINT-42", key="triage_sprint")

    analyze_clicked = st.button("\U0001f52c Analyze Bug", key="triage_analyze_btn")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Validate + call API (or mock) ──────────────────────────────────────
    if analyze_clicked:
        if not title.strip():
            st.warning("Bug Title is required.")
        elif not description.strip():
            st.warning("Description is required.")
        else:
            if demo_mode:
                st.info("\U0001f3ad Demo Mode \u2014 showing mock data")
                result = _mock_bug_analyze(title.strip(), description.strip())
                st.session_state["triage_result"] = result
            else:
                payload = {
                    "title":       title.strip(),
                    "description": description.strip(),
                    "reporter":    reporter.strip() or "unknown",
                    "sprint_id":   sprint_id.strip(),
                    "file_paths":  [],
                }
                placeholder = st.empty()
                with placeholder.container():
                    with st.spinner("Analyzing bug\u2026"):
                        time.sleep(0.3)  # brief animation pause
                        result, err = _safe_post(f"{API_BASE}/api/v1/bugs/analyze", payload)
                placeholder.empty()

                if err:
                    st.error(err)
                    st.session_state.pop("triage_result", None)
                else:
                    st.session_state["triage_result"] = result

    # ── Display results ────────────────────────────────────────────────────
    result = st.session_state.get("triage_result")
    if result is None:
        return

    st.markdown("<hr class='sg-divider'>", unsafe_allow_html=True)
    st.markdown(
        '<div class="sg-section-title">Analysis Results</div>',
        unsafe_allow_html=True,
    )

    res_col1, res_col2, res_col3 = st.columns([1, 1, 1])

    # Col 1 — Assigned Developer
    with res_col1:
        assigned_dev    = result.get("assigned_dev", "Unknown")
        conf_raw        = result.get("assignment_confidence", 0.0)
        conf_pct        = float(conf_raw) * 100

        st.markdown(
            f"""
            <div class="sg-card" style="min-height:140px;">
                <div class="metric-label">Assigned Developer</div>
                <div style="font-size:1.3rem;font-weight:700;color:#c4b5fd;margin:0.5rem 0;">{assigned_dev}</div>
                <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:0.4rem;">Confidence: {conf_pct:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(int(conf_pct), 100), text=f"{conf_pct:.1f}% confidence")

    # Col 2 — Top 3 developers table
    with res_col2:
        top3 = result.get("top3_devs", [])
        st.markdown(
            '<div class="metric-label" style="margin-bottom:0.5rem;">Top 3 Candidates</div>',
            unsafe_allow_html=True,
        )
        if top3:
            df = pd.DataFrame(
                [
                    {"Developer": d.get("dev", "\u2014"), "Probability": f"{float(d.get('probability', 0)) * 100:.1f}%"}
                    for d in top3
                ]
            )
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No candidates returned.")

    # Col 3 — Effort estimate
    with res_col3:
        effort_obj  = result.get("effort_estimate") or {}
        hours_raw   = effort_obj.get("hours")
        ci_raw      = effort_obj.get("confidence_interval")
        hours_disp  = f"{float(hours_raw):.1f} hrs" if hours_raw is not None else "N/A"
        ci_disp     = format_confidence_interval(ci_raw)

        st.markdown(
            f"""
            <div class="sg-card" style="min-height:140px;">
                <div class="metric-label">Effort Estimate</div>
                <div style="font-size:1.6rem;font-weight:700;color:#06b6d4;margin:0.5rem 0;">{hours_disp}</div>
                <div style="font-size:0.78rem;color:#94a3b8;">95% CI: {ci_disp}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Sprint impact ──────────────────────────────────────────────────────
    impact = result.get("sprint_impact") or {}
    if impact:
        st.markdown("<hr class='sg-divider'>", unsafe_allow_html=True)
        st.markdown(
            '<div class="sg-section-title">Sprint Impact</div>',
            unsafe_allow_html=True,
        )

        risk_before     = impact.get("risk_before") or "UNKNOWN"
        risk_after      = impact.get("risk_after")  or "UNKNOWN"
        replan_flag     = impact.get("replan_suggested", False)
        impact_factors  = impact.get("factors", [])

        badge_before    = render_risk_badge(risk_before)
        badge_after     = render_risk_badge(risk_after)
        replan_badge    = (
            '<span class="badge" style="background:#1c1917;color:#fb923c;border:1px solid #fb923c;">\u26a0 Replan Suggested</span>'
            if replan_flag else
            '<span class="badge" style="background:#14532d;color:#22c55e;border:1px solid #22c55e;">\u2713 No Replan Needed</span>'
        )

        st.markdown(
            f"""
            <div class="sg-card">
                <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;margin-bottom:0.75rem;">
                    <div>{badge_before}</div>
                    <div class="arrow-anim" style="font-size:1.4rem;color:#7c3aed;">\u2192</div>
                    <div>{badge_after}</div>
                    <div style="margin-left:auto;">{replan_badge}</div>
                </div>
                {"<ul style='margin:0;padding-left:1.4rem;color:#94a3b8;font-size:0.85rem;'>" + "".join(f"<li>{f}</li>" for f in impact_factors) + "</ul>" if impact_factors else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# PAGE 3 — Re-planner
# ---------------------------------------------------------------------------

def render_replanner() -> None:
    """Page 3: Check sprint risk and get re-planning suggestions."""
    st.markdown('<div class="sg-section-title">Sprint Re-planner</div>', unsafe_allow_html=True)

    demo_mode = st.session_state.get("demo_mode", True)

    # Session state init
    if "replan_stories" not in st.session_state:
        st.session_state["replan_stories"] = []
    if "replan_sprint_data" not in st.session_state:
        st.session_state["replan_sprint_data"] = None
    if "replan_results" not in st.session_state:
        st.session_state["replan_results"] = None

    # ── Sprint ID + risk check ─────────────────────────────────────────────
    rp_col1, rp_col2, _ = st.columns([2, 1, 3])
    with rp_col1:
        sprint_id = st.text_input(
            "Sprint ID",
            value=st.session_state.get("replan_sprint_id", ""),
            placeholder="e.g. SPRINT-42",
            key="replan_sprint_input",
            label_visibility="collapsed",
        )
    with rp_col2:
        check_clicked = st.button("\U0001f4ca Check Sprint Risk", key="replan_check_btn")

    if check_clicked and sprint_id:
        st.session_state["replan_sprint_id"] = sprint_id
        if demo_mode:
            data = _mock_sprint_risk(sprint_id)
            st.session_state["replan_sprint_data"] = data
            st.session_state["replan_results"] = None
        else:
            with st.spinner("Fetching sprint risk\u2026"):
                data, err = _safe_get(f"{API_BASE}/api/v1/sprint/{sprint_id}/risk")
            if err:
                st.error(err)
                st.session_state["replan_sprint_data"] = None
            else:
                st.session_state["replan_sprint_data"] = data
                # Reset previous results when sprint changes
                st.session_state["replan_results"] = None
    elif check_clicked and not sprint_id:
        st.warning("Please enter a Sprint ID first.")

    sprint_data = st.session_state.get("replan_sprint_data")

    # Show demo banner when using mock data
    if sprint_data and demo_mode:
        st.info("\U0001f3ad Demo Mode \u2014 showing mock data")

    # ── Current risk badge ─────────────────────────────────────────────────
    if sprint_data:
        current_risk  = sprint_data.get("risk_level", "UNKNOWN")
        current_score = sprint_data.get("risk_score", 0.0)
        badge_html    = render_risk_badge(current_risk)
        st.markdown(
            f"""
            <div class="sg-card" style="display:flex;align-items:center;gap:1rem;margin-top:0.5rem;">
                <span style="color:#94a3b8;font-size:0.85rem;">Current Risk:</span>
                {badge_html}
                <span style="color:#94a3b8;font-size:0.85rem;">Score: {float(current_score):.3f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='sg-divider'>", unsafe_allow_html=True)

    # ── Add stories section ────────────────────────────────────────────────
    st.markdown('<div class="sg-section-title">Stories in Scope</div>', unsafe_allow_html=True)

    with st.expander("\u2795 Add Story", expanded=False):
        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            new_story_id = st.text_input("Story ID", placeholder="STORY-001", key="new_story_id")
            new_priority = st.number_input("Priority (1\u20135)", min_value=1, max_value=5, value=3, key="new_priority")
        with s_col2:
            new_points   = st.number_input("Story Points (1\u201313)", min_value=1, max_value=13, value=3, key="new_points")
            new_effort   = st.number_input("Effort Hours", min_value=0.0, value=8.0, step=0.5, key="new_effort")
        with s_col3:
            new_must_have = st.checkbox("Must Have", value=False, key="new_must_have")
            st.write("")  # spacer
            add_story_btn = st.button("Add to List", key="add_story_btn")

        if add_story_btn:
            if not new_story_id.strip():
                st.warning("Story ID is required.")
            else:
                # Check for duplicates
                existing_ids = [s["id"] for s in st.session_state["replan_stories"]]
                if new_story_id.strip() in existing_ids:
                    st.warning(f"Story '{new_story_id.strip()}' is already in the list.")
                else:
                    st.session_state["replan_stories"].append({
                        "id":            new_story_id.strip(),
                        "story_points":  int(new_points),
                        "priority":      int(new_priority),
                        "effort_hours":  float(new_effort),
                        "must_have":     bool(new_must_have),
                    })
                    st.success(f"Added story {new_story_id.strip()}")

    # Show stories table
    stories = st.session_state.get("replan_stories", [])
    if stories:
        stories_df = pd.DataFrame(stories).rename(columns={
            "id":            "Story ID",
            "story_points":  "Points",
            "priority":      "Priority",
            "effort_hours":  "Effort (hrs)",
            "must_have":     "Must Have",
        })
        st.dataframe(stories_df, use_container_width=True, hide_index=True)

        # Remove story option
        remove_options = ["\u2014 select to remove \u2014"] + [s["id"] for s in stories]
        remove_id = st.selectbox("Remove a story", options=remove_options, key="remove_story_select")
        if remove_id != "\u2014 select to remove \u2014":
            if st.button(f"Remove {remove_id}", key="remove_story_btn"):
                st.session_state["replan_stories"] = [
                    s for s in st.session_state["replan_stories"] if s["id"] != remove_id
                ]
                st.rerun()
    else:
        st.markdown(
            '<div style="color:#475569;font-size:0.85rem;padding:0.5rem 0;">No stories added yet. Use the expander above.</div>',
            unsafe_allow_html=True,
        )

    # ── Capacity + run button ──────────────────────────────────────────────
    st.markdown("<hr class='sg-divider'>", unsafe_allow_html=True)

    capacity_col, _ = st.columns([2, 4])
    with capacity_col:
        available_capacity = st.number_input(
            "Available Capacity (hours)",
            min_value=1.0,
            value=80.0,
            step=4.0,
            key="available_capacity",
        )

    # Determine if re-planner button should be shown
    has_stories      = len(stories) > 0
    sprint_risk_high = (sprint_data or {}).get("risk_level", "").upper() == "HIGH"
    show_replan_btn  = has_stories or sprint_risk_high

    if not show_replan_btn:
        st.info("Add stories or check a HIGH-risk sprint to enable the Re-planner.")
    else:
        if not sprint_id:
            st.warning("Enter a Sprint ID and click 'Check Sprint Risk' before running the Re-planner.")
        elif not has_stories:
            st.info("Add at least one story before running the Re-planner.")
        else:
            run_clicked = st.button("\U0001f680 Run Re-planner", key="run_replan_btn")
            if run_clicked:
                current_sprint = st.session_state.get("replan_sprint_id", sprint_id)
                if demo_mode:
                    st.info("\U0001f3ad Demo Mode \u2014 showing mock re-plan suggestions")
                    result = _mock_replan(current_sprint)
                    st.session_state["replan_results"] = result
                else:
                    payload = {
                        "stories":                    stories,
                        "available_capacity_hours":   float(available_capacity),
                    }
                    placeholder = st.empty()
                    with placeholder.container():
                        with st.spinner("Running re-planning algorithm\u2026"):
                            time.sleep(0.4)
                            result, err = _safe_post(
                                f"{API_BASE}/api/v1/sprint/{current_sprint}/replan",
                                payload,
                            )
                    placeholder.empty()

                    if err:
                        st.error(err)
                        st.session_state["replan_results"] = None
                    else:
                        st.session_state["replan_results"] = result

    # ── Render suggestions ─────────────────────────────────────────────────
    replan_data = st.session_state.get("replan_results")
    if replan_data is None:
        return

    st.markdown("<hr class='sg-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="sg-section-title">Re-planning Suggestions</div>', unsafe_allow_html=True)

    suggestions      = replan_data.get("suggestions", [])
    recommended_id   = replan_data.get("recommended", "")
    final_risk       = replan_data.get("current_risk", "UNKNOWN")

    if not suggestions:
        st.info("No suggestions returned from the Re-planner.")
        return

    for suggestion in suggestions:
        s_id           = suggestion.get("id", "\u2014")
        action         = suggestion.get("action", "\u2014")
        pts_removed    = suggestion.get("story_points_removed", 0)
        proj_risk      = suggestion.get("projected_risk", "UNKNOWN")
        proj_score     = suggestion.get("projected_risk_score", 0.0)
        note           = suggestion.get("note") or ""
        is_recommended = (s_id == recommended_id)

        border_color   = get_risk_color(proj_risk)
        proj_badge     = render_risk_badge(proj_risk)
        rec_badge      = (
            '<span class="badge badge-recommended">\u2b50 Recommended</span>'
            if is_recommended else ""
        )

        # Score progress bar value (0–100)
        bar_val = max(0, min(100, int(float(proj_score) * 100)))

        st.markdown(
            f"""
            <div class="suggestion-card" style="border:1px solid {border_color}40;border-left:4px solid {border_color};">
                <div style="display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;margin-bottom:0.75rem;">
                    <span style="font-weight:700;color:#e2e8f0;font-size:1rem;">{s_id}</span>
                    {rec_badge}
                    {proj_badge}
                    <span style="margin-left:auto;font-size:0.8rem;color:#94a3b8;">
                        {pts_removed} pts removed
                    </span>
                </div>
                <div style="color:#cbd5e1;margin-bottom:0.5rem;font-size:0.9rem;">{action}</div>
                {"<div style='color:#94a3b8;font-size:0.8rem;font-style:italic;margin-bottom:0.5rem;'>" + note + "</div>" if note else ""}
                <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:0.25rem;">
                    Projected Risk Score: {float(proj_score):.3f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Progress bar for projected risk score
        st.progress(bar_val, text=f"Projected risk: {float(proj_score):.3f}")

        # Accept button
        accept_key   = f"accept_btn_{s_id}"
        accepted_key = f"accepted_{s_id}"

        if st.session_state.get(accepted_key):
            st.success(f"\u2705 Suggestion '{s_id}' accepted \u2014 sprint updated!")
        else:
            if st.button(f"Accept '{s_id}'", key=accept_key):
                st.session_state[accepted_key] = True
                # Refresh sprint risk after accepting (skip API call in demo mode)
                current_sprint = st.session_state.get("replan_sprint_id", "")
                if current_sprint:
                    if demo_mode:
                        refresh_data = _mock_sprint_risk(current_sprint)
                        st.session_state["replan_sprint_data"] = refresh_data
                    else:
                        refresh_data, _ = _safe_get(f"{API_BASE}/api/v1/sprint/{current_sprint}/risk")
                        if refresh_data:
                            st.session_state["replan_sprint_data"] = refresh_data
                st.rerun()

        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> str:
    """Render sidebar navigation and return the selected page name."""
    with st.sidebar:
        # Logo / title
        st.markdown(
            """
            <div style="text-align:center;padding:1rem 0 0.5rem 0;">
                <div style="font-size:2.4rem;">\U0001f6e1\ufe0f</div>
                <div style="font-size:1.3rem;font-weight:800;color:#c4b5fd;letter-spacing:0.04em;">SprintGuard</div>
                <div style="font-size:0.78rem;color:#64748b;margin-top:0.15rem;">Sprint Health Monitor</div>
            </div>
            <hr style="border:none;border-top:1px solid #2d2f3d;margin:0.75rem 0;">
            """,
            unsafe_allow_html=True,
        )

        # Demo mode toggle — default ON
        st.sidebar.toggle("\U0001f3ad Demo Mode", value=True, key="demo_mode")

        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

        page = st.selectbox(
            "Navigate",
            options=["Sprint Overview", "Bug Triage", "Re-planner"],
            key="nav_page",
            label_visibility="collapsed",
        )

        # Spacer + footer
        st.markdown(
            """
            <div style="position:fixed;bottom:1.5rem;left:0;width:240px;text-align:center;
                        font-size:0.7rem;color:#334155;padding:0 1rem;">
                SprintGuard v0.1.0<br>
                <span style="color:#2d2f3d;">built with Streamlit + FastAPI</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return page


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Application entry point \u2014 renders sidebar and selected page."""
    page = render_sidebar()

    if page == "Sprint Overview":
        render_sprint_overview()
    elif page == "Bug Triage":
        render_bug_triage()
    elif page == "Re-planner":
        render_replanner()


if __name__ == "__main__":
    main()
