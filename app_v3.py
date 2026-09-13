"""
Olfactos - Pilot Dashboard (v3, demo-ready for De Laat)
Builds on the same Supabase connection and live-fragment refresh as v2.
Adds: a dark cyan-accent theme, tabs, a Decision Support panel, a raw-vs-
compensated de-drift chart, and a real (not simulated) event log.

HONESTY NOTE, read before demoing:
The "Decision Support" tab shows an ESTIMATE based on a simple, transparent
rule (current signal level + its recent slope, extrapolated to a threshold).
It is NOT a trained machine-learning prediction yet -- we don't have enough
pilot data for that. The tab says so openly. Once more labelled pilot data
is collected, this panel is where the real trained model's output will go,
using the same visual layout. Do not describe this as "AI-validated" to
De Laat; describe it as "an early decision-support view, which becomes a
trained prediction as we collect more pilot data."

SETUP: same as before.
  pip install streamlit supabase pandas plotly
  streamlit run app_v3.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from supabase import create_client
from datetime import datetime, timedelta

# ---- FILL THESE IN (same anon key as the firmware) ----
SUPABASE_URL = "https://qhlrtcidqquwqxnqggyk.supabase.co"
SUPABASE_ANON_KEY = "your-anon-public-key-here"

# ---- Tunable thresholds for the heuristic decision-support tier ----
# These are placeholders until the pilot gives us real calibrated numbers.
# They are based on the |delta_baseline| (signal above the empty-chamber
# baseline). Raise/lower these once you see real De Laat data ranges.
ADVISORY_THRESHOLD = 0.15   # volts above baseline -> "keep an eye on this batch"
ALARM_THRESHOLD = 0.35      # volts above baseline -> "inspect soon"

st.set_page_config(page_title="Olfactos Dashboard", layout="wide", page_icon="🍌")
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ============================================================
# THEME  (dark, cyan-accent -- matches the Olfactos brand mockup)
# ============================================================
st.markdown("""
<style>
:root {
  --bg-main:#050914; --bg-panel:#0a1122; --bg-card:#10182b;
  --border:rgba(0,200,210,0.15); --text-muted:#8a9bb3;
  --cyan:#00d2d3; --ok:#10ac84; --warn:#feca57; --crit:#ff6b6b;
}
.stApp { background-color: var(--bg-main); }
section[data-testid="stSidebar"] { background-color: var(--bg-panel); }
div[data-testid="stMetric"] {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 16px;
}
div[data-testid="stMetricLabel"] { color: var(--text-muted); }
.status-pill {
  display:inline-flex; align-items:center; gap:8px; font-family:monospace;
  font-size:13px; padding:6px 14px; border-radius:20px; border:1px solid;
}
.pill-ok { border-color:var(--ok); color:var(--ok); background:rgba(16,172,132,0.08); }
.pill-warn { border-color:var(--warn); color:var(--warn); background:rgba(254,202,87,0.08); }
.pill-crit { border-color:var(--crit); color:var(--crit); background:rgba(255,107,107,0.08); }
.decision-card {
  background: linear-gradient(145deg, var(--bg-card), var(--bg-panel));
  border: 1px solid var(--border); border-left: 4px solid var(--cyan);
  border-radius: 12px; padding: 20px 24px; margin-bottom: 16px;
}
.decision-card.crit { border-left-color: var(--crit); }
.decision-card.warn { border-left-color: var(--warn); }
.action-box {
  background: rgba(0,0,0,0.25); border: 1px dashed var(--text-muted);
  border-radius: 8px; padding: 12px 16px; margin-top: 14px; font-size: 14px;
}
.caveat { font-size:12px; color:var(--text-muted); margin-top:10px; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div style="padding: 6px 0 18px 0;">
        <h1 style="margin-bottom:0;">🍌 Olfactos <span style="color:#00d2d3;">Pilot Dashboard</span></h1>
        <p style="color:#8a9bb3; margin-top:4px;">Early spoilage-signal monitoring — De Laat pilot</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Run selector
# ============================================================
runs_resp = supabase.table("experiment_runs").select("*").order("created_at", desc=True).execute()
runs = runs_resp.data
if not runs:
    st.warning("No runs found yet. Create a row in 'experiment_runs' in Supabase before starting a test.")
    st.stop()

run_labels = [f"{r['run_id']} — {r.get('fruit','?')} ({r.get('fruit_stage','?')}) @ {r.get('location','?')}" for r in runs]
selected_idx = st.selectbox("Select run", range(len(runs)), format_func=lambda i: run_labels[i])
selected_run_id = runs[selected_idx]["run_id"]


def line_chart(df, y_col, title, color, dash=False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df[y_col], mode="lines",
        line=dict(color=color, width=2, dash="dash" if dash else "solid"),
        name=title,
    ))
    fig.update_layout(title=title, height=280, margin=dict(l=10, r=10, t=40, b=10),
                       template="plotly_dark", paper_bgcolor="#0a1122",
                       plot_bgcolor="#0a1122", showlegend=False)
    return fig


def dual_line_chart(df, raw_col, comp_col, title):
    """The key demo visual: raw (drifting) vs compensated (de-drifted) signal."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df[raw_col], mode="lines",
                              name="Raw signal", line=dict(color="#feca57", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df[comp_col], mode="lines",
                              name="Compensated (baseline removed)", line=dict(color="#00d2d3", width=2),
                              fill="tozeroy", fillcolor="rgba(0,210,211,0.08)"))
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10),
                       template="plotly_dark", paper_bgcolor="#0a1122", plot_bgcolor="#0a1122",
                       legend=dict(orientation="h", y=1.15))
    return fig


def compute_tier(latest_row):
    """Simple, transparent rule -- NOT a trained model. See honesty note at top."""
    mq3_delta = latest_row.get("mq3_delta_baseline") or 0
    mq138_delta = latest_row.get("mq138_delta_baseline") or 0
    signal = max(abs(mq3_delta), abs(mq138_delta))

    if signal >= ALARM_THRESHOLD:
        return "crit", signal
    elif signal >= ADVISORY_THRESHOLD:
        return "warn", signal
    return "ok", signal


def estimate_time_to_threshold(df, col, threshold, window_min=30):
    """Linear extrapolation from the last `window_min` minutes of data to
    estimate when `col` will cross `threshold`. Returns hours, or None if
    the signal is flat/falling (no crossing predicted)."""
    recent = df.dropna(subset=[col]).tail(window_min)
    if len(recent) < 5:
        return None
    t = np.arange(len(recent))
    y = recent[col].values
    slope, intercept = np.polyfit(t, y, 1)
    if slope <= 0:
        return None
    current = y[-1]
    if current >= threshold:
        return 0.0
    steps_needed = (threshold - current) / slope
    # each row is ~5s apart (matches the firmware's upload interval)
    minutes_needed = steps_needed * (5 / 60)
    return round(minutes_needed / 60, 1)


@st.fragment(run_every="10s")
def live_dashboard(run_id):
    response = (
        supabase.table("sensor_readings")
        .select("*")
        .eq("run_id", run_id)
        .order("device_timestamp_ms", desc=False)
        .execute()
    )
    data = response.data

    if not data:
        st.info("No readings yet for this run. Waiting for the node to send its first upload...")
        return

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    latest = df.iloc[-1]

    tier, signal_level = compute_tier(latest)
    pill_class = {"ok": "pill-ok", "warn": "pill-warn", "crit": "pill-crit"}[tier]
    pill_text = {"ok": "● SYSTEM NOMINAL", "warn": "● ADVISORY — MONITOR BATCH", "crit": "● ALARM — INSPECT SOON"}[tier]
    st.markdown(f'<span class="status-pill {pill_class}">{pill_text}</span>', unsafe_allow_html=True)
    st.write("")

    tab_live, tab_decision, tab_dedrift, tab_raw = st.tabs(
        ["Live Monitor", "Decision Support", "De-Drift & Trends", "Raw Data"]
    )

    # ---------------- TAB 1: LIVE MONITOR ----------------
    with tab_live:
        st.subheader("Current reading")
        cols = st.columns(6)
        cols[0].metric("VOC (SGP41)", f"{latest['sgp41_voc_raw']}")
        cols[1].metric("CO2", f"{latest['scd41_co2_ppm']:.0f} ppm" if pd.notna(latest['scd41_co2_ppm']) else "—")
        cols[2].metric("BME Gas", f"{latest['bme688_gas_ohm']:.0f} Ω" if pd.notna(latest['bme688_gas_ohm']) else "—")
        cols[3].metric("Temp (SHT45)", f"{latest['sht45_temp_c']:.1f}°C" if pd.notna(latest['sht45_temp_c']) else "—")
        cols[4].metric("Humidity (SHT45)", f"{latest['sht45_rh_pct']:.1f}%" if pd.notna(latest['sht45_rh_pct']) else "—")
        cols[5].metric("MQ3 / MQ138 (V)", f"{latest['mq3_ao_v']:.2f} / {latest['mq138_ao_v']:.2f}")
        st.caption(f"Node: {latest['node_id']} | Phase: {latest['phase']} | Last update: {latest['timestamp']}")
        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(line_chart(df, "sgp41_voc_raw", "VOC index (SGP41)", "#00d2d3"), use_container_width=True)
            st.plotly_chart(line_chart(df, "scd41_co2_ppm", "CO2 (respiration)", "#2e86de"), use_container_width=True)
        with c2:
            st.plotly_chart(line_chart(df, "sht45_temp_c", "Temperature (SHT45)", "#feca57"), use_container_width=True)
            st.plotly_chart(line_chart(df, "sht45_rh_pct", "Humidity (SHT45)", "#8a9bb3"), use_container_width=True)

    # ---------------- TAB 2: DECISION SUPPORT ----------------
    with tab_decision:
        css_class = {"ok": "", "warn": "warn", "crit": "crit"}[tier]
        headline = {
            "ok": "Batch tracking normally",
            "warn": "Signal rising above baseline — advisory",
            "crit": "Signal at alarm level — action recommended",
        }[tier]
        action = {
            "ok": "No action needed. Continue routine monitoring.",
            "warn": f"Elevated VOC signal detected ({signal_level:.2f} V above baseline). "
                    f"Consider a visual check of this batch on your next round.",
            "crit": f"Signal has crossed the alarm threshold ({signal_level:.2f} V above baseline). "
                    f"Recommend inspecting this batch now and prioritising it for sale or removal.",
        }[tier]

        eta_hours = estimate_time_to_threshold(df, "mq3_delta_baseline", ALARM_THRESHOLD)
        eta_text = (
            f"Estimated {eta_hours:.1f}h until alarm threshold at current trend"
            if eta_hours is not None else "Signal currently stable or falling — no threshold crossing projected"
        )

        st.markdown(f"""
        <div class="decision-card {css_class}">
            <div style="font-size:22px; font-weight:700; margin-bottom:6px;">{headline}</div>
            <div style="color:#8a9bb3; font-family:monospace; font-size:14px;">{eta_text}</div>
            <div class="action-box"><strong>Recommended action:</strong><br>{action}</div>
            <div class="caveat">
                This is an early, rule-based estimate from the current signal trend, shown for
                pilot demonstration. It is not yet a validated machine-learning prediction —
                that requires more labelled pilot data, which this deployment is collecting.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Event log")
        events = []
        if (df["phase"] == "banana_exposure").any():
            first_exposure = df[df["phase"] == "banana_exposure"].iloc[0]
            events.append((first_exposure["timestamp"], "Batch inserted — exposure phase started"))
        for label, col, thr in [("MQ3 advisory", "mq3_delta_baseline", ADVISORY_THRESHOLD),
                                  ("MQ3 alarm", "mq3_delta_baseline", ALARM_THRESHOLD)]:
            crossed = df[df[col] >= thr]
            if not crossed.empty:
                events.append((crossed.iloc[0]["timestamp"], f"{label} threshold crossed"))
        events.sort(key=lambda e: e[0])
        if events:
            for ts, text in events:
                st.markdown(f"`{ts.strftime('%H:%M:%S')}` — {text}")
        else:
            st.caption("No events yet for this run.")

    # ---------------- TAB 3: DE-DRIFT & TRENDS ----------------
    with tab_dedrift:
        st.caption(
            "Raw sensor voltage drifts with humidity (dashed). The compensated line "
            "(delta from the empty-chamber baseline) isolates the actual VOC signal."
        )
        st.plotly_chart(dual_line_chart(df, "mq3_ao_v", "mq3_delta_baseline",
                                         "MQ3 — raw vs. compensated"), use_container_width=True)
        st.plotly_chart(dual_line_chart(df, "mq138_ao_v", "mq138_delta_baseline",
                                         "MQ138 — raw vs. compensated"), use_container_width=True)

    # ---------------- TAB 4: RAW DATA ----------------
    with tab_raw:
        st.dataframe(df, use_container_width=True)
        st.download_button("Download this run as CSV", df.to_csv(index=False), file_name=f"{run_id}.csv")


live_dashboard(selected_run_id)
st.caption("Olfactos — early spoilage detection pilot. Refreshes automatically every 10 seconds.")
