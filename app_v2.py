"""
Olfactos - Pilot Dashboard (v2, matches supabase_schema_v3.sql)
Reads from the same "sensor_readings" table the ESP32 writes to.

SETUP:
1. pip install streamlit supabase pandas plotly
2. Fill in SUPABASE_URL and SUPABASE_ANON_KEY below
   (same anon key as the firmware -- read-only for the dashboard,
   thanks to the RLS "select" policy, so this is safe to use here too)
3. Run locally to test: streamlit run app.py
4. Deploy free: push to GitHub, then deploy on streamlit.io/cloud
   (NOT Vercel -- Streamlit Cloud is the correct free host for this)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from supabase import create_client
from datetime import datetime, timedelta

# ---- FILL THESE IN (same anon key as the firmware) ----
SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
SUPABASE_ANON_KEY = "your-anon-public-key-here"

st.set_page_config(page_title="Olfactos Pilot - De Laat", layout="wide", page_icon="🍌")
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

st.markdown(
    """
    <div style="padding: 10px 0 20px 0;">
        <h1 style="margin-bottom:0;">🍌 Olfactos Pilot Dashboard</h1>
        <p style="color:gray; margin-top:4px;">Live spoilage-signal monitoring — De Laat pilot</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<meta http-equiv='refresh' content='10'>", unsafe_allow_html=True)

# ---- Pick which run to look at ----
runs_resp = supabase.table("experiment_runs").select("*").order("created_at", desc=True).execute()
runs = runs_resp.data
if not runs:
    st.warning("No runs found yet. Create a row in 'experiment_runs' in Supabase before starting a test.")
    st.stop()

run_labels = [f"{r['run_id']} — {r.get('fruit','?')} ({r.get('fruit_stage','?')}) @ {r.get('location','?')}" for r in runs]
selected_idx = st.selectbox("Select run", range(len(runs)), format_func=lambda i: run_labels[i])
selected_run_id = runs[selected_idx]["run_id"]

# ---- Fetch readings for this run ----
response = (
    supabase.table("sensor_readings")
    .select("*")
    .eq("run_id", selected_run_id)
    .order("device_timestamp_ms", desc=False)
    .execute()
)
data = response.data

if not data:
    st.info("No readings yet for this run. Waiting for the node to send its first upload...")
    st.stop()

df = pd.DataFrame(data)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
latest = df.iloc[-1]

# ---- Current reading (big numbers) ----
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

# ---- Time-series charts ----
st.subheader("Trends for this run")

def line_chart(df, y_col, title, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df[y_col], mode="lines",
                              line=dict(color=color, width=2), name=title))
    fig.update_layout(title=title, height=280, margin=dict(l=10, r=10, t=40, b=10),
                       template="plotly_white", showlegend=False)
    return fig

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(line_chart(df, "sgp41_voc_raw", "VOC index (SGP41)", "#0f7b6c"), use_container_width=True)
    st.plotly_chart(line_chart(df, "mq3_ao_v", "MQ3 (ethanol) — volts", "#d97706"), use_container_width=True)
    st.plotly_chart(line_chart(df, "scd41_co2_ppm", "CO2 (respiration)", "#2563eb"), use_container_width=True)
    st.plotly_chart(line_chart(df, "mq3_delta_baseline", "MQ3 delta from baseline", "#b45309"), use_container_width=True)

with c2:
    st.plotly_chart(line_chart(df, "mq138_ao_v", "MQ138 (broad VOC) — volts", "#dc2626"), use_container_width=True)
    st.plotly_chart(line_chart(df, "sht45_temp_c", "Temperature (SHT45)", "#7c3aed"), use_container_width=True)
    st.plotly_chart(line_chart(df, "sht45_rh_pct", "Humidity (SHT45)", "#0891b2"), use_container_width=True)
    st.plotly_chart(line_chart(df, "mq138_delta_baseline", "MQ138 delta from baseline", "#991b1b"), use_container_width=True)

st.divider()
with st.expander("Raw data (for export / ML use)"):
    st.dataframe(df, use_container_width=True)
    st.download_button("Download this run as CSV", df.to_csv(index=False), file_name=f"{selected_run_id}.csv")

st.caption("Olfactos — early spoilage detection pilot. Refreshes automatically every 10 seconds.")
