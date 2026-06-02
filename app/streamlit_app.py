"""
Traffic Demand Prediction — Interactive Demo
Glass-morphism UI · loads precomputed profiles (no raw data needed)
"""

import json as _json
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_DEC = {c: i for i, c in enumerate(_B32)}


def decode_geohash(gh: str) -> tuple:
    """Decode a geohash6 string to (latitude, longitude)."""
    lat, lon, is_lon = [-90.0, 90.0], [-180.0, 180.0], True
    for ch in str(gh).lower():
        cd = _DEC.get(ch)
        if cd is None:
            continue
        for mask in (16, 8, 4, 2, 1):
            if is_lon:
                mid = (lon[0] + lon[1]) / 2
                lon[0 if cd & mask else 1] = mid
            else:
                mid = (lat[0] + lat[1]) / 2
                lat[0 if cd & mask else 1] = mid
            is_lon = not is_lon
    return (lat[0] + lat[1]) / 2, (lon[0] + lon[1]) / 2


st.set_page_config(
    page_title="Traffic Demand Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS (same glass-morphism multi-accent theme)
# ---------------------------------------------------------------------------
CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    .stApp {
        background: #0c1220;
        background-image:
            radial-gradient(ellipse 80% 60% at 10% 20%, rgba(56,189,248,0.07) 0%, transparent 60%),
            radial-gradient(ellipse 60% 50% at 85% 15%, rgba(167,139,250,0.06) 0%, transparent 55%),
            radial-gradient(ellipse 70% 60% at 50% 90%, rgba(244,114,182,0.04) 0%, transparent 50%),
            radial-gradient(ellipse 50% 40% at 20% 70%, rgba(52,211,153,0.035) 0%, transparent 50%);
        font-family: 'Inter', sans-serif;
    }
    #MainMenu, footer, header {visibility: hidden;}
    section[data-testid="stSidebar"] {
        background: rgba(12, 16, 28, 0.88) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.08) !important;
    }
    section[data-testid="stSidebar"] label {
        color: #8892b0 !important; font-weight: 500 !important;
        font-size: 0.82rem !important; letter-spacing: 0.03em; text-transform: uppercase;
    }
    .sidebar-title {
        font-size: 1.1rem; font-weight: 800; letter-spacing: -0.02em;
        background: linear-gradient(135deg, #38bdf8, #34d399);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 8px;
    }
    .sidebar-sub { color: #4a5270; font-size: 0.78rem; margin-bottom: 20px;
        padding-bottom: 16px; border-bottom: 1px solid rgba(255,255,255,0.04); }
    .sidebar-section { color: #5a6280; font-size: 0.7rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.12em; margin: 20px 0 8px 0; }
    .hero-title {
        font-size: 2.8rem; font-weight: 900; line-height: 1.1; letter-spacing: -0.04em;
        background: linear-gradient(135deg, #e2e8f0 0%, #f472b6 35%, #fbbf24 65%, #34d399 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 8px;
    }
    .hero-sub { color: #4a5270; font-size: 0.95rem; margin-bottom: 24px; line-height: 1.5; }
    .tag-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
    .tag-coral { background: rgba(255,107,107,0.1); color: #ff6b6b;
        border: 1px solid rgba(255,107,107,0.2); display: inline-block;
        border-radius: 100px; padding: 5px 16px; font-size: 0.72rem;
        font-weight: 600; letter-spacing: 0.04em; }
    .tag-blue { background: rgba(56,189,248,0.1); color: #38bdf8;
        border: 1px solid rgba(56,189,248,0.2); display: inline-block;
        border-radius: 100px; padding: 5px 16px; font-size: 0.72rem;
        font-weight: 600; letter-spacing: 0.04em; }
    .tag-mint { background: rgba(52,211,153,0.1); color: #34d399;
        border: 1px solid rgba(52,211,153,0.2); display: inline-block;
        border-radius: 100px; padding: 5px 16px; font-size: 0.72rem;
        font-weight: 600; letter-spacing: 0.04em; }
    .tag-amber { background: rgba(251,191,36,0.1); color: #fbbf24;
        border: 1px solid rgba(251,191,36,0.2); display: inline-block;
        border-radius: 100px; padding: 5px 16px; font-size: 0.72rem;
        font-weight: 600; letter-spacing: 0.04em; }
    .metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 24px 0; }
    .m-card { background: rgba(255,255,255,0.025); backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.05);
        border-radius: 16px; padding: 28px 24px; text-align: center;
        transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
        position: relative; overflow: hidden; }
    .m-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0;
        height: 3px; border-radius: 16px 16px 0 0; }
    .m-card:hover { transform: translateY(-3px); }
    .m-coral::before { background: linear-gradient(90deg, #ff6b6b, #fbbf24); }
    .m-coral:hover { box-shadow: 0 16px 48px rgba(255,107,107,0.12);
        border-color: rgba(255,107,107,0.15); }
    .m-blue::before { background: linear-gradient(90deg, #38bdf8, #a78bfa); }
    .m-blue:hover { box-shadow: 0 16px 48px rgba(56,189,248,0.12);
        border-color: rgba(56,189,248,0.15); }
    .m-mint::before { background: linear-gradient(90deg, #34d399, #38bdf8); }
    .m-mint:hover { box-shadow: 0 16px 48px rgba(52,211,153,0.12);
        border-color: rgba(52,211,153,0.15); }
    .m-label { color: #5a6280; font-size: 0.7rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px; }
    .m-sub { color: #3d4560; font-size: 0.76rem; margin-top: 8px; }
    .m-val-coral { font-size: 2.2rem; font-weight: 900; line-height: 1.2;
        background: linear-gradient(135deg, #ff6b6b, #fbbf24);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; }
    .m-val-blue { font-size: 1.6rem; font-weight: 800; line-height: 1.2;
        background: linear-gradient(135deg, #38bdf8, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; }
    .m-val-mint { font-size: 1.2rem; font-weight: 700; line-height: 1.3;
        background: linear-gradient(135deg, #34d399, #38bdf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; }
    .sh { font-size: 1.05rem; font-weight: 700; letter-spacing: -0.01em;
        margin: 36px 0 14px 0; padding-bottom: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        display: flex; align-items: center; gap: 10px; }
    .sh-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
    .sh-coral { color: #d4a0a0; }
    .sh-coral .sh-dot { background: #ff6b6b; box-shadow: 0 0 12px rgba(255,107,107,0.4); }
    .sh-blue { color: #9ab8d4; }
    .sh-blue .sh-dot { background: #38bdf8; box-shadow: 0 0 12px rgba(56,189,248,0.4); }
    .sh-mint { color: #8ec5b0; }
    .sh-mint .sh-dot { background: #34d399; box-shadow: 0 0 12px rgba(52,211,153,0.4); }
    .sh-amber { color: #c4ad78; }
    .sh-amber .sh-dot { background: #fbbf24; box-shadow: 0 0 12px rgba(251,191,36,0.4); }
    .glass { background: rgba(255,255,255,0.02); backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px); border: 1px solid rgba(255,255,255,0.05);
        border-radius: 16px; padding: 24px; margin: 8px 0; transition: all 0.3s ease; }
    .glass:hover { border-color: rgba(255,255,255,0.08); background: rgba(255,255,255,0.03); }
    .s-row { display: flex; justify-content: space-between; padding: 12px 4px;
        border-bottom: 1px solid rgba(255,255,255,0.025); font-size: 0.88rem; }
    .s-lbl { color: #5a6280; font-weight: 500; }
    .s-val { font-weight: 600; }
    .s-val-coral { color: #ff6b6b; }
    .s-val-blue { color: #38bdf8; }
    .s-val-mint { color: #34d399; }
    .s-val-amber { color: #fbbf24; }
    .s-val-lav { color: #a78bfa; }
    .s-val-rose { color: #f472b6; }
    .aurora-div { height: 1px; margin: 36px 0; border: none;
        background: linear-gradient(90deg, transparent, rgba(255,107,107,0.15),
        rgba(56,189,248,0.15), rgba(52,211,153,0.15), transparent); }
    .app-foot { text-align: center; padding: 48px 0 24px 0; font-size: 0.76rem;
        color: #2a2f42; letter-spacing: 0.03em; }
    .app-foot span { background: linear-gradient(90deg, #ff6b6b, #38bdf8, #34d399);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; font-weight: 700; }
    .stSelectbox > div > div { background: rgba(255,255,255,0.03) !important;
        border-color: rgba(255,255,255,0.06) !important; border-radius: 10px !important; }
    div[data-testid="stMetric"] { display: none; }
    div[data-baseweb="select"] > div { background: rgba(255,255,255,0.03) !important; }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #38bdf8, #34d399) !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load PRECOMPUTED PROFILES (no raw dataset needed)
# ---------------------------------------------------------------------------
@st.cache_data
def load_profiles():
    """Load aggregated demand profiles. No raw train.csv required."""
    for root in [
        Path(__file__).parent / "data",
        Path(__file__).parent,
        Path("data"),
        Path("."),
    ]:
        if (root / "profiles_meta.json").exists():
            meta = _json.load(open(root / "profiles_meta.json"))
            gs = pd.read_csv(root / "profiles_slot.csv", index_col=[0, 1])["demand"]
            gh = pd.read_csv(root / "profiles_hour.csv", index_col=[0, 1])["demand"]
            gm = pd.read_csv(root / "profiles_geohash.csv", index_col=0)["demand"]
            st_ = pd.read_csv(root / "profiles_stats.csv")
            return gs, gh, gm, st_, meta
    return None


profiles = load_profiles()
if profiles is None:
    st.markdown(
        '<p style="color:#ff6b6b;">Profile data not found. '
        "Place the profile CSVs in app/data/.</p>",
        unsafe_allow_html=True,
    )
    st.stop()

gh_slot, gh_hour, gh_mean, gh_stats, meta = profiles
all_gh = sorted(gh_mean.index.tolist())
all_road = meta["road_types"]
all_weather = meta["weather_types"]
global_mean = meta["global_mean"]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">Traffic Demand</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sidebar-sub">Configure location, time and context to get a '
        "demand prediction.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sidebar-section">Location and Time</div>',
        unsafe_allow_html=True,
    )
    gh = st.selectbox("Geohash", all_gh, index=0)
    hour = st.slider("Hour", 0, 23, 8)
    minute = st.selectbox("Minute", [0, 15, 30, 45], index=0)

    st.markdown(
        '<div class="sidebar-section">Road Context</div>',
        unsafe_allow_html=True,
    )
    road = st.selectbox("Road type", all_road)
    lanes = st.slider("Lanes", 1, 8, 3)
    large_v = st.selectbox("Large vehicles", ["Allowed", "Not Allowed"])
    landmarks = st.selectbox("Landmarks", ["Yes", "No"])

    st.markdown(
        '<div class="sidebar-section">Environment</div>',
        unsafe_allow_html=True,
    )
    weather = st.selectbox("Weather", all_weather)
    temp = st.slider("Temperature", -5.0, 45.0, 25.0, 0.5)

slot = (hour * 60 + minute) // 15
lat, lon = decode_geohash(gh)


# ---------------------------------------------------------------------------
# Prediction (profile lookup)
# ---------------------------------------------------------------------------
if (gh, slot) in gh_slot.index:
    pred = gh_slot[(gh, slot)]
    source = "geohash x slot"
elif (gh, hour) in gh_hour.index:
    pred = gh_hour[(gh, hour)]
    source = "geohash x hour"
elif gh in gh_mean.index:
    pred = gh_mean[gh]
    source = "geohash mean"
else:
    pred = global_mean
    source = "global mean"

pred = float(np.clip(pred, 0, 1))


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="hero-title">Traffic Demand<br>Prediction</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-sub">Real-time demand forecasting across a 15-minute geohash '
    "grid, powered by a stacked ensemble of five gradient-boosted models with "
    "spatio-temporal feature engineering.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    """<div class="tag-row">
    <span class="tag-coral">spatio-temporal</span>
    <span class="tag-blue">gradient boosting</span>
    <span class="tag-mint">geohash grid</span>
    <span class="tag-amber">15-min intervals</span>
</div>""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------------------
if pred < 0.05:
    intensity = "low"
elif pred < 0.15:
    intensity = "moderate"
elif pred < 0.4:
    intensity = "high"
else:
    intensity = "very high"

st.markdown(
    f"""
<div class="metric-grid">
    <div class="m-card m-coral">
        <div class="m-label">Predicted Demand</div>
        <div class="m-val-coral">{pred:.4f}</div>
        <div class="m-sub">intensity: {intensity}</div>
    </div>
    <div class="m-card m-blue">
        <div class="m-label">Coordinates</div>
        <div class="m-val-blue">{lat:.4f}, {lon:.4f}</div>
        <div class="m-sub">decoded from {gh}</div>
    </div>
    <div class="m-card m-mint">
        <div class="m-label">Resolution</div>
        <div class="m-val-mint">{source}</div>
        <div class="m-sub">profile lookup level</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown('<div class="aurora-div"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Demand profile chart
# ---------------------------------------------------------------------------
st.markdown(
    f'<div class="sh sh-coral"><span class="sh-dot"></span>Demand Profile '
    f"- {gh}</div>",
    unsafe_allow_html=True,
)

profile_data = (
    gh_hour.loc[gh_hour.index.get_level_values(0) == gh]
    if gh in gh_mean.index
    else pd.Series(dtype=float)
)
profile = profile_data.droplevel(0) if len(profile_data) > 0 else pd.Series(dtype=float)

if len(profile) > 0:
    chart_data = (
        pd.DataFrame({"hour": range(24)})
        .merge(
            profile.reset_index().rename(
                columns={"hour": "hour", "demand": "avg_demand"}
            ),
            on="hour",
            how="left",
        )
        .fillna(0)
    )
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.bar_chart(
        chart_data.set_index("hour")["avg_demand"],
        use_container_width=True,
        color="#38bdf8",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:#3d4560;font-size:0.78rem;text-align:center;">'
        f"Hourly average demand. Selected: {hour:02d}:{minute:02d}</p>",
        unsafe_allow_html=True,
    )

st.markdown('<div class="aurora-div"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="sh sh-blue"><span class="sh-dot"></span>Location</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=11)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown('<div class="aurora-div"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
stats_row = gh_stats[gh_stats["geohash"] == gh]
if len(stats_row) > 0:
    sr = stats_row.iloc[0]
    st.markdown(
        '<div class="sh sh-amber"><span class="sh-dot"></span>Location Statistics</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="glass">
        <div class="s-row"><span class="s-lbl">Training rows</span>
            <span class="s-val s-val-amber">{int(sr["count"]):,}</span></div>
        <div class="s-row"><span class="s-lbl">Demand range</span>
            <span class="s-val s-val-coral">{sr["demand_min"]:.4f} — {sr["demand_max"]:.4f}</span></div>
        <div class="s-row"><span class="s-lbl">Mean demand</span>
            <span class="s-val s-val-blue">{sr["demand_mean"]:.4f}</span></div>
        <div class="s-row"><span class="s-lbl">Unique timestamps</span>
            <span class="s-val s-val-mint">{int(sr["n_timestamps"])}</span></div>
        <div class="s-row"><span class="s-lbl">Road types</span>
            <span class="s-val s-val-lav">{int(sr["n_road"])}</span></div>
        <div class="s-row"><span class="s-lbl">Weather conditions</span>
            <span class="s-val s-val-rose">{int(sr["n_weather"])}</span></div>
    </div>""",
        unsafe_allow_html=True,
    )

st.markdown('<div class="aurora-div"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Context card
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="sh sh-mint"><span class="sh-dot"></span>Current Input Context</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f"""<div class="glass">
    <div class="s-row"><span class="s-lbl">Road type</span>
        <span class="s-val s-val-lav">{road}</span></div>
    <div class="s-row"><span class="s-lbl">Lanes</span>
        <span class="s-val s-val-blue">{lanes}</span></div>
    <div class="s-row"><span class="s-lbl">Weather</span>
        <span class="s-val s-val-amber">{weather}</span></div>
    <div class="s-row"><span class="s-lbl">Temperature</span>
        <span class="s-val s-val-coral">{temp:.1f} C</span></div>
    <div class="s-row"><span class="s-lbl">Large vehicles</span>
        <span class="s-val s-val-mint">{large_v}</span></div>
    <div class="s-row"><span class="s-lbl">Landmarks</span>
        <span class="s-val s-val-rose">{landmarks}</span></div>
</div>""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="app-foot">Built with <span>Traffic Demand Prediction</span>'
    " / Stacked Ensemble / Streamlit</div>",
    unsafe_allow_html=True,
)
