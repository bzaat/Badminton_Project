"""Badminton analytics demo (Streamlit).

    uv run --group demo streamlit run demo/app.py

Needs the BFMD package (configs/default.yaml or BFMD_ROOT) and the Milestone 1 outputs
(`python -m badminton_analysis all`). The match video is optional: with it, the rally replay shows
real frames; without it, the same annotations are drawn on a blank court.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from badminton_analysis.build import load_match, singles_matches  # noqa: E402
from badminton_analysis.config import load_config  # noqa: E402
from badminton_analysis.court import CourtModel  # noqa: E402
from badminton_analysis.demo_data import (load_outputs, rally_hits, rally_label,  # noqa: E402
                                          rally_tracks, shuttle_speed)
from badminton_analysis.viz import draw_frame_overlay, read_frames  # noqa: E402

TOP_C, BOTTOM_C, HIT_C, DET_C, FP_C = "#e8710a", "#c5221f", "#1a7f37", "#1f6feb", "#8250df"
VIDEO_DIRS = [Path("/mnt/d/badminton/videos"), Path("D:/badminton/videos")]

st.set_page_config(page_title="Badminton Analytics", page_icon="🏸", layout="wide")


@st.cache_resource
def _cfg():
    return load_config()


@st.cache_resource(show_spinner="Loading match…")
def _match(name: str):
    return load_match(_cfg(), name)


@st.cache_data
def _outputs():
    return load_outputs(_cfg().output_dir)


@st.cache_data
def _matches() -> list[str]:
    return singles_matches(_cfg())


def _short(m: str) -> str:
    """'VICTOR-China-Open-2025-Shi-Yu-Qi-CHN-3-vs.-Wang-Zheng-Xing-CHN-F' -> 'China Open 2025 F: Shi Yu Qi vs Wang Zheng Xing'."""
    parts = m.split("-")
    try:
        y = next(i for i, p in enumerate(parts) if p.isdigit() and len(p) == 4)
        event = " ".join(parts[1:y]).replace("Open", "Open")
        a, b = "-".join(parts[y + 1:]).split("-vs.-")
        clean = lambda s: " ".join(w for w in s.split("-") if not (w.isupper() and len(w) == 3) and not w.isdigit())
        rnd = b.split("-")[-1]
        return f"{event} {parts[y]} {rnd}: {clean(a)} vs {clean(b.rsplit('-', 1)[0])}"
    except (StopIteration, ValueError):
        return m


def _video_for(match: str) -> Path | None:
    cfg = _cfg()
    dirs = ([cfg.video_root] if cfg.video_root else []) + VIDEO_DIRS
    for d in dirs:
        p = Path(d) / f"{match}.mp4"
        if p.exists():
            return p
    return None


@st.cache_data(max_entries=64, show_spinner=False)
def _frame_image(match: str, frame: int, use_video: bool) -> np.ndarray:
    md = _match(match)
    model = CourtModel.from_config(_cfg()["court"])
    img = None
    if use_video:
        v = _video_for(match)
        if v is not None:
            img = read_frames(v, [frame]).get(frame)
    if img is None:
        img = np.full((_cfg().height, _cfg().width, 3), (40, 70, 45), np.uint8)   # blank court-green canvas
    return draw_frame_overlay(img, md, frame, model)[:, :, ::-1]                # BGR -> RGB


def _court_figure(model: CourtModel, tr: dict, upto: int, hits: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for name, seg in model.lines():
        fig.add_trace(go.Scatter(x=seg[:, 0], y=seg[:, 1], mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color="#f9ab00" if name == "net" else "#9aa5a0", width=3 if name == "net" else 1)))
    k = int(np.searchsorted(tr["frame"], upto, side="right"))
    for side, col, label in (("top", TOP_C, "far player"), ("bottom", BOTTOM_C, "near player")):
        p = tr[f"{side}_m"][:k]
        fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="lines", name=f"{label} path",
                                 line=dict(color=col, width=2), opacity=0.55, hoverinfo="skip"))
        last = p[~np.isnan(p).any(axis=1)][-1:] if len(p) else p
        if len(last):
            fig.add_trace(go.Scatter(x=last[:, 0], y=last[:, 1], mode="markers", name=label,
                                     marker=dict(color=col, size=14, line=dict(color="white", width=2))))
    done = hits[hits["frame"] <= upto]
    if len(done):
        idx = np.searchsorted(tr["frame"], done["frame"].to_numpy())
        idx = np.clip(idx, 0, len(tr["frame"]) - 1)
        pos = np.array([tr[f"{s}_m"][i] for s, i in zip(done["side"], idx)])
        fig.add_trace(go.Scatter(x=pos[:, 0], y=pos[:, 1], mode="markers+text", name="hits",
                                 text=done["shot (annotated)"], textposition="top center", textfont=dict(size=10),
                                 marker=dict(color=HIT_C, size=8, symbol="diamond")))
    fig.update_yaxes(range=[model.length + 1.2, -1.2], scaleanchor="x", scaleratio=1, title="metres (far baseline = 0)")
    fig.update_xaxes(range=[-1.2, model.width_doubles + 1.2], title="metres")
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.08))
    return fig


def _timeline(tr: dict, hits: pd.DataFrame, fps: list[int], frame: int) -> go.Figure:
    v = shuttle_speed(tr["shuttle_px"])
    fig = go.Figure(go.Scatter(x=tr["frame"], y=v, mode="lines", name="shuttle speed (px/frame)",
                               line=dict(color="#6e7781", width=1.5)))
    ymax = float(np.nanmax(v)) if np.isfinite(v).any() else 1.0
    for f, s in zip(hits["frame"], hits["shot (annotated)"]):
        fig.add_vline(x=f, line=dict(color=HIT_C, width=1.5))
    fig.add_trace(go.Scatter(x=hits["frame"], y=[ymax * 1.08] * len(hits), mode="markers", name="annotated hit",
                             marker=dict(color=HIT_C, symbol="diamond", size=9), text=hits["shot (annotated)"],
                             hovertemplate="frame %{x}<br>%{text}<extra>annotated</extra>"))
    det = hits.dropna(subset=["detection frame"])
    fig.add_trace(go.Scatter(x=det["detection frame"], y=[ymax * 0.98] * len(det), mode="markers", name="detected (Task 2)",
                             marker=dict(color=DET_C, symbol="triangle-down", size=10)))
    if fps:
        fig.add_trace(go.Scatter(x=fps, y=[ymax * 0.98] * len(fps), mode="markers", name="false detection",
                                 marker=dict(color=FP_C, symbol="x", size=10)))
    fig.add_vline(x=frame, line=dict(color="#24292f", width=2, dash="dot"))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="video frame (30 fps)",
                      yaxis_title="px / frame", legend=dict(orientation="h", y=1.15))
    return fig


# ------------------------------------------------------------------------------------------- #
st.title("🏸 Badminton broadcast analytics")
st.caption("Milestone 1 on BFMD: shot timing, shot types and player movement, evaluated leave-one-match-out. "
           "Code: [bzaat/Badminton_Project](https://github.com/bzaat/Badminton_Project)")

try:
    matches = _matches()
except Exception as e:  # data missing
    st.error(f"Could not find the BFMD package: {e}. Set BFMD_ROOT or edit configs/default.yaml.")
    st.stop()
outs = _outputs()

with st.sidebar:
    st.header("Choose a rally")
    order = sorted(matches, key=lambda m: (_video_for(m) is None, _short(m)))
    match = st.selectbox("Match", order, format_func=lambda m: ("🎥 " if _video_for(m) else "") + _short(m))
    md = _match(match)
    games = sorted({r["game"] for r in md.rallies})
    game = st.radio("Game", games, horizontal=True)
    ridx = [i for i, r in enumerate(md.rallies) if r["game"] == game and r["n_hits"] > 0]
    rally_i = st.selectbox("Rally", ridx, format_func=lambda i: rally_label(md.rallies[i]))
    has_video = _video_for(match) is not None
    use_video = st.toggle("Show broadcast video", value=has_video, disabled=not has_video,
                          help="Only matches marked 🎥 have a video on this machine.")
    st.divider()
    r = md.rallies[rally_i]
    st.markdown(f"**{r['top_player']}** (far) vs **{r['bottom_player']}** (near)  \n"
                f"Score after rally: {r.get('score_after') or '—'}")

tab_rally, tab_match, tab_models, tab_about = st.tabs(["Rally replay", "Match movement", "Model results", "About"])

with tab_rally:
    hits, fps = rally_hits(md, rally_i, outs["t2_pred"], outs["t3_pred"])
    tr = rally_tracks(md, rally_i)
    lo, hi = int(tr["frame"][0]), int(tr["frame"][-1])
    key = f"frame_{match}_{rally_i}"
    if key not in st.session_state:
        st.session_state[key] = int(hits["frame"].iloc[0]) if len(hits) else lo
    jump = st.columns([3, 1])
    with jump[1]:
        pick = st.selectbox("Jump to hit", [None] + list(hits.index),
                            format_func=lambda i: "—" if i is None else
                            f"#{hits.loc[i, '#']} {hits.loc[i, 'shot (annotated)']} (frame {hits.loc[i, 'frame']})")
        if pick is not None and st.session_state.get(f"{key}_last_pick") != pick:
            st.session_state[key] = int(hits.loc[pick, "frame"])
            st.session_state[f"{key}_last_pick"] = pick
    with jump[0]:
        frame = st.slider("Frame", lo, hi, key=key)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Hits in rally", len(hits))
    m2.metric("Detected within ±2 frames", f"{int(hits['detected'].sum())} / {len(hits)}" if len(hits) else "—")
    m3.metric("False detections", len(fps))
    known = hits["shot correct"].dropna()
    m4.metric("Shot type correct", f"{int(known.sum())} / {len(known)}" if len(known) else "—")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.image(_frame_image(match, frame, use_video), width="stretch",
                 caption="Annotations drawn on the frame: court lines, player boxes, pose, ground points (cyan), shuttle (yellow ring).")
    with c2:
        st.plotly_chart(_court_figure(CourtModel.from_config(_cfg()["court"]), tr, frame, hits),
                        width="stretch")
    st.plotly_chart(_timeline(tr, hits, fps, frame), width="stretch")
    st.dataframe(hits.drop(columns=["shot correct"]), hide_index=True, width="stretch")

with tab_match:
    s, rr = outs["t4_summary"], outs["t4_rally"]
    if s is None or rr is None:
        st.info("Run `python -m badminton_analysis task4` to create the movement tables.")
    else:
        ms = s[s["match"] == match]
        cols = st.columns(max(1, len(ms)))
        for col, (_, row) in zip(cols, ms.iterrows()):
            col.subheader(row["player"])
            col.metric("Distance", f"{row['distance_m'] / 1000:.2f} km")
            col.metric("Per minute of rally play", f"{row['distance_per_min_m']:.0f} m")
            col.metric("Median rally mean speed", f"{row['median_rally_mean_speed_mps']:.2f} m/s")
        mr = rr[rr["match"] == match].copy()
        mr["rally label"] = "G" + mr["game"].astype(str) + " R" + (mr["rally"] + 1).astype(str)
        fig = go.Figure()
        for p, g in mr.groupby("player"):
            fig.add_trace(go.Scatter(x=np.arange(len(g)), y=g["distance_m"], mode="lines+markers", name=p,
                                     text=g["rally label"], hovertemplate="%{text}: %{y:.1f} m<extra>%{fullData.name}</extra>"))
        fig.update_layout(height=320, xaxis_title="rally (in match order)", yaxis_title="distance per rally (m)",
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")
        zones = [c for c in ms.columns if c.startswith("zone_")]
        z = ms.set_index("player")[zones].T
        z.index = [c.replace("zone_", "").replace("_", " ") for c in zones]
        st.markdown("**Share of time per court zone** (each player's own half, front / mid / rear × left / centre / right)")
        st.bar_chart(z)
        st.caption("Measured from ankle keypoints on main-camera frames, 9-frame smoothing. Noise floor 0.08–0.10 m/s; "
                   "absolute distances are uncertain by about ±10–20%. The far side reads slightly faster (pixel density).")

with tab_models:
    t2, t3 = outs["t2_metrics"], outs["t3_metrics"]
    if t2 is None or t3 is None:
        st.info("Run `python -m badminton_analysis task2` and `task3` to create the result files.")
    else:
        a, b = st.columns(2)
        micro = t2["variants"]["logistic"]["micro"]["2"]
        a.metric("Hit detection F1 @ ±2 (all 12 held-out matches)", f"{micro['f1']:.3f}",
                 help=f"precision {micro['precision']:.3f}, recall {micro['recall']:.3f}")
        pm = t2["variants"]["logistic"]["per_match"].get(match, {}).get("tol2")
        if pm:
            a.metric("This match, held out", f"{pm['f1']:.3f}", help=f"tp {pm['tp']}, fp {pm['fp']}, fn {pm['fn']}")
        fine = t3["schemes"]["fine"]
        hgb = fine["models"]["hgb"]
        ci = t3["bootstrap_fine"]["hgb"]["macro_f1"]
        b.metric("Shot type macro F1, 12 classes", f"{hgb['macro_f1']:.3f}", help=f"95% CI {ci[0]:.3f}–{ci[1]:.3f}")
        b.metric("Shot type accuracy", f"{hgb['accuracy']:.3f}")
        cm = np.array(t3["confusion_fine"], float)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        heat = go.Figure(go.Heatmap(z=cmn, x=fine["classes"], y=fine["classes"], colorscale="Greens", zmin=0, zmax=1,
                                    text=cm.astype(int), hovertemplate="annotated %{y}<br>predicted %{x}<br>%{text} hits (%{z:.0%})<extra></extra>"))
        heat.update_yaxes(autorange="reversed", title="annotated")
        heat.update_xaxes(title="predicted")
        heat.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), title="Shot-type confusion (row-normalised, all held-out hits)")
        st.plotly_chart(heat, width="stretch")

with tab_about:
    st.markdown("""
**What you are looking at.** Each rally comes from the BFMD dataset (12 BWF Super 1000 men's singles
matches, 2025). Player positions are in court metres via a homography; hits are the dataset's
annotations; *detected* hits and *predicted* shot types come from models that never saw this match
(leave-one-match-out).

**How to read the timeline.** Hits show up as spikes in shuttle speed. Green diamonds are annotated
hits, blue triangles are Task 2 detections matched to a hit within ±2 frames, purple crosses are
detections that matched no hit.

**Limits.** 12 matches, 9 players, one broadcast style. Inputs are annotations (TrackNet shuttle,
BFMD pose), so their errors cap performance. Video is only available here for one match.
All numbers: `REPORT.md`.
""")
