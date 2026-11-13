"""Rally visualisation: image-space overlay, top-down court, and a signal timeline.

No video is needed: the image panel is drawn on a blank 1280x720 canvas. If a frame image is
passed as `background` (e.g. read from the broadcast video), it is drawn underneath.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .court import CourtModel  # noqa: E402
from .schema import BOTTOM, COCO_SKELETON, SIDE_INDEX, TOP, MatchData  # noqa: E402

SIDE_COLOR = {TOP: "#e8710a", BOTTOM: "#c5221f"}
LEFT_WRIST, RIGHT_WRIST = 9, 10


def _draw_court_image(ax, md: MatchData, model: CourtModel) -> None:
    calib = md.calibration(model)
    for name, seg in model.lines():
        if name == "net":
            continue
        p = calib.to_image(seg)
        ax.plot(p[:, 0], p[:, 1], color="#2e7d32", lw=1.0, alpha=0.8)
    net = md.court["net_px"]
    for a, b in (("left_base", "right_base"), ("left_top", "right_top"), ("left_base", "left_top"), ("right_base", "right_top")):
        if a in net and b in net:
            ax.plot([net[a][0], net[b][0]], [net[a][1], net[b][1]], color="#f9ab00", lw=1.2)


def _draw_skeleton(ax, kp: np.ndarray, color: str, min_score: float, lw: float = 1.2, alpha: float = 0.9) -> None:
    for i, j in COCO_SKELETON:
        if kp[i, 2] >= min_score and kp[j, 2] >= min_score:
            ax.plot([kp[i, 0], kp[j, 0]], [kp[i, 1], kp[j, 1]], color=color, lw=lw, alpha=alpha)


def _draw_court_topdown(ax, model: CourtModel) -> None:
    for name, seg in model.lines():
        ax.plot(seg[:, 0], seg[:, 1], color="#f9ab00" if name == "net" else "#2e7d32",
                lw=2.0 if name == "net" else 1.0)
    ax.set_xlim(-1.5, model.width_doubles + 1.5)
    ax.set_ylim(model.length + 1.5, -1.5)  # far side (top player) at the top
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m), far baseline = 0")


def wrist_speed(md: MatchData, rows: slice, side: int, min_score: float) -> np.ndarray:
    """Per-frame speed (px/frame) of the faster wrist of one player; NaN where unknown."""
    kp = md.pose_kp_px[rows, side][:, [LEFT_WRIST, RIGHT_WRIST]].astype(np.float64)
    xy = np.where(kp[..., 2:3] >= min_score, kp[..., :2], np.nan)
    v = np.linalg.norm(np.diff(xy, axis=0), axis=-1)  # (n-1, 2)
    v = np.where(np.isnan(v), -np.inf, v).max(axis=1)
    v = np.where(np.isinf(v), np.nan, v)
    return np.r_[np.nan, v]


def read_frames(video_path: str | Path, frames: list[int]) -> dict[int, np.ndarray]:
    """Decode the requested 0-based frames (BGR) from a video, seeking once per frame."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    out = {}
    for f in sorted(set(int(x) for x in frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if ok:
            out[f] = img
    cap.release()
    return out


def draw_frame_overlay(img: np.ndarray, md: MatchData, frame: int, model: CourtModel | None = None,
                       min_score: float = 0.3) -> np.ndarray:
    """Draw the annotations of one frame on a video image (BGR copy): court lines, player
    boxes (thin), pose skeletons, shuttle (circle), ground points (filled dots)."""
    import cv2
    img = img.copy()
    model = model or CourtModel()
    calib = md.calibration(model)
    for name, seg in model.lines():
        p = calib.to_image(seg).round().astype(int)
        cv2.line(img, tuple(p[0]), tuple(p[1]), (0, 200, 0) if name != "net" else (0, 200, 255), 1, cv2.LINE_AA)
    row = int(md.rows_for_frames(np.array([frame]))[0])
    if row < 0:
        cv2.putText(img, f"frame {frame}: not in a rally", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img
    colors = {TOP: (10, 113, 232), BOTTOM: (31, 34, 197)}  # BGR versions of SIDE_COLOR
    for s in (TOP, BOTTOM):
        b = md.player_box_px[row, s]
        if not np.isnan(b).any():
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), colors[s], 1)
        if md.pose_assign[row, s] > 0:
            kp = md.pose_kp_px[row, s]
            for i, j in COCO_SKELETON:
                if kp[i, 2] >= min_score and kp[j, 2] >= min_score:
                    cv2.line(img, (int(kp[i, 0]), int(kp[i, 1])), (int(kp[j, 0]), int(kp[j, 1])), colors[s], 2, cv2.LINE_AA)
        fp = md.foot_px[row, s]
        if not np.isnan(fp).any():
            cv2.circle(img, (int(fp[0]), int(fp[1])), 4, (255, 255, 0), -1)
    sh = md.shuttle_px[row]
    if not np.isnan(sh).any():
        cv2.circle(img, (int(round(sh[0])), int(round(sh[1]))), 9, (0, 255, 255), 2, cv2.LINE_AA)
    flag = {0: "main camera", 1: "refit", 2: "no court", -1: "unknown"}[int(md.court_flag[row])]
    r = md.rallies[int(md.rally_idx[row])]
    cv2.putText(img, f"frame {frame} | g{r['game']} r{r['index']} | {flag}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def plot_rally(md: MatchData, rally_idx: int, out: str | Path, model: CourtModel | None = None,
               background: np.ndarray | None = None, min_score: float = 0.3) -> str:
    """Render one rally to `out` (PNG). Returns the path written."""
    model = model or CourtModel()
    r = md.rallies[rally_idx]
    sl = md.rally_slice(rally_idx)
    fr = md.frame[sl]
    t = (fr - fr[0]) / 30.0
    hits = [h for h in md.hits if h["rally_idx"] == rally_idx]
    hit_rows = md.rows_for_frames(np.array([h["frame"] for h in hits], dtype=np.int64))

    fig = plt.figure(figsize=(17, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], height_ratios=[3, 1.25])
    ax_img, ax_top, ax_ts = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[:, 1]), fig.add_subplot(gs[1, 0])

    # ---- image space
    W, H = md.info["size"]
    if background is not None:
        ax_img.imshow(background[..., ::-1] if background.ndim == 3 else background, extent=(0, W, H, 0))
    else:
        ax_img.add_patch(plt.Rectangle((0, 0), W, H, color="#f1f3f4", zorder=-1))
    _draw_court_image(ax_img, md, model)
    sh = md.shuttle_px[sl].astype(np.float64)
    ok = ~np.isnan(sh[:, 0])
    sc = ax_img.scatter(sh[ok, 0], sh[ok, 1], c=t[ok], cmap="viridis", s=6, zorder=3)
    for k, (h, row) in enumerate(zip(hits, hit_rows)):
        s = SIDE_INDEX[h["side"]]
        if row >= 0 and md.pose_assign[row, s] > 0:
            _draw_skeleton(ax_img, md.pose_kp_px[row, s].astype(np.float64), SIDE_COLOR[s], min_score)
        if row < 0:
            continue
        # shuttle position at the hit (may be a TrackNet outlier, so it is not used as the label anchor)
        if not np.isnan(md.shuttle_px[row, 0]):
            ax_img.scatter(*md.shuttle_px[row], marker="*", s=110, color=SIDE_COLOR[s], edgecolor="k", zorder=4)
        # label above the hitter's box (pose box, else manual box)
        box = md.pose_box_px[row, s] if md.pose_assign[row, s] > 0 else md.player_box_px[row, s]
        if not np.isnan(box).any():
            anchor = ((box[0] + box[2]) / 2, box[1])
            ax_img.annotate(f"{k + 1}:{h['shot_type']}", anchor, xytext=(0, 4), textcoords="offset points",
                            fontsize=8, ha="center", color=SIDE_COLOR[s], zorder=5,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    ax_img.set_xlim(0, W); ax_img.set_ylim(H, 0); ax_img.set_aspect("equal")
    ax_img.set_title("image space (px): court, shuttle (colour = time), hitter pose + label at each hit, "
                     "star = shuttle at the hit frame", fontsize=9)
    fig.colorbar(sc, ax=ax_img, fraction=0.025, label="time in rally (s)")

    # ---- top-down court
    _draw_court_topdown(ax_top, model)
    cam_ok = md.court_flag[sl] == 0
    for s, name in ((TOP, r.get("top_player")), (BOTTOM, r.get("bottom_player"))):
        fc = md.foot_court[sl, s].astype(np.float64)
        good = cam_ok & (md.foot_src[sl, s] >= 0)
        ax_top.plot(np.where(good, fc[:, 0], np.nan), np.where(good, fc[:, 1], np.nan),
                    color=SIDE_COLOR[s], lw=1.0, alpha=0.7, label=f"{'top' if s == TOP else 'bottom'}: {name}")
    for k, (h, row) in enumerate(zip(hits, hit_rows)):
        s = SIDE_INDEX[h["side"]]
        if row >= 0 and md.foot_src[row, s] >= 0:
            p = md.foot_court[row, s]
            ax_top.scatter(*p, s=60, color=SIDE_COLOR[s], edgecolor="k", zorder=4)
            ax_top.annotate(str(k + 1), p, xytext=(4, 2), textcoords="offset points", fontsize=8)
    ax_top.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), fontsize=8)
    ax_top.set_title("top-down court (m): feet, hit positions (numbers)\nmain-camera frames only")

    # ---- timeline
    ax_ts.plot(t, sh[:, 0], lw=1, label="shuttle x (px)", color="#1a73e8")
    ax_ts.plot(t, sh[:, 1], lw=1, label="shuttle y (px)", color="#9334e6")
    ax_ts.set_ylabel("px"); ax_ts.set_xlabel("time in rally (s)")
    ax2 = ax_ts.twinx()
    for s in (TOP, BOTTOM):
        ax2.plot(t, wrist_speed(md, sl, s, min_score), lw=0.8, alpha=0.6, color=SIDE_COLOR[s],
                 label=f"{'top' if s == TOP else 'bottom'} wrist speed")
    ax2.set_ylabel("wrist speed (px/frame)")
    for k, h in enumerate(hits):
        th = (h["frame"] - fr[0]) / 30.0
        ax_ts.axvline(th, color=SIDE_COLOR[SIDE_INDEX[h["side"]]], ls="--", lw=0.8)
        ax_ts.text(th, ax_ts.get_ylim()[1], str(k + 1), fontsize=7, va="bottom", ha="center")
    h1, l1 = ax_ts.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax_ts.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right", ncol=4)
    ax_ts.set_title("signals (dashed lines = annotated hits)")

    flags = []
    if r.get("suspect"):
        flags.append(f"SUSPECT: players in wrong half on {r['wrong_half_frac']:.0%} of main-camera frames")
    cov = (md.court_flag[sl] == 0).mean()
    fig.suptitle(
        f"{md.name}\ngame {r['game']} rally {r['index']} | frames {r['start']}-{r['end']} ({len(fr) / 30:.1f} s) | "
        f"{len(hits)} hits | winner: {r.get('winner')} | score after: {r.get('score_after')} | "
        f"main-camera frames: {cov:.0%}" + (" | " + "; ".join(flags) if flags else ""), fontsize=10)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return str(out)
