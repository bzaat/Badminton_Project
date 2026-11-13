"""`MatchData`: the documented per-match container used by every Milestone 1 task.

Frame axis
    One row per frame inside a rally (metadata rally intervals, inclusive), in aligned frame
    numbers (raw + `frame_shift` from the config). F rows in total.

Player axis (size 2)
    Index 0 = TOP player (far side of the main camera), index 1 = BOTTOM (near side). Who is
    top changes between games (and at 11 points in game 3); per-rally names are in
    `rallies[i]["top_player"]` / `["bottom_player"]`.

Arrays (NaN = missing for floats)
    frame          (F,)          int64    aligned frame number
    rally_idx      (F,)          int32    index into `rallies`
    shuttle_px     (F, 2)        float32  TrackNet shuttle centre, pixels
    player_box_px  (F, 2, 4)     float32  manual player box, xyxy pixels
    pose_kp_px     (F, 2, 17, 3) float32  COCO-17 keypoints: x px, y px, confidence
    pose_box_px    (F, 2, 4)     float32  box the pose was estimated in (xyxy px)
    pose_assign    (F, 2)        int8     0 none, 1 matched to manual box by IoU, 2 geometric fallback
    foot_px        (F, 2, 2)     float32  ground contact point in pixels (see foot_src)
    foot_src       (F, 2)        int8     -1 none, 0 ankle keypoints (+ ankle_ground_offset),
                                          1 bottom-centre of the player box (+ box_ground_offset)
    foot_court     (F, 2, 2)     float32  foot_px projected to court metres (static calibration)
    court_flag     (F,)          int8     camera state from court_perframe: 0 valid, 1 refit,
                                          2 no_court, -1 unknown (no sample within the max gap)

Tables
    rallies : list of dicts {game, index, start, end, n_hits, in_timeline, starts_with_serve, winner, winner_from,
                             score_after, top_player, bottom_player, players_from,
                             main_camera_frames, main_camera_frac, wrong_half_frac, suspect}
              starts_with_serve = first annotated hit is serve/flick_serve. When False (139 rallies)
              the evidence says the rally interval starts after the serve (see REPORT.md).
              main_camera_* counts frames with court_flag == 0 and both feet known.
              `suspect` = on main-camera frames, a player's feet are in the wrong half for more
              than court.max_wrong_half_frac of the rally (exclude from court-space features).
    hits    : list of dicts {frame, shot_type, side, player, game, rally, rally_idx}
    info    : dict {match, player1, player2, game1_top, match_winner, frame_shift, corners_are,
                    schema_version}
    court   : static calibration (corners_px (4, 2), net_px) -> `calibration()`
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .court import CourtCalibration, CourtModel

SCHEMA_VERSION = 4  # bump whenever build output changes
N_KP = 17
TOP, BOTTOM = 0, 1
SIDE_INDEX = {"top": TOP, "bottom": BOTTOM}

COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
COCO_SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
    (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)

# name -> (dtype, shape without the frame axis)
ARRAY_SPEC: dict[str, tuple[type, tuple[int, ...]]] = {
    "frame": (np.int64, ()),
    "rally_idx": (np.int32, ()),
    "shuttle_px": (np.float32, (2,)),
    "player_box_px": (np.float32, (2, 4)),
    "pose_kp_px": (np.float32, (2, N_KP, 3)),
    "pose_box_px": (np.float32, (2, 4)),
    "pose_assign": (np.int8, (2,)),
    "foot_px": (np.float32, (2, 2)),
    "foot_src": (np.int8, (2,)),
    "foot_court": (np.float32, (2, 2)),
    "court_flag": (np.int8, ()),
}


class SchemaError(ValueError):
    pass


@dataclass
class MatchData:
    info: dict
    rallies: list[dict]
    hits: list[dict]
    court: dict
    arrays: dict[str, np.ndarray] = field(repr=False)

    # ---- convenience ------------------------------------------------------------------
    def __getattr__(self, name: str) -> np.ndarray:
        arrays = self.__dict__.get("arrays", {})
        if name in arrays:
            return arrays[name]
        raise AttributeError(name)

    @property
    def name(self) -> str:
        return self.info["match"]

    @property
    def n_frames(self) -> int:
        return int(self.arrays["frame"].shape[0])

    def calibration(self, model: CourtModel | None = None) -> CourtCalibration:
        return CourtCalibration(
            np.asarray(self.court["corners_px"]),
            {k: np.asarray(v) for k, v in self.court["net_px"].items()},
            model or CourtModel(),
            self.info.get("corners_are", "doubles_outer"),
        )

    def rally_slice(self, rally_idx: int) -> slice:
        """Rows of one rally (rows are sorted by frame, rallies do not overlap)."""
        r = self.arrays["rally_idx"]
        lo = int(np.searchsorted(self.arrays["frame"], self.rallies[rally_idx]["start"]))
        hi = int(np.searchsorted(self.arrays["frame"], self.rallies[rally_idx]["end"], side="right"))
        assert (r[lo:hi] == rally_idx).all()
        return slice(lo, hi)

    def find_rally(self, game: int, index: int) -> int:
        for i, r in enumerate(self.rallies):
            if r["game"] == game and r["index"] == index:
                return i
        raise KeyError(f"no rally game={game} index={index} in {self.name}")

    def rows_for_frames(self, frames: np.ndarray) -> np.ndarray:
        """Row index for each frame (-1 if the frame is not inside a rally)."""
        fr = self.arrays["frame"]
        idx = np.clip(np.searchsorted(fr, frames), 0, max(len(fr) - 1, 0))
        ok = (len(fr) > 0) & (fr[idx] == frames)
        return np.where(ok, idx, -1)

    # ---- validation -------------------------------------------------------------------
    def validate(self) -> None:
        """Raise SchemaError if any array or table violates the documented schema."""
        a = self.arrays
        missing = set(ARRAY_SPEC) - set(a)
        if missing:
            raise SchemaError(f"missing arrays: {sorted(missing)}")
        F = a["frame"].shape[0]
        for name, (dtype, shape) in ARRAY_SPEC.items():
            arr = a[name]
            if arr.dtype != dtype:
                raise SchemaError(f"{name}: dtype {arr.dtype}, expected {np.dtype(dtype)}")
            if arr.shape != (F, *shape):
                raise SchemaError(f"{name}: shape {arr.shape}, expected {(F, *shape)}")
        if F and np.any(np.diff(a["frame"]) <= 0):
            raise SchemaError("frame must be strictly increasing")
        if F and (a["rally_idx"].min() < 0 or a["rally_idx"].max() >= len(self.rallies)):
            raise SchemaError("rally_idx out of range")
        for i, r in enumerate(self.rallies):
            if r["end"] < r["start"]:
                raise SchemaError(f"rally {i}: end < start")
        if not set(np.unique(a["pose_assign"])) <= {0, 1, 2}:
            raise SchemaError("pose_assign has unknown codes")
        if not set(np.unique(a["foot_src"])) <= {-1, 0, 1}:
            raise SchemaError("foot_src has unknown codes")
        if not set(np.unique(a["court_flag"])) <= {-1, 0, 1, 2}:
            raise SchemaError("court_flag has unknown codes")
        # pose and foot consistency
        no_pose = a["pose_assign"] == 0
        if np.any(~np.isnan(a["pose_kp_px"][no_pose][..., 0])):
            raise SchemaError("keypoints present where pose_assign == 0")
        if np.any(np.isnan(a["foot_px"][a["foot_src"] >= 0])):
            raise SchemaError("foot_px NaN where foot_src says present")
        for h in self.hits:
            if h["side"] not in SIDE_INDEX:
                raise SchemaError(f"hit with unknown side {h['side']!r}")
            r = self.rallies[h["rally_idx"]]
            if not (r["start"] <= h["frame"] <= r["end"]):
                raise SchemaError(f"hit at frame {h['frame']} outside its rally {r['start']}-{r['end']}")
        if [h["frame"] for h in self.hits] != sorted(h["frame"] for h in self.hits):
            raise SchemaError("hits must be sorted by frame")
        corners = np.asarray(self.court["corners_px"])
        if corners.shape != (4, 2):
            raise SchemaError("court corners must be (4, 2)")
