"""
BFMD (Full-Match Badminton Dataset) -> object-detection dataset.

Detection classes (0 = background, as torchvision detection models expect):
    1 top_player     from annotations/player_bbox  (Label Studio, percent coords)
    2 bottom_player  from annotations/player_bbox
    3 shuttle        from annotations/shuttle      (TrackNet predictions, percent coords)

Annotations only exist inside rallies, so every sample is one rally frame of one
singles match. Images come either from
  * pre-extracted JPEGs  <frames_root>/<match_name>/<frame:06d>.jpg   (fast, recommended), or
  * the match videos     <video_root>/**/<match_name>.mp4             (random seek, slower).

Usage:
    # one-off: extract the annotated frames you need (every 5th frame here)
    python badminton_dataset.py extract --ann-root <BFMD_data>/annotations \
        --video-root <BFMD_data>/videos --out dataset/bfmd_frames --stride 5

    # dataset statistics / sanity check
    python badminton_dataset.py stats --ann-root <BFMD_data>/annotations

    # in code
    train_set = BadmintonDetection(ann_root, frames_root="dataset/bfmd_frames",
                                   split="train", stride=5, transform=train_transform)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors

CLASSES = ["__background__", "top_player", "bottom_player", "shuttle"]
_PLAYER_LABELS = {"Top_Player": 1, "Bottom_Player": 2}
SHUTTLE = 3
IMG_W, IMG_H = 1280, 720  # all BFMD videos are 1280x720 @ 30 fps

# Default location of the data; works from Windows Python and from WSL.
DEFAULT_ROOT = (
    "D:/badminton/BFMD_data-20261026T161257Z-1-001/BFMD_data"
    if os.name == "nt"
    else "/mnt/d/badminton/BFMD_data-20261026T161257Z-1-001/BFMD_data"
)
DEFAULT_ANN_ROOT = os.path.join(DEFAULT_ROOT, "annotations")

# Hold out whole matches (never frames of the same match) to avoid leakage.
DEFAULT_VAL_MATCHES = [
    "VICTOR-China-Open-2025-Chou-Tien-Chen-TPE-6-vs.-Shi-Yu-Qi-CHN-3-SF",
    "YONEX-All-England-Open-2025-Lee-Chia-Hao-TPE-vs.-Alex-Lanier-FRA-SF",
]


# --------------------------------------------------------------------------- #
# Annotation parsing
# --------------------------------------------------------------------------- #
def list_matches(ann_root: str) -> list[str]:
    """Singles matches that have both player boxes and shuttle tracks."""
    players = {Path(p).stem for p in glob.glob(os.path.join(ann_root, "player_bbox", "*.json"))}
    shuttles = {Path(p).stem for p in glob.glob(os.path.join(ann_root, "shuttle", "*.json"))}
    return sorted(players & shuttles)


def _pct_to_xyxy(x, y, w, h):
    """Label Studio percent (top-left x/y, width/height) -> pixel xyxy."""
    x1 = x * IMG_W / 100.0
    y1 = y * IMG_H / 100.0
    return [x1, y1, x1 + w * IMG_W / 100.0, y1 + h * IMG_H / 100.0]


def parse_match(ann_root: str, match: str) -> dict:
    """
    Returns {"frames": int64 (N,), "boxes": float32 (N, 3, 4)} where boxes[i, c-1]
    is the pixel xyxy box of class c in frames[i], or NaN when that object is absent.
    """
    per_frame: dict[int, np.ndarray] = {}

    def slot(frame):
        if frame not in per_frame:
            per_frame[frame] = np.full((3, 4), np.nan, dtype=np.float32)
        return per_frame[frame]

    with open(os.path.join(ann_root, "player_bbox", f"{match}.json"), encoding="utf-8") as f:
        pb = json.load(f)
    for track in pb["annotations"][0]["result"]:
        label = _PLAYER_LABELS.get(track["value"]["labels"][0])
        if label is None:
            continue
        for s in track["value"]["sequence"]:
            if s.get("enabled", True):
                slot(int(s["frame"]))[label - 1] = _pct_to_xyxy(s["x"], s["y"], s["width"], s["height"])

    with open(os.path.join(ann_root, "shuttle", f"{match}.json"), encoding="utf-8") as f:
        sh = json.load(f)
    for track in sh["predictions"][0]["result"]:
        for s in track["value"]["sequence"]:
            if s.get("enabled", True):
                slot(int(s["frame"]))[SHUTTLE - 1] = _pct_to_xyxy(s["x"], s["y"], s["width"], s["height"])

    frames = np.array(sorted(per_frame), dtype=np.int64)
    boxes = np.stack([per_frame[f] for f in frames]) if len(frames) else np.zeros((0, 3, 4), np.float32)
    return {"frames": frames, "boxes": boxes}


def load_match(ann_root: str, match: str, cache_dir: str | None) -> dict:
    """parse_match with an .npz cache (player_bbox JSONs are ~10 MB each, slow to parse)."""
    if cache_dir is None:
        return parse_match(ann_root, match)
    cache = os.path.join(cache_dir, f"{match}.npz")
    srcs = [os.path.join(ann_root, d, f"{match}.json") for d in ("player_bbox", "shuttle")]
    if os.path.exists(cache) and os.path.getmtime(cache) >= max(os.path.getmtime(s) for s in srcs):
        z = np.load(cache)
        return {"frames": z["frames"], "boxes": z["boxes"]}
    data = parse_match(ann_root, match)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache, **data)
    return data


def split_matches(matches: list[str], split: str, val_matches=None) -> list[str]:
    val = set(DEFAULT_VAL_MATCHES if val_matches is None else val_matches)
    if split == "train":
        return [m for m in matches if m not in val]
    if split == "val":
        return [m for m in matches if m in val]
    if split == "all":
        return list(matches)
    raise ValueError(f"split must be train / val / all, got {split!r}")


def find_video(video_root: str, match: str) -> str | None:
    hits = glob.glob(os.path.join(video_root, "**", f"{glob.escape(match)}.mp4"), recursive=True)
    return hits[0] if hits else None


def frame_path(frames_root: str, match: str, frame: int) -> str:
    return os.path.join(frames_root, match, f"{frame:06d}.jpg")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class BadmintonDetection(Dataset):
    """
    Each item: (image, target)
        image  : tv_tensors.Image uint8 (3, H, W) RGB, or whatever `transform` returns
        target : {"boxes": BoundingBoxes float32 (K, 4) xyxy pixels,
                  "labels": int64 (K,), "image_id": int, "match": str, "frame": int}

    `transform` is called as transform(image, target), so torchvision.transforms.v2
    pipelines (RandomHorizontalFlip, RandomResizedCrop, ...) move the boxes together
    with the image. Add v2.SanitizeBoundingBoxes() after crops to drop boxes that
    fall outside the crop.

    Args:
        ann_root:      BFMD_data/annotations
        frames_root:   folder from `extract` (preferred image source)
        video_root:    folder holding <match>.mp4 files (fallback image source)
        split:         "train" | "val" | "all"   (split by match)
        matches:       explicit list of match names (overrides split)
        stride:        keep every n-th annotated frame (consecutive frames are near-duplicates)
        classes:       subset of {"top_player", "bottom_player", "shuttle"} to keep
        require_shuttle: only keep frames where the shuttle was tracked
        min_box_size:  drop boxes smaller than this (pixels, either side)
        cache_dir:     where parsed annotations are cached as .npz (None = no cache)
    """

    def __init__(
        self,
        ann_root: str = DEFAULT_ANN_ROOT,
        frames_root: str | None = None,
        video_root: str | None = None,
        split: str = "train",
        matches: list[str] | None = None,
        val_matches: list[str] | None = None,
        stride: int = 1,
        classes=("top_player", "bottom_player", "shuttle"),
        require_shuttle: bool = False,
        min_box_size: float = 2.0,
        transform=None,
        cache_dir: str | None = "dataset/bfmd_cache",
    ):
        if frames_root is None and video_root is None:
            raise ValueError("Give frames_root (extracted JPEGs) and/or video_root (mp4 files).")
        self.ann_root = ann_root
        self.frames_root = frames_root
        self.video_root = video_root
        self.transform = transform
        self.min_box_size = min_box_size
        self.class_ids = np.array(sorted(CLASSES.index(c) for c in classes), dtype=np.int64)

        all_matches = list_matches(ann_root)
        if not all_matches:
            raise FileNotFoundError(f"No player_bbox/shuttle annotations found under {ann_root}")
        self.matches = list(matches) if matches is not None else split_matches(all_matches, split, val_matches)

        # Flat index: (match_id, frame) + boxes for that frame
        self.videos: dict[str, str | None] = {}
        idx_match, idx_frame, idx_boxes = [], [], []
        for mi, m in enumerate(self.matches):
            data = load_match(ann_root, m, cache_dir)
            frames, boxes = data["frames"], data["boxes"]
            keep = np.zeros(len(frames), dtype=bool)
            keep[::stride] = True
            if require_shuttle:
                keep &= ~np.isnan(boxes[:, SHUTTLE - 1, 0])
            # frame must contain at least one of the requested classes
            keep &= (~np.isnan(boxes[:, self.class_ids - 1, 0])).any(axis=1)
            idx_match.append(np.full(keep.sum(), mi, dtype=np.int32))
            idx_frame.append(frames[keep])
            idx_boxes.append(boxes[keep])
            if video_root is not None:
                self.videos[m] = find_video(video_root, m)
        self.idx_match = np.concatenate(idx_match)
        self.idx_frame = np.concatenate(idx_frame)
        self.idx_boxes = np.concatenate(idx_boxes)

        self._caps: dict[str, cv2.VideoCapture] = {}  # per worker process
        self._last_read: dict[str, int] = {}

    def __len__(self):
        return len(self.idx_frame)

    # -- image loading ------------------------------------------------------ #
    def _read_from_video(self, match: str, frame: int) -> np.ndarray | None:
        path = self.videos.get(match)
        if path is None:
            return None
        cap = self._caps.get(match)
        if cap is None:
            cap = self._caps[match] = cv2.VideoCapture(path)
        if self._last_read.get(match) != frame - 1:  # avoid seeking when reading sequentially
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        self._last_read[match] = frame if ok else -10
        return img if ok else None

    def load_image(self, index: int) -> np.ndarray:
        """BGR uint8 (H, W, 3)."""
        match, frame = self.matches[self.idx_match[index]], int(self.idx_frame[index])
        img = None
        if self.frames_root is not None:
            p = frame_path(self.frames_root, match, frame)
            if os.path.exists(p):
                img = cv2.imread(p)
        if img is None and self.video_root is not None:
            img = self._read_from_video(match, frame)
        if img is None:
            raise FileNotFoundError(
                f"Frame {frame} of {match} not found (frames_root={self.frames_root}, video={self.videos.get(match)})"
            )
        return img

    def get_target(self, index: int) -> dict:
        boxes = self.idx_boxes[index][self.class_ids - 1]  # (C, 4)
        labels = self.class_ids
        ok = ~np.isnan(boxes[:, 0])
        boxes, labels = boxes[ok], labels[ok]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, IMG_W)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, IMG_H)
        big = ((boxes[:, 2] - boxes[:, 0]) >= self.min_box_size) & ((boxes[:, 3] - boxes[:, 1]) >= self.min_box_size)
        return {"boxes": boxes[big], "labels": labels[big]}

    def __getitem__(self, index: int):
        img = cv2.cvtColor(self.load_image(index), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        t = self.get_target(index)
        image = tv_tensors.Image(torch.from_numpy(img).permute(2, 0, 1).contiguous())
        target = {
            "boxes": tv_tensors.BoundingBoxes(
                torch.as_tensor(t["boxes"], dtype=torch.float32).reshape(-1, 4),
                format="XYXY", canvas_size=(h, w),
            ),
            "labels": torch.as_tensor(t["labels"], dtype=torch.int64),
            "image_id": index,
            "match": self.matches[self.idx_match[index]],
            "frame": int(self.idx_frame[index]),
        }
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target

    def __getstate__(self):  # VideoCapture can't be pickled into DataLoader workers
        state = self.__dict__.copy()
        state["_caps"], state["_last_read"] = {}, {}
        return state


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


# --------------------------------------------------------------------------- #
# CLI: frame extraction / stats / visual check
# --------------------------------------------------------------------------- #
def extract_frames(ann_root, video_root, out, stride=1, matches=None, quality=95):
    """Decode each video sequentially once and save only the annotated frames."""
    for m in matches or list_matches(ann_root):
        video = find_video(video_root, m)
        if video is None:
            print(f"[skip] no video for {m}")
            continue
        wanted = set(load_match(ann_root, m, "dataset/bfmd_cache")["frames"][::stride].tolist())
        os.makedirs(os.path.join(out, m), exist_ok=True)
        todo = {f for f in wanted if not os.path.exists(frame_path(out, m, f))}
        print(f"{m}: {len(todo)}/{len(wanted)} frames to extract")
        if not todo:
            continue
        cap = cv2.VideoCapture(video)
        last = max(todo)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        i = 0
        while i <= last:
            if i in todo:
                ok, img = cap.read()
                if not ok:
                    break
                cv2.imwrite(frame_path(out, m, i), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif not cap.grab():  # grab() skips decoding to BGR -> faster
                break
            i += 1
        cap.release()


def print_stats(ann_root, stride=1):
    print(f"{'match':<75} {'frames':>7} {'top':>7} {'bottom':>7} {'shuttle':>7}")
    tot = np.zeros(4, dtype=np.int64)
    for m in list_matches(ann_root):
        b = load_match(ann_root, m, "dataset/bfmd_cache")["boxes"][::stride]
        row = np.array([len(b), *(~np.isnan(b[:, :, 0])).sum(0)])
        tot += row
        tag = " (val)" if m in DEFAULT_VAL_MATCHES else ""
        print(f"{(m + tag)[:75]:<75} {row[0]:>7} {row[1]:>7} {row[2]:>7} {row[3]:>7}")
    print(f"{'TOTAL':<75} {tot[0]:>7} {tot[1]:>7} {tot[2]:>7} {tot[3]:>7}")


def draw_sample(ds: BadmintonDetection, index: int, out: str):
    img = ds.load_image(index).copy()
    t = ds.get_target(index)
    colors = {1: (0, 165, 255), 2: (0, 0, 255), 3: (0, 255, 255)}
    for (x1, y1, x2, y2), c in zip(t["boxes"].astype(int), t["labels"]):
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[int(c)], 2)
        cv2.putText(img, CLASSES[c], (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[int(c)], 1)
    cv2.imwrite(out, img)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description="BFMD badminton detection dataset tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "stats", "show"):
        s = sub.add_parser(name)
        s.add_argument("--ann-root", default=DEFAULT_ANN_ROOT)
        s.add_argument("--stride", type=int, default=1)
        if name in ("extract", "show"):
            s.add_argument("--video-root", default=os.path.join(DEFAULT_ROOT, "videos"))
        if name == "extract":
            s.add_argument("--out", default="dataset/bfmd_frames")
            s.add_argument("--match", default=None, help="substring filter on match name")
        if name == "show":
            s.add_argument("--frames-root", default=None)
            s.add_argument("--index", type=int, default=0)
            s.add_argument("--out", default="sample.jpg")
    a = p.parse_args()

    if a.cmd == "stats":
        print_stats(a.ann_root, a.stride)
    elif a.cmd == "extract":
        ms = [m for m in list_matches(a.ann_root) if a.match is None or a.match in m]
        extract_frames(a.ann_root, a.video_root, a.out, a.stride, ms)
    elif a.cmd == "show":
        ds = BadmintonDetection(a.ann_root, frames_root=a.frames_root, video_root=a.video_root,
                                split="all", stride=a.stride)
        draw_sample(ds, a.index, a.out)


if __name__ == "__main__":
    sys.exit(main())
