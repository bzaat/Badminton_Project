# Badminton Singles Video Analysis

> **Status: work in progress.** Early-stage research project. No final results yet.
> The baselines below are first versions; numbers in [REPORT.md](REPORT.md) are working
> measurements, not final results.

## Goal

Build a computer-vision pipeline that turns broadcast badminton singles video
into structured data and player-level insight. The long-term goal is a system
that detects players and the shuttlecock, estimates pose, recognizes hits and
shot types, measures player movement, and later summarizes a player's style
and weaknesses for coaches.

## Current focus (Milestone 1)

Working on the BFMD dataset (Full-Match Badminton Dense dataset), using cached annotations
from 12 BWF Super 1000 men's singles matches (2025). The analysis code is in
`badminton_analysis/`; evidence and data checks are in [REPORT.md](REPORT.md) and
[AUDIT.md](AUDIT.md).

| Item | Status |
|---|---|
| **Data layer and court calibration**: BFMD annotations parsed into per-match arrays; court homography from BFMD's court corners; player positions in court metres | Baseline implemented |
| **Hit-event detection**: from shuttle trajectory changes and wrist motion, evaluated against annotated hit events with a frame tolerance | Baseline implemented |
| **Shot-type recognition**: a simple skeleton + trajectory baseline, evaluated with a match-level split (held-out matches) | Baseline implemented |
| **Movement features**: distance covered, speed, court-zone occupancy and recovery position, measured in court coordinates | Baseline implemented (measurements only) |
| **Player and shuttle detection**: Faster R-CNN detector for the two players and the shuttle (`train.py`) | In progress: code in place, no training results yet |
| **Shuttle tracking**: TrackNetV3 as the reference model | In progress: comparison code on branch `task5-tracknetv3`, not merged |
| **Player tracking** (near and far player) | Planned: players currently come from BFMD's manual boxes |
| **Pose estimation including foot keypoints** | Planned: currently uses BFMD's provided COCO-17 pose (ankles, no foot keypoints) |

## Roadmap

- **Phase 1**: detect players, shuttle and pose; court calibration.
- **Phase 2**: hit-type recognition, footwork analysis, per-player stats board.
  Footwork "error" definitions are still undecided and will be designed with a
  coach; for now only measurable movement features are computed.
- **Phase 3**: player style summary with uncertainty estimates, weakness
  detection (with confidence intervals and minimum sample sizes), and
  evidence-based strategy suggestions for coaches.

## Scope and limitations

- Singles only, broadcast video first. Own-camera data may come later.
- The dataset contains the same players across many matches, so results may
  not generalize to unseen players. Evaluation is split by match
  (leave-one-match-out); for shot types, per-player results are also reported separately.
- Datasets, videos and model checkpoints are not included in this repo.
  Source video comes from public BWF broadcasts; check the dataset and video
  terms before reuse.

## Running Milestone 1

```bash
uv sync
uv run python -m badminton_analysis all          # everything below in one go (~8 min, CPU)
uv run python -m badminton_analysis build        # parse BFMD JSON -> dataset/bfmd_m1/
uv run python -m badminton_analysis validate     # data checks -> outputs/task1/
uv run python -m badminton_analysis viz --match Antonsen-DEN-3-vs.-Chou --game 1 --rally 2
uv run python -m badminton_analysis task2        # hit detection    -> outputs/task2/
uv run python -m badminton_analysis task3        # shot types       -> outputs/task3/
uv run python -m badminton_analysis task4        # movement CSVs    -> outputs/task4/
uv run pytest
```

Google Colab: open `notebooks/colab_milestone1.ipynb` (uses `configs/colab.yaml`).

Paths, frame shifts and thresholds live in `configs/default.yaml`. The BFMD location can also be
set with `BFMD_ROOT=/path/to/BFMD_data`. For another machine (e.g. Colab), copy the config and
change only `paths`, then pass `--config`.

## Project layout

```
badminton_analysis/    Milestone 1: data layer, court model, hits, shot types, movement
configs/               paths and parameters (default.yaml, colab.yaml)
notebooks/             Colab notebook for Milestone 1
tests/                 unit tests (synthetic data) + checks on the real data (skipped if absent)
badminton_dataset.py   BFMD -> PyTorch detection dataset + extract / stats / show CLI
train.py               detector training + per-class AP@0.5 validation (in progress)
main.py                Pascal VOC detection dataset (earlier experiment, unrelated)
dataset/               local data: extracted frames, annotation cache (git-ignored)
trained_models/        checkpoints (git-ignored)
```

## Setup

Requires Python ≥ 3.10. A CUDA GPU is needed for reasonable detector training speed; Milestone 1
runs on the CPU.

```bash
uv sync            # or: pip install torch torchvision opencv-python tqdm
```

### Data

1. Get the BFMD data package (`BFMD_data/`, annotations only, ~1.8 GB).
   By default the scripts look in
   `D:/badminton/BFMD_data-20261026T161257Z-1-001/BFMD_data` on Windows or the same path
   under `/mnt/d/` in WSL. Pass `--ann_root` / `--ann-root` to use another location.
2. Download the match videos with the script that ships with BFMD (needs `yt-dlp` and `ffmpeg`):
   ```bash
   cd <BFMD_data>
   python download_youtube.py
   ```
3. Extract the annotated frames once (only needed for the detector). Decoding is sequential,
   which is much faster than seeking into the video for every sample:
   ```bash
   python badminton_dataset.py extract --video-root <BFMD_data>/videos --out dataset/bfmd_frames --stride 5
   ```
   `--stride 5` keeps every 5th annotated frame (about 66k images). Neighbouring frames are
   nearly identical, so a stride costs little accuracy. `--match <substring>` extracts only
   matching matches. Train on a stride that is a multiple of the one you extracted.

Check the data:

```bash
python badminton_dataset.py stats                                         # frames / boxes per match
python badminton_dataset.py show --frames-root dataset/bfmd_frames --index 0 --out sample.jpg
```

## Detector training (in progress)

Faster R-CNN (MobileNetV3-Large FPN) for three classes: `top_player` (far side),
`bottom_player` (near side) and `shuttle`. Player boxes come from BFMD's manual
`annotations/player_bbox`; shuttle boxes from `annotations/shuttle` (TrackNet predictions).
Only the 12 singles matches have boxes; the 7 doubles matches are not used.
No full training run has been reported yet.

```bash
python train.py --amp
```

Useful options (`python train.py -h` for all):

| Option | Default | Meaning |
|---|---|---|
| `--frames_root` | `dataset/bfmd_frames` | extracted frames |
| `--video_root` | – | read frames straight from the mp4s instead (slower) |
| `--stride` / `--val_stride` | 5 / 30 | keep every n-th annotated frame |
| `--image_size` | 720 | shorter side the model resizes to |
| `--batch_size` | 8 | |
| `--epoch` | 10 | |
| `--lr` / `--warmup_iters` | 0.01 / 500 | SGD, linear warmup then cosine decay |
| `--amp` | off | mixed-precision training |
| `--default_anchors` | off | use torchvision's 32–512 px anchors instead of 16–256 px |
| `--trained_model` | `trained_models` | checkpoint directory |
| `--resume` | off | continue from `last_model.pth` |

After each epoch the script reports AP@0.5 per class and their mean, and saves
`last_model.pth` and `best_model.pth` (best mean AP).

### Design choices

- **Split by match.** Validation uses two whole matches
  (`DEFAULT_VAL_MATCHES` in `badminton_dataset.py`), so no frames of a validation match are
  seen in training.
- **Native resolution, smaller anchors.** The shuttle box is only ~10×10 px, so images are
  kept at 720p and the smallest anchor is 16 px.
- **Light augmentation.** Horizontal flip and colour jitter only. Random crops are left out
  because they easily cut away the shuttle. A horizontal flip keeps the top/bottom player
  labels correct.
- **No `Normalize` in the transforms.** torchvision detection models normalise inputs
  themselves.

### Using the dataset in your own code

```python
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch
from badminton_dataset import BadmintonDetection, collate_fn

transform = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.ToDtype(torch.float32, scale=True),
    v2.SanitizeBoundingBoxes(),
])
ds = BadmintonDetection(frames_root="dataset/bfmd_frames", split="train", stride=5, transform=transform)
image, target = ds[0]
# target: {"boxes": BoundingBoxes (K, 4) xyxy pixels, "labels": (K,), "image_id", "match", "frame"}
loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)
```

Other options: `split="train" | "val" | "all"`, `matches=[...]` for an explicit list,
`classes=("shuttle",)` to keep a subset of classes, `require_shuttle=True` to keep only
frames where the shuttle was tracked. Parsed annotations are cached in `dataset/bfmd_cache/`.

## Known caveats

- Shuttle boxes come from TrackNet, not from people. They have a fixed size and some
  frames are noisy or missing the shuttle (about 4% of frames).
- In BFMD, player-box frame numbers run one frame ahead of the rally boundaries in
  `metadata/`. The detector dataset uses each annotation file's own frame numbers as they are;
  Milestone 1 aligns the sources with the frame shifts in `configs/default.yaml`.
- Video copyright belongs to BWF TV; don't redistribute the videos or the extracted frames.

## References

- BFMD: Ding, Fujii, Tamaki, "BFMD: A Full-Match Badminton Dense Dataset for
  Dense Shot Captioning" (arXiv:2603.25533; CVPR 2026 Workshops)
- TrackNetV3: Chen and Wang, "TrackNetV3: Enhancing ShuttleCock Tracking with
  Augmentations and Trajectory Rectification" (ACM MMAsia 2023)
- BST: Chang, "BST: Badminton Stroke-type Transformer for Skeleton-based
  Action Recognition in Racket Sports" (arXiv:2502.21085)
- ShuttleSet / ShuttleSet22 stroke-level datasets (Wang et al.)

```bibtex
@InProceedings{Ding_2026_CVPR,
    author    = {Ding, Ning and Fujii, Keisuke and Tamaki, Toru},
    title     = {BFMD: A Full-Match Badminton Dense Dataset for Dense Shot Captioning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2026},
    pages     = {9909-9918}
}
```
