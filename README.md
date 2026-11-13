# Badminton Object Detection

> **Status: work in progress.** The dataset loader and training script are in place, but no full training run or benchmark results have been published yet. Results will be added here once available.

Detect the **two players** and the **shuttlecock** in broadcast badminton footage with
Faster R-CNN (MobileNetV3-Large FPN), trained on the
[BFMD](https://openaccess.thecvf.com/content/CVPR2026W/CVsports/papers/Ding_BFMD_A_Full-Match_Badminton_Dense_Dataset_for_Dense_Shot_Captioning_CVPRW_2026_paper.pdf)
full-match badminton dataset.

| Label | Class | Source in BFMD |
|---|---|---|
| 0 | background | — |
| 1 | `top_player` (far side of the court) | `annotations/player_bbox` (manual) |
| 2 | `bottom_player` (near side) | `annotations/player_bbox` (manual) |
| 3 | `shuttle` | `annotations/shuttle` (TrackNet predictions) |

The 12 men's singles matches provide **329,168 annotated rally frames** (1280×720, 30 fps).
The 7 doubles matches in BFMD have no boxes and are not used.

## Milestone 1: singles analysis (`badminton_analysis/`)

A separate package for hit detection, shot types and movement features on the BFMD annotations.
It does not touch the detection code below. Results and evidence are in [REPORT.md](REPORT.md),
and the data audit is in [AUDIT.md](AUDIT.md).

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
badminton_dataset.py   BFMD -> PyTorch detection dataset + extract / stats / show CLI
train.py               training + per-class mAP@0.5 validation
main.py                Pascal VOC detection dataset (earlier experiment)
dataset/               local data: extracted frames, annotation cache, VOC (git-ignored)
trained_models/        checkpoints (git-ignored)
```

## Setup

Requires Python ≥ 3.10 and a CUDA GPU for reasonable training speed.

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
3. Extract the annotated frames once. Decoding is sequential, which is much faster than
   seeking into the video for every sample:
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

## Training

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

## Using the dataset in your own code

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
  `metadata/`. Each annotation file's own frame numbers are used as they are.
- Video copyright belongs to BWF TV; don't redistribute the videos or the extracted frames.

## Citation

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
