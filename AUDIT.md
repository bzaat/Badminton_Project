# AUDIT — Milestone 1, Task 0 (read-only)

Date: 2026-11-13. Nothing in the repo or data was modified; this file is the only addition.
All numbers below were measured by scripts run against the files, unless marked *(claimed)*.

---

## 1. What the current code does

| File | What it does |
|---|---|
| `badminton_dataset.py` | Parses BFMD `player_bbox` + `shuttle` JSON into a PyTorch **object-detection** dataset (classes: `top_player`, `bottom_player`, `shuttle`). Has CLI `extract` (decode annotated frames from mp4 → JPEG), `stats`, `show`. Writes the `.npz` cache via `load_match()`. Fixed match-level val split (`DEFAULT_VAL_MATCHES`: China Open Chou–Shi SF, All England Lee–Lanier SF). |
| `train.py` | Faster R-CNN MobileNetV3-Large-FPN, smaller anchors for the ~10 px shuttle, SGD + warmup/cosine, AMP, per-class VOC AP@0.5. Checkpoints to `trained_models/`. |
| `main.py` | Earlier Pascal VOC 2012 dataset wrapper experiment (unrelated to badminton). |
| `README.md` | Documents the detection project; points to BFMD on `D:` and the BFMD video download script. |
| `pyproject.toml` | Python ≥3.10; deps: torch ≥2.14.1, torchvision ≥0.29.2, opencv-python, tqdm. Description is still the placeholder. |

So the repo today is a **detection** project. None of the Milestone-1 pieces (pose, hits, shot types, court, movement) exist yet.

---

## 2. The data

### 2a. `dataset/bfmd_cache/*.npz` — confirmed: derived from BFMD, but boxes only

The cache is written by `badminton_dataset.py:load_match()` and contains **only two arrays**:

| key | shape | dtype | meaning |
|---|---|---|---|
| `frames` | (N,) | int64 | 0-based frame index in the full 30 fps broadcast video |
| `boxes` | (N, 3, 4) | float32 | pixel xyxy for [top_player, bottom_player, shuttle]; NaN if absent |

- 12 files, 329,168 frames total (matches the README claim). Video spec 1280×720, 30 fps *(from BFMD README; not checked, since no video is present)*.
- Shuttle boxes are a fixed 10.24×10.24 px (TrackNet centre + constant size), so they carry **position only**.
- Contiguous frame runs per file = 77, 81, 109, 70, 65, 95, 74, 106, 75, 112, 78, 112 (1,054 total). These equal the `Rally` interval counts in `shot_type/`, so each run is one rally.
- **Missing from the cache:** pose, hit events, shot types, rally/game/score structure, player identities, court geometry. The cache alone cannot support Tasks 2–4.

### 2b. The full BFMD package exists locally (annotations only)

`/mnt/d/badminton/BFMD_data-20261026T161257Z-1-001/BFMD_data/` (1.8 GB; a 324 MB zip sits next to it).
This has everything the cache lacks. Per-match JSON, keyed by match name:

| Folder | Present for | Content (verified) |
|---|---|---|
| `metadata/` | 12 singles | players, `game1_top_player`, per-game rally `[start,end]` frames, score after each rally, winners |
| `hit_inferred/` | 12 singles | **11,301 hits**: `{frame, shot_type, side (top/bottom), player (name), game, rally}` |
| `shot_type/` | 12 singles | Manual Label Studio timeline. `hit`, `bounce`, `net_hit` and each shot type are **single-frame** marks (range length 0); `Rally`, `replay`, `hawk_eye_challenge` and interval breaks are ranges. `hit_inferred.frame` equals the manual `hit` frame exactly (checked on one match, 978/978). |
| `pose/` | 12 singles (1.6 GB) | Per rally: `{bbox: pixel xyxy, keypoints: 17 × [x, y, score]}`. **COCO-17**, x/y **normalized 0–1** (verified against the bbox). No track ID. |
| `shuttle/` | 12 singles | TrackNet predictions, percent coordinates, one track |
| `player_bbox/` | 12 singles | Manual top/bottom player boxes, percent coordinates, interpolated |
| `court/` | 12 singles | One static calibration per match: 4 court corners + 4 net points (pixels), **main camera only** |
| `court_perframe/` | 12 singles | Automatic corners at ~1 Hz with flags `valid` / `refit` / `no_court`. Over the whole video, 35–50% of samples are `valid`; breaks and replays are included in that count. |
| `caption/` | 12 singles | Per-shot text captions (not needed for M1) |
| `hits_doubles/` | 7 doubles | Hit events only, no shot types, no pose/shuttle/court. **Out of scope (singles only).** |

**Singles vs doubles:** every file in `hit_inferred/pose/shuttle/court` is one of the 12 men's singles matches. The singles filter is simply "match is in `hit_inferred/`" (or `match_index.csv`).

**Shot-type distribution (12 classes, `hit_inferred`):** net_shot 2875, lift 2098, smash 1354, block 919, drop 905, drive 868, serve 833, clear 606, push 510, press 155, net_kill 93, flick_serve 85. The classes are strongly imbalanced.

**Players (hits per player):** Shi Yu Qi 3304, Anders Antonsen 2293, Chou Tien Chen 1307, Li Shi Feng 1204, Lee Chia Hao 859, Wang Zheng Xing 805, Kodai Naraoka 579, Kunlavut Vitidsarn 495, Alex Lanier 455.
Only 9 players across 12 matches. **Shi Yu Qi plays in 7 of 12 matches, Antonsen in 5, Chou in 3.** Kodai, Kunlavut and Lanier each appear in a single match.

**Coverage, measured per match:**

| | range across the 12 matches |
|---|---|
| pose frames with both players | 93–99%; the rest have 1 person |
| person[0] is the top player (by bbox) | 97.2–100%; **not guaranteed**, so it must be assigned explicitly |
| keypoints with score < 0.3 | 0.4–1.2% |
| hits that have a pose frame | 92.7–99.7% |
| hits with shuttle present at every frame in ±2 | 91.2–95.6% |
| median gap between consecutive hits in a rally | 23–29 frames (~0.8 s) |
| minimum gap | 7–9 frames; **1 frame** in Malaysia Open F (probably an annotation error) |

### 2c. Inconsistencies found

1. Rally count: `metadata` gives 1,058 rallies but `shot_type` `Rally` intervals give 1,054. Off by 1 in Malaysia Shi–Li, by 2 in China Chou–Shi, by 1 in AE Lee–Lanier.
2. The `metadata/` folder holds a stray `...Lee-Chia-Hao-TPE-vs.-Alex-Lanier-FRA-SF.json.bak`.
3. The BFMD README says ~6,500 of 11,301 captions are "refined", but all 11,301 are non-null in this copy, so the copy is newer than its README.
4. The repo README notes that player-box frames run one frame ahead of the `metadata/` rally boundaries. I have not re-verified this; I will test the offset in Task 1.
5. `shot_type` contains 3 `other` labels (China Open F) that do not appear in `hit_inferred`.
6. The court corners' definition is **not stated**: singles or doubles sidelines, back line or long-service line. I can't check visually without video frames, and the homography depends on it.
7. The paper (arXiv) describes 19 matches, 1,687 rallies and 16,751 hits including doubles. The local singles subset is 1,054–1,058 rallies and 11,301 hits, which is consistent with that.

### 2d. Not available

- **Videos:** not present anywhere under `/mnt/d/badminton`; the `videos/` symlinks did not survive the export. The singles videos total ~9.6 GB according to `match_index.csv`.
- Without videos: no frame overlays, no Task 5 (TrackNetV3), and no visual check of the court corners. BFMD's own `vis.py` also needs the videos.

---

## 3. Environment

| Item | Status |
|---|---|
| GPU | NVIDIA RTX 3050 **Laptop, 6 GB** (~2 GB already in use by the display) |
| torch | 2.14.1+cu130, `cuda.is_available()` = **True** |
| torchvision | 0.29.2+**cu132**, which does not match torch's cu130. This may break CUDA ops (e.g. NMS); worth checking with `uv sync`. |
| Python | 3.10.12 (`.python-version` = 3.10) |
| uv | 0.11.14 at `~/.local/bin/uv`; on PATH only in login shells |
| RAM (WSL) | 7.8 GB, **no swap**. Pose JSON is ~130 MB per match: parse one match at a time and cache to compact arrays. |
| Disk | WSL `/` reports 909 GB free, but **C: is 98% full (12 GB free)**, and the WSL disk image normally lives on C:. **D: has 271 GB free.** Large data (videos, frames) should go on D:. |
| Pascal VOC | `dataset/VOCdevkit/VOCtrainval_11-May-2012.tar` (2.0 GB) + extracted copy (3.8 GB total). Not needed for this project; it could be deleted to free space, but only if you decide that. |

**`.gitignore`:** it already ignores `dataset/`, `.venv`, `trained_models/`, `*.pth`, `__pycache__`, `.idea/`. Only small source files are tracked (largest: `uv.lock`, 252 KB).
**Not ignored:** `outputs/`, `checkpoints/`, `runs/`, `*.pt`, `*.ckpt`, `*.mp4`, `*.npz` outside `dataset/`. I'll add these in Task 1.

---

## 4. Licensing and terms

| Item | Finding | Verified? |
|---|---|---|
| BFMD paper | arXiv 2603.25533, Ding, Fujii, Tamaki, "CVSports2026 accepted". Distributed under the arXiv non-exclusive distribution licence, which covers the **paper only**. | Yes (arXiv abstract page) |
| BFMD annotations | **No licence found.** The paper/arXiv page has no dataset link, licence or terms of use, and the local package README has none either; it only asks for a citation. The package looks like a Google Drive export (folder-name timestamp). | Licence not found. **Where did you get it, and under what terms?** |
| BWF match videos | The paper says they come from "publicly available broadcast recordings released by the BWF via its official YouTube channel". The BFMD README says "Video copyright belongs to BWF TV; do not redistribute". BWF's terms-of-use pages returned HTTP 403, so I could not read them. Separately, YouTube's Terms of Service generally prohibit downloading without permission or a YouTube-provided download feature. That is from my own knowledge; I did not re-fetch the terms. | **Unverified.** Treat as no redistribution. Downloading is a decision for you, and possibly your supervisor. |
| TrackNetV3 (qaz812345) | The repo README says the code **and** the pretrained checkpoints (Google Drive) are MIT-licensed. I did not open the LICENSE file itself (the default branch is `master`). Developed on Python 3.8.7 / torch 1.10.0. | Partly (README text). |

Working assumption until you say otherwise: use the annotations for research only, cite BFMD, never commit or redistribute annotations, videos, frames or derived per-frame data.

---

## 5. Implications for Tasks 1–5, and decisions I need from you

1. **Data source for Task 1.** The cache cannot support Tasks 2–4. I propose a new loader that reads the raw BFMD JSON from `D:` (path in YAML config) and writes a richer per-match cache to a **new** folder (`dataset/bfmd_m1/`). The existing `bfmd_cache/`, `badminton_dataset.py` and `train.py` would stay untouched.
2. **Visualizer without video.** Overlays would be drawn on a blank canvas, plus a top-down court view (a homography from `court/` is available, so the click-to-annotate tool is probably unnecessary). Overlays on real frames need the videos.
3. **Hit ground truth and tolerance (research choice).** The GT is a single manually chosen frame per hit. At 30 fps I propose **±2 frames (67 ms) as the primary metric, and also reporting ±1, ±3 and ±5**. The tolerance must stay below half the minimum hit gap (~7 frames) so one prediction can't match two hits. Matching would be one-to-one (Hungarian/greedy) within a rally. Alternatives: a single looser ±5 tolerance, which is common in the literature but inflates recall.
4. **Shot-type split (research choice).** There are 12 matches and 9 players, with Shi Yu Qi in 7 matches.
   - (a) **Leave-one-match-out CV (12 folds)**, recommended: uses all the data and gives a per-match variance.
   - (b) A fixed train/val/test split by match: simpler, but a 2–3-match test set is noisy.
   - (c) A "less-seen opponent" check: test on matches involving Kodai, Kunlavut or Lanier. The opponent is still seen in training, so this can only *flag* generalisation, not measure it.

   Also: merge `serve`/`flick_serve` and fold rare `net_kill`/`press`? I'd keep all 12 and report macro-F1 plus a confusion matrix, then try a merged variant.
5. **Court-coordinate features.** Use the ankle midpoint (keypoints 15/16) or the bbox bottom-centre as the ground point, projected through the static homography. Only frames inside rallies are used, where the main camera is the norm, but `court_perframe` flags will be checked to drop frames where the camera cut away. The corner definition (item 6 in §2c) needs to be confirmed, ideally on one real frame.
6. **Videos / Task 5.** Downloading needs ~9.6 GB for all 12 matches, or ~0.6–1 GB for one. That is over your 1 GB threshold for all of them, the licensing is unverified, and C: is nearly full. I won't download anything without your explicit OK. Task 5 stays blocked until then.

---

*Inspection scripts (outside the repo, not committed) are in the session scratchpad.*
