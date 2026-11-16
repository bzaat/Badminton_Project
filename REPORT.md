# Milestone 1 report

Badminton singles analysis on the BFMD annotations (12 men's singles matches, BWF Super 1000, 2025).
Every number below was produced by the code in `badminton_analysis/`. Re-run with the commands
listed per task. Generated details: `outputs/task1/validation.md` (git-ignored).

---

## Task 1: data layer — status: done

```bash
uv run python -m badminton_analysis build         # raw JSON -> dataset/bfmd_m1/ (~60 s, 0.6 GB RAM peak)
uv run python -m badminton_analysis validate      # -> outputs/task1/validation.{md,json}
uv run python -m badminton_analysis viz --match Antonsen-DEN-3-vs.-Chou --game 1 --rally 2
uv run python -m badminton_analysis viz-samples   # 1 typical rally per match + all suspect rallies
uv run pytest                                     # 36 tests (5 need the real data, else skipped)
```

### What was built
- `MatchData` (schema in `badminton_analysis/schema.py`): one row per rally frame (329,819 frames,
  1,058 rallies, 11,301 hits), top/bottom player axis, shuttle (px), manual player boxes, COCO-17
  pose assigned to top/bottom, ground point in px and court metres, camera state per frame,
  rally table (winner, score, players, quality flags) and hit table. Validated on load.
- Singles filter: a match must have hits, pose, shuttle, player boxes, metadata and court files;
  this is exactly the 12 singles matches (doubles only have `hits_doubles/`).
- Court model (BWF dimensions) and homographies (normalized DLT); visualiser with image-space,
  top-down court and signal-timeline panels. The click-to-annotate fallback was **not** built:
  a static court calibration exists for all 12 matches.

### Measured decisions (all in `configs/default.yaml`)
| Decision | Evidence |
|---|---|
| Court corners = **doubles outer boundary** | Net-post bases project to x = 0 / 6.10 m within 5 cm in all 12 matches; the singles-sideline hypothesis is off by up to 47 cm. Net line lands at y = 6.58 m on average (expected 6.70). |
| `player_bbox` frames **+1** | Pose boxes equal the manual boxes one frame later: IoU 0.97–1.00 at that shift vs 0.88–0.91 unshifted, all 12 matches. Pose clip keys, rally intervals and hit frames already agree with each other. |
| Shuttle frames **+0** | Shuttle-to-hitter-wrist closest approach is smallest at shift 0 in 9/12 matches (−1 in 2, −2 in 1; curves are flat within ~1 px). |
| Ground point = ankles **+ 0.10 × box height** | Raw ankles put far-side servers 4.3 m and near-side servers 2.7 m from the net (should be equal). Fitted on 6 matches; on the other 6, server asymmetry 0.01 m and receiver asymmetry −0.04 m (from 1.6 / 1.5 m). |
| Rally winner from score change | The explicit winner field is null for 981/1,058 rallies; score-derived winners agree with the captions' `rally_winner` on 1,053/1,053. |

### Validation results (real data)
- **Serve rule check** (server stands in the right court on an even score, left on odd):
  **555 / 563 = 98.6 %** of main-camera serves. This independently confirms the court x-mapping,
  top/bottom assignment and serve labels.
- **Serve depth after correction**: server 3.32 m / 3.32 m from the net (top / bottom, median of match
  medians), receiver 3.58 m / 3.59 m.
- Coverage: shuttle on 93–96 % of rally frames; hitter pose at 89–97 % of hits; both player boxes
  on 96–100 % of rally frames.
- Annotated hit vs shuttle-wrist closest approach: median offset −1 frame (the label is ~1 frame
  after the closest approach). Relevant for Task 2.

### Data problems found (flagged, not hidden)
1. **139 rallies (13 %) do not start with a serve** (53 in China Open F). In all 138 checkable cases
   the first hitter is the expected server, the hit comes 0.37 s after the annotated rally start,
   yet only 39 % pass the service-court rule (labelled serves: 98.6 %). Most likely the rally
   interval starts after the serve and the first hit is the server's third shot. Flag:
   `starts_with_serve`. **Unconfirmed without video.**
2. **21 rallies flagged `suspect`** (7 matches): on frames marked main-camera, players' feet are in
   the wrong half for > 20 % of the rally. Visual check: some are a different camera view that the
   ~1 Hz court detector marked `valid`; in others the far player appears ~50 px lower for the whole
   rally (a zoomed framing?). **64 rallies** have < 50 % main-camera frames.
3. TrackNet shuttle track contains isolated false detections (visible in the rally figures); two
   can coincide with hit frames. Outlier filtering belongs to Task 2.
4. 4 metadata "rallies" (14–117 frames) have no hits and no timeline interval; 0–3 rallies per match
   have two consecutive hits from the same side; one 1-frame hit gap (Malaysia Open F).
5. Camera state comes from ~1 Hz samples, so camera cuts shorter than ~1 s can be mislabelled.

### Video check (one match, China Open 2025 F, 720p30 video-only stream, 4,965.0 s = BFMD's 4,965.1 s)
- **Absolute alignment confirmed:** the shuttle annotation and the wrist keypoints sit on the
  shuttle / wrist in the *same* decoded frame and are visibly off at ±1
  (`outputs/task1/video_check/shuttle_alignment_grid.jpg`, `player_alignment_hit.jpg`).
  So shuttle +0, pose +0 and player_bbox +1 are correct against the video, not just relative.
- **Suspect rallies explained (for this match):** rally g1 r10 is a *different broadcast camera*
  (side angle). The static court overlay does not match the court, yet `court_perframe` marks it
  `valid`. Boxes and poses are correct; only the camera-state label is wrong.

### Limitations
- Alignment was checked against video for one match only.
- The ground-point offset is one global constant; the static homography is only valid for the
  main camera.

---

## Task 2: hit-event baseline — status: done

```bash
uv run python -m badminton_analysis task2   # ~65 s -> outputs/task2/{report.md, metrics.json, *.csv, task2_summary.png}
```

### Method
Image-space signals per rally frame: shuttle acceleration (velocity change over 2-frame steps,
after removing TrackNet outliers and filling ≤ 5-frame gaps), turn angle, shuttle speed,
wrist–shuttle distance and wrist speed (both relative to the player's box height). Candidates are
local maxima of acceleration (suppression radius r). Two scorers on those candidates:
- **rules** (`full`): acceleration ≥ A, nearest wrist within P box heights, wrist speed ≥ S;
- **logistic**: logistic regression over 7 candidate features.

**Evaluation:** leave-one-match-out. Every parameter (r, thresholds, frame offset, regression
weights) is chosen on the other 11 matches and evaluated on the held-out match. One-to-one
matching within ±tolerance. Design exploration (suppression radius, normalisation, timing
refinement) used only the 3 Indonesia matches; the other 9 were never inspected while designing.

### Results (pooled over the 12 held-out matches)
| variant | F1 @±1 | P / R / **F1 @±2** | F1 @±3 | F1 @±5 | macro F1 @±2 | F1 @±2 dev / not-dev |
|---|---|---|---|---|---|---|
| shuttle only | 0.615 | 0.701 / 0.679 / **0.690** | 0.732 | 0.775 | 0.687 ± 0.063 | 0.640 / 0.707 |
| + wrist proximity | 0.637 | 0.754 / 0.685 / **0.718** | 0.761 | 0.804 | 0.715 ± 0.061 | 0.663 / 0.737 |
| + wrist speed (`full`) | 0.637 | 0.757 / 0.683 / **0.718** | 0.761 | 0.804 | 0.716 ± 0.061 | 0.664 / 0.738 |
| **logistic** | 0.651 | 0.757 / 0.716 / **0.736** | 0.782 | 0.824 | 0.735 ± 0.062 | 0.690 / 0.753 |
| pose only (reference) | 0.178 | 0.232 / 0.376 / **0.287** | 0.395 | 0.535 | 0.288 ± 0.013 | 0.283 / 0.289 |

- Pose adds +2.8 F1 points over the shuttle alone (mostly precision, via wrist proximity);
  wrist *speed* adds nothing on top of proximity. The shuttle is essential (pose only: 0.29).
- No sign of design overfitting: the 9 never-inspected matches score higher than the dev matches.
- Large between-tournament differences: Malaysia Open 0.82–0.85, Indonesia Open 0.67–0.71.
- Logistic weights (standardized): wrist proximity −1.49, shuttle speed +0.94, missing pose
  +0.86, turn angle +0.63, wrist speed +0.53, log accel −0.17 (stable across folds, std ≤ 0.03).

### Failure analysis (logistic, ±2)
- **Serves are nearly undetected:** recall 0.133 (serve) / 0.082 (flick serve); the shuttle
  starts from rest and is visible on only 59 % of serve frames (dev matches). Recall excluding
  serves: **0.768**. 869 of 3,204 misses are the first hit of a rally.
- **Timing:** 1,255 misses have a prediction 3–8 frames away. On the dev matches, every way of
  picking the frame (acceleration peak, wrist–shuttle closest approach, maximum turn, 1-frame
  acceleration) puts only 70–72 % of near-hit candidates within ±2 of the label. A video check
  of 3 disagreements found the label at contact in 2 (detector late on a smash, early on a drop)
  and the label ~4 frames late in 1 (net shot). Both sources err; with n = 3 this is anecdotal.
- Other misses: 1,249 candidates scored below threshold (weak net shots / blocks: recall 0.71 /
  0.64), 432 with the shuttle missing, 268 with no trajectory peak at all.
- False positives: 1,895 mid-rally (largely the timing pairs above), 297 after the last hit
  (shuttle landing), 199 duplicates, 165 at annotated bounces / net hits; 365 have a removed
  shuttle outlier within ±3 frames.
- Hitter side (nearest wrist) is correct for 80.3 % of detected hits.
- Clean rallies F1 0.746 vs flagged rallies (suspect / < 50 % main camera / no serve) 0.687.

### Limitations / next steps for hits
- Every signal is derived from annotations (TrackNet shuttle, BFMD pose); errors there cap the
  baseline. Rally intervals come from the annotations too, so this is "hits within known rallies".
- A small manual re-annotation of contact frames from the video (e.g. 100 hits) would measure
  label timing noise and tell how much of the ±2 gap is label noise.
- Serve detection needs a dedicated cue (e.g. server pose before the first shuttle motion).

---

## Task 3: shot-type baseline — status: done

```bash
uv run python -m badminton_analysis task3   # ~5 min -> outputs/task3/{report.md, metrics.json, predictions.csv, confusion_fine.png}
```

### Method
37 named features per hit, in 4 groups:
- **context:** hit index in the rally, time since the previous / until the next hit, side;
- **shuttle:** velocity before and after the hit and its rise, in hitter body heights, in the
  hitter's body frame; shuttle position relative to the hitter at contact;
- **court position:** hitter / opponent position in metres, and the next hitter's position as a
  landing proxy;
- **pose:** upper-body joints relative to the hips, wrist above shoulder, wrist speed.

Models: majority class, logistic regression and gradient-boosted trees (`hgb`), with
hyperparameters **fixed in the config, not tuned**. Leave-one-match-out over the 12 matches,
evaluated on the **annotated** hits (main setting) and on Task 2's detections (end-to-end).
Features use the whole rally (offline analysis, not real-time).

### Results (held-out, pooled over 12 matches)
| model | 12 classes: accuracy | 12 classes: macro F1 [95 % CI] | 10 merged classes: macro F1 |
|---|---|---|---|
| majority | 0.254 | 0.034 | 0.041 |
| logistic regression | 0.736 | 0.646 [0.601, 0.676] | 0.685 |
| **hgb** | **0.793** | **0.707 [0.661, 0.737]** | **0.733** |

- CIs come from a match-level bootstrap (2,000 resamples of whole matches).
- hgb − logreg = +0.043 to +0.083 macro F1 (paired CI), so the difference is robust.
- Per-match macro F1 for hgb: 0.684 ± 0.076.
- Merged scheme: serve + flick_serve → serve, and net_kill + press → net_attack.

**Per class (hgb), F1:**
- strong: serve 0.95, lift 0.90, net_shot 0.88, smash 0.84, clear 0.81, flick_serve 0.80;
- weak: block 0.70, drop 0.69, drive 0.52, net_kill 0.50, push 0.49, press 0.42.

**Main confusions** (row-normalised), between strokes that look alike:
- press → drive 0.26, drive → block 0.19, push → lift 0.17, drop → smash 0.14.

**Feature groups** (hgb, 120 iterations, macro F1):
| setting | macro F1 |
|---|---|
| all features | 0.706 |
| only context | 0.554 |
| only position | 0.538 |
| only shuttle | 0.536 |
| only pose | 0.344 |
| all minus context | 0.657 |
| all minus shuttle | 0.672 |
| all minus position | 0.677 |
| all minus pose | 0.702 |

Every group adds something. Pose contributes least (−0.004); the 2D skeleton at one frame
says little beyond where the shuttle goes.

### Players: generalisation caveat
Only 9 players across 12 matches; Shi Yu Qi plays 7 matches, Antonsen 5. For most held-out
hits the hitter (and always the opponent) was in the training data. Kodai Naraoka, Kunlavut
Vitidsarn and Alex Lanier each appear in a single match, so their hits are truly unseen when
that match is held out:

| hitter in training? | hits | accuracy | macro F1 |
|---|---|---|---|
| seen | 9,772 | 0.793 | 0.704 |
| unseen | 1,529 | 0.792 | 0.713 |

- No drop for unseen players here, but this is **3 players, 1,529 hits**, all against seen
  opponents, from the same 4 tournaments and broadcast style. That is weak evidence, not proof
  of generalisation to new players.
- Per player, accuracy ranges from 0.661 (Wang Zheng Xing; his 2 matches are the China Open
  ones with the most camera problems) to 0.855 (Li Shi Feng).
- Clean rallies: macro F1 0.741. Flagged rallies (suspect / < 50 % main camera / no serve):
  0.534. The flagged set includes the 139 rallies whose first annotated hit is not a serve,
  where the "first hit" context feature misleads.

### End-to-end (classify Task 2 detections, ±2)
| | value |
|---|---|
| detections / annotated hits / matched | 10,546 / 11,301 / 8,032 |
| accuracy on matched detections (macro F1) | 0.681 (0.539) |
| annotated hits detected **and** correctly classified | 0.484 |
| detections that are a real hit of the correct class | 0.518 |

Detection errors propagate: a missed or 3-frame-off hit also corrupts the neighbours' context
features.

---

## Task 4: movement features — status: done (measurements only)

```bash
uv run python -m badminton_analysis task4   # ~5 s -> outputs/task4/*.csv, report.md, occupancy_heatmaps.png
```

### What is measured
Court metres from the ankle-based ground point (Task 1), main-camera frames only, smoothed with
a 9-frame centred moving average after filling gaps of ≤ 5 frames. 977 of 1,058 rallies are
used (suspect or < 50 % main-camera rallies excluded); 95.1 % of their frames are valid.

**CSV outputs:**
- `rally_player_movement.csv` (rally × player): distance, mean / p95 / max speed, time share in
  a 3 × 3 grid of the player's own half (front / mid / rear × left / centre / right, in the
  player's own left-right), outside fraction.
- `shot_recovery.csv` (9,722 hits followed by an opponent hit): hitter position at the hit and at
  the opponent's next contact, path length, displacement, peak speed, time available, shot type.
- `player_match_summary.csv` (match × player): totals, m per minute of play, median speeds,
  zone shares, median position at the opponent's contact ("recovery position").

Nothing is labelled a footwork "error". These are inputs for the coach-defined analysis later.

### Validation
**Noise floor.** The laws require server and receiver to stand still until the serve, so speed
measured just before the serve is measurement noise (or allowed weight shifts):

| smoothing (frames) | far player | near player |
|---|---|---|
| none | 0.166 m/s | 0.133 m/s |
| 9 (used) | 0.104 m/s | 0.083 m/s |

It plateaus from about 7 frames. n ≈ 430 pre-serve segments per side.

**Distance depends on smoothing; there is no plateau.**

| smoothing (frames) | 1 | 5 | 9 | 15 |
|---|---|---|---|---|
| total distance (km) | 50.8 | 40.3 | 36.2 | 32.6 |

- At the 9-frame setting, the noise floor accounts for ≈ 1.8 km (≈ 5 %), and larger windows also
  cut real direction changes.
- **Absolute distances are uncertain by roughly ±10–20 %**. Compare only values produced with the
  same settings.

**Far-side bias.**
- All 9 players measure faster on the far side than on the near side: median rally mean speed
  +0.02 to +0.37 m/s, typically about +0.07 (two-sided sign test p ≈ 0.004).
- This is measurement, not behaviour: there are fewer pixels per metre on the far side.
- Players change ends every game, so per-match totals mostly average it out. Rally-level
  comparisons should account for the side.

**Plausibility.**
- Speed over valid frames: p50 1.53, p95 4.19, p99 6.24 m/s; 0.65 % of frames > 7 m/s.
- Recovery positions cluster 3.6–4.1 m from the net and within ±0.2 m of the centre line.
- Occupancy maps show the expected four-corner "X" pattern.

**Known bias.** The ground point is the midpoint of both ankles, so in a lunge it sits between
the feet. Reach towards the net (little occupancy within 2 m of it) and towards the back
corners is understated.

### Per-match values
| | range |
|---|---|
| distance per player | 0.94–2.20 km |
| distance per minute of rally play | 101–124 m/min |
| median rally mean speed | 1.63–1.99 m/s |

Full table: `outputs/task4/report.md`.

---

## Task 5: TrackNetV3 on raw video — status: done (local run + Colab notebook)

```bash
uv run python -m badminton_analysis task5-clip --match Wang-Zheng-Xing-CHN-F --video <match.mp4> --out clip.mp4
# TrackNetV3 (see the settings below) -> clip_ball.csv
uv run python -m badminton_analysis task5-eval --match Wang-Zheng-Xing-CHN-F --pred clip_ball.csv \
    --clip-start 14339 --video <match.mp4>          # -> outputs/task5/{report.md, metrics.json, overlay_*.jpg}
```

**Caveat:** BFMD's shuttle track is itself TrackNet output (unspecified version), so this
measures **agreement between two trackers, not accuracy**. Where they disagree, either can be wrong.

### Result: China Open 2025 F, 50 s clip (video frames 14339–15847, 3 rallies)
| | value |
|---|---|
| clip frames / inside annotated rallies | 1,508 / 630 |
| both see the shuttle / only BFMD / only TrackNetV3 | 556 / 52 / 22 |
| visibility agreement | 0.883 |
| position difference where both see it: median / mean / p90 | **2.2 px** / 39.0 / 101.7 |
| within 2 / 5 / 10 / 20 px | 0.437 / **0.730** / 0.795 / 0.842 |
| frame-alignment check: median px at BFMD shift −1 / 0 / +1 | 10.5 / **2.2** / 11.1 |

- **Frame alignment is exact:** shifting BFMD by one frame either way multiplies the median by ~5.
- **Two populations:** most frames agree within a few pixels (the median of 2.2 px is about one
  heatmap cell of TrackNetV3, which works at 512 × 288 and truncates to integers); the 16 % beyond
  20 px are runs of tens of frames where one tracker follows something else, hence the mean of 39 px.
- **What the disagreement runs are** (frames checked by eye on overlays):
  - 15448 (28 frames): a **close-up camera shot at the serve** inside the annotated rally window.
    Neither tracker can be right (the shuttle is in the server's hand). BFMD's per-frame court
    flag still says main camera here, so that flag misses some camera cuts.
  - 14929 (17 frames): BFMD at the top edge of the image (a high lift leaving the frame),
    TrackNetV3 on the far player's shoe.
  - 15676 (6 frames): TrackNetV3 on a white streak at the near player's racket (the shuttle);
    **BFMD on the far player's white shoes**.
  - 14825–14828: BFMD on the far player's head, TrackNetV3 on his chest; no shuttle clearly visible.
- So the failures go both ways; white shoes and other small white objects fool both trackers.

### Found on the way: BFMD shuttle coordinates were read 5.12 px off (fixed)
BFMD's README says shuttle centre = `(x + width/2, y + height/2)`. That is wrong for this data:
- every shuttle box in all 12 matches is the same dummy size (0.8 % × 1.4222 % = 10.24 × 10.24 px);
- on the video, the stored **(x, y) corner sits on the shuttle** in 11 of 12 evenly spaced
  frames checked at 10× zoom, and the README "centre" lies 5.12 px down-right of it;
- TrackNetV3 (which outputs the blob centre) agrees with (x, y): before the fix the median offset
  was exactly −6.12 px in both axes with scale 1.000 (5.12 + ~1 px of integer truncation).

`raw.load_shuttle` now uses (x, y) (schema version 5). Effect on Milestone 1 after a full re-run
(Tasks 2–3 numbers in this report are the corrected ones): hit F1 @±2 0.7367 → **0.7362**, shot
macro F1 0.7096 → **0.7062**, Task 4 unchanged. A constant offset doesn't change shuttle speed or
turning, only the wrist–shuttle distance features, so the change is small and inside the CIs.

**TrackNetV3 facts (verified from the code):**
| item | value |
|---|---|
| repo | qaz812345/TrackNetV3, commit `6eda442` |
| licence | MIT, explicitly covering the checkpoints |
| checkpoint | `TrackNetV3_ckpts.zip`, Google Drive ID `1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA`, 126 MB, SHA-256 `8de68d34ea2457e368e22947f1309143d60a6a7f9ddccd31cd526d4c4571bcc4` |
| zip contents | `ckpts/TrackNet_best.pt` (136 MB), `ckpts/InpaintNet_best.pt` (6 MB) |
| output | `<clip>_ball.csv` with columns `Frame, Visibility, X, Y` |

**Changes needed for current PyTorch / Python** (found by running it with torch 2.14 on the
local GPU):
1. `requirements.txt` pins torch 1.10, numpy 1.22 and opencv 4.4, which don't install on current
   Python. Use the current libraries instead; inference imports fine with them.
2. `test.py` (imported by `predict.py`) needs **`pycocotools`**, which is missing from
   `requirements.txt`; `parse` is needed too.
3. Use `--large_video`: without it, `predict.py` loads the whole clip into RAM (≈ 4 GB for 50 s
   of 720p).
4. Use **`--max_sample_num 300`**: the default of 1,800 frames for the background median was
   **killed for running out of memory (exit 137 at 5.8 GB)** on this machine.
5. `torch.load` works unchanged on torch 2.14 (default `weights_only=True`): the checkpoints load
   and inference runs; no patch needed.
6. Speed: 8 min 34 s for the 1,508-frame clip on an RTX 3050 laptop GPU (≈ 3 frames/s); InpaintNet
   takes under a second. The bottleneck is CPU frame loading: `predict.py` starts `batch_size`
   (16) DataLoader workers regardless of the CPU count (4 here, warning printed).

Full command used:
```bash
uv run --project <repo> --with parse --with pycocotools python predict.py --video_file clip.mp4 \
    --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt \
    --save_dir pred_out --large_video --max_sample_num 300
```

### Colab notebook: `notebooks/colab_task5_tracknetv3.ipynb` (T4 GPU)
Runs the steps above end to end: clones this repo and TrackNetV3 at `6eda442`, copies only this
match's BFMD files from Drive, downloads the checkpoints once with `gdown` (cached in
`My Drive/badminton/task5/`, SHA-256 checked), cuts the clip, runs `predict.py` with the flags
above, runs `task5-eval` (results in `My Drive/badminton/outputs/task5/`), and checks the metrics
against the local run with tolerances.

**Tested** by executing every cell with `nbclient` against a copy of the Drive layout (fresh
GitHub clone, real checkpoint download, local GPU): 615 s in total, clip identical (start
14339, 1,508 frames), predictions **byte-identical** to the local run, so all metrics in the
table above are reproduced exactly. Not yet run on a Colab T4 itself; there a different PyTorch /
OpenCV build may move a few frames, which the tolerance check reports.

## Hit label noise — status: tool ready, human annotation pending

Goal: measure how far BFMD's hit frames are from the true racket–shuttle contact, to tell how
much of Task 2's ±2 gap (F1 0.736 @±2 vs 0.824 @±5) is label noise. Context: the CoachAI
Badminton Challenge 2023 also scores hit frames at ±2 frames (30 fps, 720p); ShuttleSet re-labelled
3 matches for consistency but publishes no agreement number.

```bash
uv run python -m badminton_analysis labelnoise-kit --match "China-Open-2025-Shi-Yu-Qi-CHN-3-vs.-Wang" \
    --video <match.mp4> --out <kit>          # ~150 s: sequential decode, 5,500 images
# open <kit>/index.html, label every item, "Download results"
uv run python -m badminton_analysis labelnoise-eval --kit <kit> --annotations labelnoise_results.json
    # -> outputs/labelnoise/{report.md, metrics.json, items.csv}
```

**Design.** 100 hits from China Open 2025 F (the match with video), stratified by shot type
(≥ 2 per type), plus 10 repeats for annotator repeatability; seed 0. Each item is 25 frames around
the BFMD frame, shifted by a random −4..+4, so the label is not at a fixed position; the page
never reads the key. Crops are centred on the hitter's median box over the window (no leak of the
label frame); items whose hit frame is not on the main camera (court_perframe flag) open in full
view. The evaluation reports human − BFMD offsets (mean, |offset|, within 0/±1/±2/±3/±5, bootstrap
CIs, per shot type, main-camera only, sure only), repeatability, and the Task 2 detector's error
against both references.

**Checks done.** Video frame index = decoder timestamp index for all 148,950 frames (constant
30 fps, no gaps), so sequential decoding matches the labels' frame numbers. Pilot of 6 items:
a first crop design was too loose (whole court) and was replaced; one sampled serve (g2 r12) falls
in a broadcast **close-up**, and one rally (g2 r15) is a side camera that `court_perframe` marks
valid, as in Task 1 — the page has "no hit" and full-frame options for these.

**Not done: the annotation itself.** An attempt to label the kit with the model (Claude, viewing
blinded frame grids and full-resolution zooms) was stopped after 7 items: none could be labelled
with confidence (contact hidden by blur or by the other player, tiny players on the side camera,
contact possibly before the window). No offset numbers are reported until a human labels the kit.

## Milestone 1: how to reproduce

```bash
uv sync
uv run python -m badminton_analysis all     # build, validate, viz-samples, task2, task3, task4 (~8 min, CPU)
uv run pytest                               # 68 tests
```

- **Colab:** `notebooks/colab_milestone1.ipynb` with `configs/colab.yaml` (same settings,
  different paths).
- **Clean-checkout reproducibility:** running only the committed files with a fresh cache gave
  identical numbers (hit F1 0.736737, shot macro F1 0.709614), before the shuttle-coordinate fix
  (Task 5). After the fix, a full re-run gives hit F1 0.736158, shot macro F1 0.706158.

## Overall limitations
- **Data:** 12 matches, 9 players and 4 tournaments, all from BWF broadcasts. Every signal comes
  from annotations (TrackNet shuttle, BFMD pose, a manual timeline), so annotation errors cap all
  four tasks.
- **Video check:** done for one match.
- **Licences:** the BFMD dataset licence and the BWF video terms are still unverified (AUDIT.md).
