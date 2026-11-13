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
| + wrist proximity | 0.641 | 0.767 / 0.682 / **0.722** | 0.764 | 0.807 | 0.720 ± 0.060 | 0.664 / 0.743 |
| + wrist speed (`full`) | 0.641 | 0.770 / 0.680 / **0.723** | 0.764 | 0.807 | 0.720 ± 0.060 | 0.666 / 0.743 |
| **logistic** | 0.651 | 0.759 / 0.716 / **0.737** | 0.783 | 0.825 | 0.735 ± 0.061 | 0.689 / 0.754 |
| pose only (reference) | 0.178 | 0.232 / 0.376 / **0.287** | 0.395 | 0.535 | 0.288 ± 0.013 | 0.283 / 0.289 |

- Pose adds +3.2 F1 points over the shuttle alone (mostly precision, via wrist proximity);
  wrist *speed* adds nothing on top of proximity. The shuttle is essential (pose only: 0.29).
- No sign of design overfitting: the 9 never-inspected matches score higher than the dev matches.
- Large between-tournament differences: Malaysia Open 0.82–0.85, Indonesia Open 0.67–0.71.
- Logistic weights (standardized): wrist proximity −1.57, shuttle speed +0.95, missing pose
  +0.92, turn angle +0.64, wrist speed +0.53, log accel −0.19 (stable across folds, std ≤ 0.03).

### Failure analysis (logistic, ±2)
- **Serves are nearly undetected:** recall 0.126 (serve) / 0.082 (flick serve); the shuttle
  starts from rest and is visible on only 59 % of serve frames (dev matches). Recall excluding
  serves: **0.768**. 875 of 3,212 misses are the first hit of a rally.
- **Timing:** 1,255 misses have a prediction 3–8 frames away. On the dev matches, every way of
  picking the frame (acceleration peak, wrist–shuttle closest approach, maximum turn, 1-frame
  acceleration) puts only 70–72 % of near-hit candidates within ±2 of the label. A video check
  of 3 disagreements found the label at contact in 2 (detector late on a smash, early on a drop)
  and the label ~4 frames late in 1 (net shot). Both sources err; with n = 3 this is anecdotal.
- Other misses: 1,254 candidates scored below threshold (weak net shots / blocks: recall 0.71 /
  0.63), 432 with the shuttle missing, 271 with no trajectory peak at all.
- False positives: 1,885 mid-rally (largely the timing pairs above), 298 after the last hit
  (shuttle landing), 190 duplicates, 152 at annotated bounces / net hits; 359 have a removed
  shuttle outlier within ±3 frames.
- Hitter side (nearest wrist) is correct for 81.0 % of detected hits.
- Clean rallies F1 0.747 vs flagged rallies (suspect / < 50 % main camera / no serve) 0.688.

### Limitations / next steps for hits
- Every signal is derived from annotations (TrackNet shuttle, BFMD pose); errors there cap the
  baseline. Rally intervals come from the annotations too, so this is "hits within known rallies".
- A small manual re-annotation of contact frames from the video (e.g. 100 hits) would measure
  label timing noise and tell how much of the ±2 gap is label noise.
- Serve detection needs a dedicated cue (e.g. server pose before the first shuttle motion).

### Next step
Task 3: shot-type baseline (small transparent model on skeleton + trajectory features),
leave-one-match-out, with per-player results.
