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

### Limitations
- All alignment is **relative** between annotation sources; absolute alignment to decoded video
  frames, and the cause of the suspect rallies, need at least one video.
- The ground-point offset is one global constant; the static homography is only valid for the
  main camera.

### Next step
Task 2 (hit-event baseline), with the agreed ±2-frame primary tolerance (also ±1/±3/±5).
