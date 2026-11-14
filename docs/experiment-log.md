# Experiment log

Every parameter of this project was chosen by a rule written down before the
run that tested it. This page lists those rules, the runs and the decisions, in
the order they were made, so that each "chosen by the pre-registered rule"
comment in `configs/`, `scripts/` and `src/` can be checked. The current
numbers, with their confidence intervals, are in [results.md](results.md); this
page is the history behind them, rejected hypotheses included.

The work ran in two plans. The [floor-heatmap plan](#floor-heatmap-plan) built
the pipeline and its evaluation, phases 1 to 10. The
[improvements plan](#improvements-plan) is the follow-up that started once the
floor-heatmap plan closed, phases 1 to 8.

## How the experiments were run

- **Data and splits.** NVIDIA MTMC_Tracking_2024, test scenes 071 to 080.
  Every choice was made on the *tune* scenes (071 to 075). The *sealed* scenes
  (076 to 080) were scored once per accepted change, with nothing changed after
  the tune decision ([results.md, Terms](results.md#terms)). WILDTRACK, CAVIAR
  and MOT17 are evaluation only, never tuned on.
- **One change per hypothesis.** Each hypothesis changes one thing against the
  config it competes with, and gets its own config in `configs/` or its own
  sweep script in `scripts/`.
- **Rule first.** The metric, the target, the guards and the selection rule
  were written before the first run of each hypothesis. Where a rule was
  changed after a run, or a run deviated from it, this page says so.
- **Default acceptance.** Paired block bootstrap (30 s blocks, 1000 resamples,
  95 %, recentred percentile; seed 0 from the improvements plan on). A change
  is accepted when the CI of its paired Δ excludes 0, |Δ| ≥ .01 and no guard
  breaks. Phases with their own rule state it.
- **Tracking guard.** An accepted step may not lower HOTA or IDF1 by more than
  1 point against the config it replaced
  ([results.md, Terms](results.md#terms)).
- **Metrics.** TVD, mass error, zone WAPE, EMD, HOTA and IDF1 are defined in
  [results.md, Terms](results.md#terms). mc-IDF1 is defined in
  [phase 10](#phase-10-e2b-geometric-cross-camera-ids).
- **Truth blocks.** A metric is read on a truth and a mask, written
  `truth@mask`:
  - *visible*: the people the camera has a visible box of; *full*: every
    person, hidden or not.
  - *coverage*: the floor cells the camera sees; *joint*: the cells at least
    one fused camera sees; *anchor*: the E1 camera's coverage, to compare a
    fused map with one camera; *overlap*: the cells two or more cameras see.
  - E1 is judged on visible@coverage. The fused map (E2a) is judged on
    full@joint.
- **Numbers.** Values are pooled over the five scenes of a split (unweighted
  mean), at 0.5 m cells, dwell layer, unless stated. Δ is variant minus
  baseline; for error metrics a negative Δ is better. Brackets are the 95 % CI.
  "n.s." means the CI includes 0.
- **Links.** An accepted change links to the pull request that merged it.
  A rejected hypothesis links to the pull request in which it was measured,
  or, when it was never merged, to its branch, which is kept as it was.

## Summary

| Plan, phase | Hypothesis | Decision | Link |
|---|---|---|---|
| Floor heatmap, 1 | Evaluation, oracle runs | evaluation valid | [5ef57a5][c-phase1] |
| Floor heatmap, 2 | Iteration 0: the Ultralytics reference recipe | baseline | [#2][pr2] |
| Floor heatmap, 3a | H4 foot point (auto-track) | accepted | [#4][pr4] |
| Floor heatmap, 3b | H1 YOLOX-s, own pipeline | accepted | [#6][pr6] |
| Floor heatmap, 3b | H2a input 960 | accepted | [#8][pr8] |
| Floor heatmap, 3b | H2 input 1280 (CUDA), H1b D-FINE-S, H1c RF-DETR-N | rejected | [#8][pr8] |
| Floor heatmap, 4 | H3a OC-SORT, H3b floor association (two settings each) | rejected | [#10][pr10] |
| Floor heatmap, 5 | H6a-f thresholds, H7 stride 2, H2a′ 1280 on TensorRT | rejected | [#12][pr12] |
| Floor heatmap, 5 | H5 TensorRT fp16 | accepted (speed) | [#12][pr12] |
| Floor heatmap, 6, 7 | Persistent slots, privacy, live viewer | criteria met | [#14][pr14], [#16][pr16] |
| Floor heatmap, 8 | E1 on sealed, external datasets, label-free metrics | E1 not accepted; every step holds | [#19][pr19], [#20][pr20], [#22][pr22] |
| Floor heatmap, 9 | E2a fusion F1-F5 | F5 accepted | [#24][pr24] |
| Floor heatmap, 10 | E2b association A1-A4 | A4 accepted; NVIDIA target missed | [#26][pr26] |
| Improvements, 1 | README demo GIF | done | [#29][pr29] |
| Improvements, 2 | Tracklet stitching S1-S6 | S6 accepted; target missed | [#30][pr30] |
| Improvements, 3 | Detection stride R4-R16, cameras per GPU | stride 10, 8 cameras | [#32][pr32] |
| Improvements, 4 | Fusion weights W1-W3 | W1 accepted | [#34][pr34] |
| Improvements, 5 | Multi-camera recording P1-P4 | criteria met | [#38][pr38] |
| Improvements, 6 | H1 partial covered-cell normalisation | rejected | [branch][b36] |
| Improvements, 6 | H2 chord-length passage | rejected | [branch][b37] |
| Improvements, 6 | H3 k counted on stitched chains | rejected | [branch][b39] |
| Improvements, 6 | H4 occupancy reading in the viewer | not accepted as registered; merged as a defect fix | [#41][pr41] |
| Improvements, 6 | H5 outlier mass gating | not run (stop rule) | – |
| Improvements, 7 | H7 single-camera recall factor | rejected; reading labelled approximate | [#44][pr44] |
| Improvements, 7 | H6 k on coarser units | rejected; measured rates stated | [#45][pr45] |
| Improvements, 7 | H8 higher id threshold | rejected; trade-off stated | [#45][pr45] |
| Improvements, 8 | Opening the repository | not an experiment | [#47][pr47] |

## Floor-heatmap plan

Goal: floor-plan heatmaps from fixed oblique cameras on a 4 GB GPU, measured
against ground truth and improved over the Ultralytics reference recipe with
evidence (E1), then multi-camera fusion (E2a) and geometric cross-camera ids
(E2b). The hypotheses H1 to H7 were listed in this order before iteration 0:
detector, input resolution, tracker, foot point of a truncated box, TensorRT
fp16, detection and tracker thresholds, frame stride with interpolation.

### Phase 1: evaluation (oracle)

Question: does the evaluation score a perfect input as perfect, and how good is
the box-to-floor geometry alone?

Rule: ground-truth floor positions fed through the evaluation must give exactly
0 on every metric; ground-truth boxes, projected from their bottom centre with
the camera homography, give the geometry ceiling of the approach.

Result:
- Ground-truth positions, scene 071, all 15 calibrated cameras: mass error,
  TVD, zone WAPE and EMD all exactly 0 (visible@visibility).
- Ground-truth boxes, camera 645, visible@coverage: TVD .299, zone WAPE .166,
  EMD 1.03 m. On the five tune cameras (071→645, 072→663, 073→677, 074→695,
  075→709): TVD .299 / .334 / .227 / .235 / .438, mean .307.
- Against every person in coverage (full@coverage), camera 645 reaches TVD
  .517 with 41.6 % of the people occluded.

Decision: evaluation valid. E1 is measured on visible@coverage (what one camera
can observe); full@coverage becomes the E2a target. Since the box ceiling
already sat above the initial TVD target, the foot point (H4) was moved ahead
of the tracker hypotheses. Link: [5ef57a5][c-phase1].

### Iteration 0 (phase 2): baseline

Question: where does the Ultralytics reference recipe start?

Setup: YOLOv8n COCO, its built-in ByteTrack with defaults (conf 0.1, input
640), box centre projected to the floor (`configs/iter0.yaml`), tune split.

Result (visible@coverage): mass error .273 [.256, .290], TVD .629 [.611, .648],
zone WAPE .614 [.574, .654], EMD 2.89 m [2.74, 3.04]; per scene TVD 071 .638,
072 .629, 073 .613, 074 .535, 075 .731; full@coverage TVD .672. Speed 86.8 fps
(p5 74.3) over 30 min, 108 MB VRAM.

Decision: baseline fixed. Link: [#2][pr2].

### Target recalibration

The initial targets could be recalibrated once, after iteration 0, and the
change recorded. Done once, final (visible@coverage, tune, 0.5 m, dwell):

| Metric | Initial | Iteration 0 | Ceiling (ground-truth boxes) | Recalibrated | Why |
|---|---|---|---|---|---|
| TVD | ≤ 0.15 | .629 | .307 | ≤ 0.30, and below iteration 0 with paired CI | 0.15 is below what perfect boxes reach with the naive foot point |
| Mass error | ≤ 10 % | .273 | about 0.0 to .07 | ≤ 10 % (unchanged) | reachable |
| Zone WAPE | ≤ 15 % | .614 | about .17 (camera 645) | ≤ 30 % | 15 % is below the single-camera ceiling on 2 m zones |
| EMD | – | 2.89 m | about 1.0 m (camera 645) | ≤ 1.2 m | new: mass within about two cells of the truth |
| fps, 30 min, one 1080p camera | ≥ 15 | 86.8 | – | ≥ 15 (unchanged) | |
| VRAM peak | ≤ 3.5 GB | 108 MB | – | ≤ 3.5 GB (unchanged) | |
| WILDTRACK median ground error | ≤ 30 cm | – | – | ≤ 30 cm (unchanged) | |

E1 acceptance on sealed: the same targets, with sealed within 20 % relative of
tune (unchanged).

### Phase 3a: foot point (H4)

Question: when shelves hide the legs, does a floor point estimated from the
head beat the bottom of the box?

Rule: measured first on ground-truth boxes against the oracle ceiling, then on
the iteration 0 tracks with the default acceptance against iteration 0.

Variants on ground-truth boxes (five tune cameras, TVD): bottom .307, head .454,
auto (implied height < 1.40 m → head) .251, auto-track (causal per-track p90
height, a box below 0.88 of it is hidden) .215. Auto-track median position
error .15 to .17 m, p90 .27 to .87 m (bottom p90 up to 6.8 m).

Result, iteration 0 tracks with auto-track (`configs/h4-auto-track.yaml`):
TVD .381, Δ vs iteration 0 −.248 [−.264, −.232]; Δ vs bottom −.018
[−.021, −.014] (per scene −.019, −.028, −.003 n.s., −.033, −.005). Most of the
gain is box centre → bottom.

Decision: **accepted** (auto-track). Rejected on the way: head alone (worse,
ill-conditioned where the horizon is inside the image), auto with a fixed
1.40 m threshold (sends children to the head rule, 073 +.10), and a whole-track
height (reads future frames, so it cannot run live; −.019 against the causal
−.018). Link: [#4][pr4].

### Phase 3b: detector and input resolution (H1, H2)

Question: does a stronger detector with a permissive licence, then a larger
input, lower TVD? Default acceptance, each against the config before it.

| Hypothesis | Config | Mass error | TVD | Δ TVD | Decision |
|---|---|---|---|---|---|
| H1 YOLOX-s 640 + ByteTrack + auto-track, own ONNX pipeline | `h1-yolox-s.yaml` | .252 | .355 | −.026 [−.034, −.018] vs H4 | **accepted** ([#6][pr6]) |
| H2a input 960 | `h2-yolox-s-960.yaml` = `h1-best.yaml` | .148 | .308 | −.047 [−.056, −.038] vs H1 | **accepted** ([#8][pr8]) |
| H2 input 1280 (CUDA) | `h2-yolox-s-1280.yaml` | – | – | – | rejected without evaluation: 13.3 fps breaks the ≥ 15 fps guard ([#8][pr8]) |
| H1b D-FINE-S | `h1-dfine-s.yaml` | .064 | .344 | +.035 [.027, .043] vs 960 | rejected ([#8][pr8]) |
| H1c RF-DETR-N | `h1-rfdetr-n.yaml` | .120 | .349 | +.041 [.030, .051] vs 960 | rejected ([#8][pr8]) |

H1 per scene TVD: 071 .355, 072 .339, 073 .355, 074 .241 (+.019 worse), 075
.485. H2a per scene: 071 .295, 072 .305, 073 .283, 074 .226, 075 .434; 21.1 fps
(p5 20.6). D-FINE-S has the lowest mass error but loses on TVD, mostly on 075
(.553).

### Phase 4: tracker (H3)

Question: does another tracker lower TVD on the same cached detections?
Default acceptance against `h1-best.yaml` (TVD .308, HOTA .236, IDF1 .182).

| Hypothesis | Config | TVD | Δ TVD | Mass error | HOTA / IDF1 | Decision |
|---|---|---|---|---|---|---|
| H3a OC-SORT, published defaults (detection threshold 0.6) | `h3-ocsort.yaml` | .337 | +.028 [+.022, +.034] | .243 | .232 / .182 | rejected |
| H3b floor-plane association after UCMCTrack, defaults | `h3-ucmc.yaml` | .337 | +.028 [+.023, +.033] | .208 | .226 / .173 (HOTA −1.1 pt, guard broken) | rejected |
| H3a-2 OC-SORT, detection threshold 0.1 | `h3-ocsort-lowthr.yaml` | .319 | +.011 [+.007, +.015] | .076 | .234 / .177 | rejected |
| H3b-2 floor association, low thresholds (MOT17-01 settings) | `h3-ucmc-lowthr.yaml` | .329 | +.020 [+.016, +.025] | .156 | .239 / .189 | rejected |

The two "-2" variants are fairness checks of the first two. Both trackers'
defaults drop the boxes scored below 0.5 to 0.6, which ByteTrack's second stage
keeps down to 0.1; lowering OC-SORT's threshold closes most of its gap, but not
all. Decision: ByteTrack stays. Link: [#10][pr10].

### Phase 5: thresholds, engine and stride (H6, H5, H2a′, H7)

**H6, registered before any run.** Base `h1-best.yaml`. One change per variant;
the best is chosen by paired Δ TVD on tune and confirmed on sealed in phase 8.
H6g (best detector variant with best tracker variant) only if both are accepted
alone.

| Variant | Config | Δ TVD | Mass error Δ | Decision |
|---|---|---|---|---|
| H6a ByteTrack activation 0.7 → 0.6 | `h6a-activation.yaml` | +.001 [−.001, +.004] n.s. | −.029 [−.034, −.024] | rejected |
| H6b high and activation 0.5 | `h6b-high.yaml` | +.001 [−.002, +.003] n.s. | −.045 [−.051, −.038] | rejected |
| H6c lost buffer 30 → 60 frames | `h6c-buffer.yaml` | −.000 [−.001, .000] | – | rejected (no effect) |
| H6d detector score 0.1 → 0.05 | `h6d-score-low.yaml` | +.002 [−.001, +.006] n.s. | −.014 [−.019, −.009] | rejected |
| H6e detector score 0.1 → 0.2 | `h6e-score-high.yaml` | +.002 [−.000, +.005] n.s. | +.022 [.017, .028] | rejected |
| H6f NMS IoU 0.45 → 0.6 | `h6f-nms.yaml` | +.001 [−.002, +.004] n.s. | – | rejected |

H6g was not run: no variant was accepted alone. Lower thresholds cut mass error
without changing the shape of the map.

**H5, H2a′ and H7, registered before any run.** H5 must match the CUDA engine
per box and on tune metrics within CI; it is an engine change, so the TVD rule
does not apply. H2a′ runs only if TensorRT gives the fps headroom. H7 processes
one frame in two and fills the skipped frame.

| Hypothesis | Config | TVD | Δ TVD vs h1-best | Other | Decision |
|---|---|---|---|---|---|
| H5 TensorRT fp16 | `h5-trt.yaml` = `h5-best.yaml` | .308 | −.001 [−.002, .000] | 1701 of 1701 boxes matched against the CUDA engine on 200 frames; 21.1 → 42.4 fps (p5 39.9, 30 min), 174 MB | **accepted** (speed) |
| H2a′ YOLOX-s 1280 on TensorRT | `h2a2-yolox-s-1280-trt.yaml` | .300 | −.009 [−.017, +.001] | 25.0 fps; HOTA +1.2 pt | rejected (CI includes 0, below .01) |
| H7 stride 2 | `h7-stride2.yaml` | .310 | +.001 [−.002, +.004] n.s. | mass error +.006 [.003, .010] | rejected as an accuracy change; kept as a speed lever |

Decision: `configs/h5-best.yaml` (h1-best on TensorRT fp16). On tune it meets
fps, VRAM and zone WAPE (.231), and misses mass error (.154), EMD (1.50 m) and
TVD (.308 against ≤ .30). Link: [#12][pr12].

### Phase 6: persistent slots and privacy

Product phase, judged on an observable criterion, not a metric: run on scene
071 into SQLite, the schema holds only the `slots` table (aggregated grids),
cells with fewer than 5 distinct track ids read as empty, and no image file is
written. Observed on 071/645: 2 slots, only the slots table, no cell under k, no
image written. Link: [#14][pr14].

### Phase 7: live mode and viewer

Product phase. Criterion: with a local RTSP stream of scene 071, the viewer at
desktop width updates each second (screenshots 3 s apart differ, no console
errors), and a past slot shows its stored map. Met. The live layer also applies
k=5, over the last 120 s. Link: [#16][pr16].

### Phase 8: E1 on sealed and external datasets

Rule: each step accepted on tune is scored once on sealed with nothing changed;
its Δ must keep its sign with a CI excluding 0 (H5: no accuracy loss), or it is
reverted. E1 is accepted if sealed meets the recalibrated targets within 20 %
of tune.

Result: every accepted step holds on sealed (H4 −.191, H1 −.023, H2a −.038,
H5 −.001), and no tracking guard breaks; sealed TVD is .419 against .308 on
tune (+36 %). The full table is in
[results.md, E1, one camera](results.md#e1-one-camera).

Decision: **E1 not accepted**: sealed misses the targets and the 20 % rule. No
step is reverted. The gap is mostly occlusion, carried to phase 9.

External datasets, evaluation only: WILDTRACK camera C5 first measured a median
ground error of 29.1 cm (image pairing), p90 108.8 m; C5 sits at 1.68 m, below
the head plane of the auto-track rule. A camera-height guard (the head rule
applies only to cameras at least 3.0 m high; the margin comes from an error
bound and is not fitted) brought it to median 21.8 cm, p90 45.1 cm, without
changing the floor points of the 10 NVIDIA cameras (4.02 to 4.88 m high). CAVIAR
and MOT17 values are in [results.md](results.md#external-datasets), and the
label-free metrics in [results.md](results.md#label-free-metrics). Links: [#19][pr19], [#20][pr20],
[#22][pr22].

### Phase 9: E2a multi-camera fusion

Rule, registered before any fused run: fusion is a per-cell convex combination
of the covering cameras. Variants F1 to F5 are tuned on 071-075 only and chosen
by the lowest pooled full@joint TVD. Acceptance needs Δ TVD against the single
E1 camera on full@anchor and on full@joint, CI excluding 0 and |Δ| ≥ .01, then
confirmed on sealed with no change. Every calibrated camera, stride 2.

| Variant | Mass error full@joint | TVD full@joint (anchor) | EMD m | Decision |
|---|---|---|---|---|
| F1 mean of covering cameras | .402 | .313 | – | rejected (over-counts in overlaps) |
| F2 visibility weight, power 1 | .056 | .270 (.260) | 1.29 | not chosen; overlap mass error .086 |
| F3 power 2 | .101 | .275 (.265) | 1.38 | rejected; overlap .136 |
| F4 power ∞ (camera that sees most) | .153 | .276 (.267) | 1.46 | rejected; overlap .194 |
| F5 power 1, visibility on 2 m zones (`configs/e2a-fusion.yaml`) | .081 | .260 (.252) | 0.88 | **chosen** |

Result, F5 against the single camera:
- Tune: full@joint TVD .260 [.250, .270], Δ −.487 [−.496, −.477]; full@anchor
  .450 → .252, Δ −.198 [−.208, −.187]; overlap mass error .069 [.062, .075];
  WAPE .167.
- Sealed: full@joint .274 [.264, .286], Δ −.471 [−.480, −.461]; full@anchor
  .557 → .273, Δ −.285 [−.294, −.274]; overlap .087 [.081, .093]; WAPE .284.
- Above 10 % overlap mass error per scene: 073 (.119), 079 (.102), 080 (.149).
  Scene 071 EMD 1.44 m misses 1.2. Passage mass error stays about .6 (tune
  .578, sealed .647).

Decision: **accepted**. The phase was later regenerated from the frozen
detection cache, with every recorded value reproduced exactly. That rerun also
scored a variant outside F1-F5, power 2 on 2 m zones (TVD .259, overlap .039);
it was not selectable, and became W1 in
[improvements phase 4](#improvements-plan-phase-4-fusion-weights). Link:
[#24][pr24].

### Phase 10: E2b geometric cross-camera ids

**Metric.** mc-IDF1: TrackEval's identity F1 on the floor, with floor distance
in place of box IoU. A truth position and a prediction of the same frame match
within 1.0 m (fixed, never tuned). Truth: the people at least one fused camera
has a visible box of, on the joint coverage. Blocks of 30 s on NVIDIA, 10 s on
WILDTRACK.

**Rule, registered before any association variant ran on real data.**
Baseline: no association (per-camera ids), pooled .114 [.095, .133]. Reference:
oracle association of the same tracks, .795 [.787, .805]. Target: pooled
mc-IDF1 ≥ baseline + ½ (oracle − baseline) = .455, written as ≥ .45; on
WILDTRACK the same derivation from its own baseline and oracle. Acceptance:
default rule against the baseline; guards: per-camera tracks and fused dwell
unchanged. Selection: highest pooled mc-IDF1 among the accepted variants, all
geometric, no appearance.

| Variant | mc-IDF1 | Δ vs baseline | Decision |
|---|---|---|---|
| A1 co-location, 0.5 m gate | .162 [.141, .183] | +.047 [+.041, +.054] | accepted, not best |
| A2 co-location, 1.0 m gate | .173 [.152, .195] | +.059 [+.052, +.066] | accepted, best of A1-A3 |
| A3 mutual-nearest co-location, 1.0 m | .170 [.150, .192] | +.056 [+.049, +.063] | accepted, not best |
| A4 A2 + 2 s handover (`configs/e2b-association.yaml`) | .175 [.154, .196] | +.061 [+.053, +.069] | **chosen** |

Confirmation, A4 unchanged: sealed .157 [.137, .179] against .094, Δ +.063
[+.054, +.074]; WILDTRACK (C1-C7) .375 [.355, .395] against .284, Δ +.091
[+.076, +.107], target ≥ .337 met. Dwell Δ exactly 0 on every split.

Decision: **accepted** against per-camera ids; the NVIDIA target (.45) is
missed on tune and sealed. On 071, the links co-location and a 2 s handover can
see cap identity at .22 to .26 even when perfect. Counting passage with global
ids raised its mass error (tune .578 → 1.034, sealed .647 → 1.501), so the
fused map (E2a) stays the passage layer. Link: [#26][pr26].

## Improvements plan

Follow-up work in an order fixed by value per effort. Iteration rules are those
of the floor-heatmap plan. The detection cache is the frozen input; everything
downstream is deterministic (seed 0).

Dropped before any run:
- Reproducibility tooling (input checksums, a repro-check script): the
  floor-heatmap phase 9 rerun already showed determinism, with 13 of 13 outputs
  byte-identical.
- E2b long-gap association: the heatmap does not need it, and its geometric
  ceiling (mc-IDF1 .408 on 071) is below the .455 target of
  [phase 10](#phase-10-e2b-geometric-cross-camera-ids). Its side effect on
  mc-IDF1 is reported in phase 2 below.
- E1 single-camera occlusion: fusion covers what one camera cannot see.

### Improvements plan, phase 1: demo GIF

Not an experiment. A 12 s, two-panel GIF of scene 071 camera 645 (foot points
only, no boxes or ids; k=5 on the map panel), from a window picked by the number
of people visible, without looking at the metrics. The dataset licence (CC BY
4.0) was checked first. Link: [#29][pr29].

### Improvements plan, phase 2: tracklet stitching

Question: fused passage over-counts because one camera splits a person into
many tracks (707 ids for 26 people on the 071 E1 camera). Does joining one
camera's tracklets across short gaps, on the floor and without appearance,
lower it?

Rule, registered before any run:
- Baseline `configs/e2a-fusion.yaml`: passage mass error full@joint .578
  [.521, .636], dwell TVD .260.
- Variants: gap G ∈ {2, 5, 10} s × radius R ∈ {0.5, 1.0} m, v_max 1.5 m/s.
- Selection: lowest pooled fused passage mass error full@joint among the
  variants that pass the guards. Target ≤ .30.
- Acceptance: default rule on the paired Δ.
- Guards: E1 camera HOTA/IDF1 drop at most 1 point; fused dwell byte-identical.

| Variant | Passage mass error | Δ | E1 ΔHOTA / ΔIDF1 (points) | Decision |
|---|---|---|---|---|
| S1 G2 R0.5 | .561 | −.017 | +1.10 / +1.45 | guards pass |
| S2 G2 R1.0 | .556 | −.022 | +1.20 / +1.61 | guards pass |
| S3 G5 R0.5 | .530 | −.048 | +1.42 / +2.00 | guards pass |
| S4 G5 R1.0 | .520 | −.057 | +1.50 / +2.11 | guards pass |
| S5 G10 R0.5 | .473 | −.104 | +1.54 / +2.41 | guards pass |
| S6 G10 R1.0 (`configs/e2a-stitch.yaml`) | .464 | −.114 | +1.55 / +2.50 | **chosen** |

Confirmation: tune Δ −.114 [−.123, −.104]; sealed .647 → .520, Δ −.127
[−.137, −.117]. Dwell byte-identical. WILDTRACK HOTA −0.98 points (an
observation; no threshold was registered for it).

Decision: **accepted**; the target ≤ .30 is missed. mc-IDF1 with the
association unchanged drops from .175 to .153 and its oracle from .795 to .503:
some joins cross people, so stitching stays off in every mc-IDF1 recipe. Link:
[#30][pr30].

### Improvements plan, phase 3: real-time budget

Question: how many frames per second per camera does the fused map need, and
how many cameras then fit the GTX 1650 Ti?

Rule, registered before any run: subsample the cached stride-2 detections to
strides 2, 4, 6, 8, 10 and 16 (16 replaces the planned 15, which is not a
multiple of 2). A stride passes when the upper CI bound of Δ (stride S −
stride 2) is ≤ +.01 for both fused dwell TVD and fused passage mass error
(full@joint, tune). The largest passing stride is chosen. A camera count fits
when its p5 processed fps per camera is ≥ 30 / stride.

| Stride (processed fps) | Dwell TVD | Δ TVD | Passage mass error | Δ passage | Rule |
|---|---|---|---|---|---|
| 4 (7.5) | .2605 | +.0009 [−.0005, +.0023] | .4022 | −.0618 [−.0677, −.0560] | pass |
| 6 (5) | .2592 | −.0004 [−.0018, +.0012] | .3555 | −.1086 [−.1163, −.1010] | pass |
| 8 (3.75) | .2625 | +.0028 [+.0011, +.0045] | .3112 | −.1528 [−.1618, −.1437] | pass |
| 10 (3) | .2634 | +.0038 [+.0020, +.0056] | .2663 | −.1978 [−.2086, −.1873] | pass, **chosen** |
| 16 (1.875) | .2711 | +.0114 [+.0093, +.0135] | .0858 | −.3783 [−.3990, −.3533] | fail |

Bench at stride 10, one process, one shared detector, 10 min per count: 1
camera p5 31.33, 2 → 15.51, 4 → 7.84, 8 → 3.87, 15 → 2.14 processed fps per
camera; peak VRAM 174 MB. Sealed at stride 10: Δ TVD +.0054 [+.0033, +.0077],
Δ passage −.2269 [−.2386, −.2158]; the rule holds.

Decision: **stride 10, 8 cameras** (15 do not fit). The falling passage error
is a change in the metric (sparse foot points cross fewer cells), not a gain,
so passage is not compared across strides. During the bench another process
held 242 MiB of GPU memory at 0 % utilisation. Link: [#32][pr32].

### Improvements plan, phase 4: fusion weights

Question: the overlap mass error is above .10 on 073, 079 and 080. Do other
fusion weights lower it without losing TVD?

Rule, registered before any run: base F5 on the phase 3 recipe (stride 10).
Variants W1 power 2 on 2 m zones; W2 covered-cell normalised power 1; W3
covered-cell normalised power 2. Selection: among the variants whose Δ TVD
full@joint CI has a lower bound ≤ 0, the lowest pooled overlap mass error.
Acceptance: pooled Δ overlap CI excludes 0, Δ ≤ −.01. Target (reported, not a
gate): ≤ .10 on every tune scene.

| Variant | Overlap mass error | Δ vs F5 | Per scene 071/072/073/074/075 | Δ TVD | Decision |
|---|---|---|---|---|---|
| F5 (base) | .0922 | – | .059/.131/.146/.073/.053 | – | – |
| W1 (`configs/e2a-weights.yaml`) | .0604 | −.0318 [−.0330, −.0305] | .022/.095/.110/.049/.025 | −.0010 [−.0015, −.0005] | **chosen, accepted** |
| W2 | .0845 | −.0077 [−.0089, −.0067] | .052/.125/.133/.065/.048 | +.0022 [+.0014, +.0028] | not eligible |
| W3 | .0484 | −.0438 [−.0462, −.0402] | .012/.084/.091/.037/.018 | +.0027 [+.0017, +.0035] | not eligible |

Sealed, W1 against F5: overlap .1174 → .0691, Δ −.0483 [−.0502, −.0462];
Δ TVD −.0023 [−.0029, −.0016].

Decision: **W1 accepted**. The per-scene target is missed on 073 (.110 tune)
and 080 (.124 sealed). W3 meets it on every tune scene but loses TVD, so the
rule excludes it. Link: [#34][pr34].

### Improvements plan, phase 5: multi-camera recording

Product phase with binary criteria, registered before any run. Recipe: stride
10 with W1, stitching off.

- P1 map parity: the recorded fused dwell of scene 071 (15 cameras, replayed
  from files), read back with k=5, against the evaluation export of the same
  recipe; pass at TVD ≤ .01.
- P2 real time: 8 cameras of 071 as RTSP loops for 10 min; pass at p5 ≥ 3.0
  processed fps per camera and VRAM ≤ 3.5 GB.
- P3 privacy: only aggregated grids stored, no shown cell under 5 distinct
  ids, fusion weights frozen per site.
- P4 viewer: fused slots and live layer at 1440 and 390 px, no horizontal
  scroll, no console errors, explicit empty states.

Result:
- P1: first attempt TVD .105 (the recorder did not fill skipped frames as the
  evaluation does); after the fix, .0028, and .00025 with weights recalibrated
  with the fill (16021 against 16016 person-seconds, 2085 cells each). Pass.
- P2: 3.0 processed fps on all 8 cameras on an idle machine, the ceiling of a
  live source at stride 10; recorder VRAM 174 MB. Under a concurrent CPU load
  it read 2.0; the requirement was set as a run on an idle machine, which
  passed. The loops were served by VLC over UDP on loopback, since mediamtx
  could not be run in the test environment.
- P3 and P4: pass.

Decision: criteria met. Link: [#38][pr38].

### Improvements plan, phase 6: research and experiment loop

A literature survey of the open gaps returned six candidates; five were
registered before any run and one (decoder threads) was dropped, since 8
cameras already exceeded the need. Base: the product recipe
`configs/e2a-weights.yaml` (stride 10, W1), tune, from cache. Default
acceptance; each accepted metric change to be scored once on sealed at the end.
Stop rule: after 2 consecutive rejections, reframe once or close.

**H1, partial covered-cell normalisation.** The zone ratio is divided by
zone_cells^(1−α) · covered_cells^α, power 2, α ∈ {.25, .5, .75}, against W1
(α = 0). Metric: overlap mass error, target ≤ .10 on every tune scene. Rule:
phase 4's.

| α | Overlap mass error | Δ vs W1 | 073 | Δ TVD | Decision |
|---|---|---|---|---|---|
| .25 | .0573 | −.0031 [−.0036, −.0023] | .105 | +.0004 [+.000007, +.0007] | not eligible |
| .5 | .0540 | −.0064 [−.0074, −.0046] | .0999 | +.0012 [+.0005, +.0017] | not eligible |
| .75 | .0509 | −.0095 [−.0111, −.0068] | .095 | +.0023 [+.0014, +.0031] | not eligible |

Decision: **rejected**. Every α raises TVD with a CI above 0, and no Δ reaches
−.01. W1 stays. Branch: [feat/36-partial-normalisation][b36].

**H2, identity-free passage by chord length.** Each tracklet adds
min(1, length in cell / (π·a/4)) per cell, without stitching. Metric: passage
mass error full@joint at stride 10, target ≤ .30. Acceptance: Δ against the
current passage at stride 10 with CI excluding 0 and Δ ≤ −.01, and stride
invariance |passage(s10) − passage(s2)| ≤ .05. Guards: passage TVD Δ upper
bound ≤ +.01, dwell byte-identical.

| Pair | Passage mass error | Δ |
|---|---|---|
| Baseline: ids s10 vs ids s2 | .433 vs .609 | −.176 [−.185, −.167] |
| Chord s10 vs ids s10 (the rule's Δ) | .278 vs .433 | −.155 [−.162, −.148] |
| Chord s10 vs chord s2 (stride invariance) | .278 vs .543 | −.265 [−.278, −.253] |

Decision: **rejected**. The effect holds and the guards hold, but stride
invariance fails (.265 against ≤ .05). The three runs started together, so the
baseline was not measured before the chord runs; the rule was fixed in the
script before any run. Branch: [feat/37-chord-passage][b37].

**Stop rule.** H2 and H1 were two consecutive rejections, so the metric track
was closed: H5 (outlier mass gating, with a pre-check on 071 before any run)
was not run, and sealed was not opened again. H3 and H4 address stated product
defects and ran under their own rules.

**H3, k counted on stitched chains.** k=5 counted on G10 R1 chains instead of
raw track ids. Metric: k-violation rate, the share of shown cells per slot
whose true distinct people are fewer than 5; target ≤ 1 %. Guard: hidden true
dwell mass rises by at most +.05. Acceptance: the rate falls with a CI
excluding 0, reaches ≤ 1 %, and the guard holds.

| Surface | Ids | Chains | Δ | Hidden mass ids → chains |
|---|---|---|---|---|
| Camera slot | .194 [.168, .228] | .180 [.156, .211] | −.014 [−.018, −.010] | .203 → .217 |
| Fused slot | .178 [.140, .205] | .162 [.127, .187] | −.016 [−.019, −.013] | .211 → .228 |

Decision: **rejected** (far from ≤ 1 %). A diagnostic after the run found the
violations come mostly from foot-point error spreading people over neighbouring
cells, not from fragmentation. `k_count: chains` exists as an opt-in switch
only on the branch. Branch: [feat/39-k-on-chains][b39].

**H4, occupancy reading in the viewer.** Mean occupancy (Σ dwell / slot
seconds) and peak 1-minute occupancy replace id-based "people" counts; the
page opens on the latest slot; a table of the top 2 m zones; the map cropped to
the observed extent. Criteria, all required, at 1440 and 390 px: no "people"
number not derived from dwell; mean occupancy within ±10 % of truth; map drawn
on load with 0 console errors; ≥ 10 zone rows; no horizontal scroll; no
information lost.

Result: site 071 (15 cameras) reads 25.4 against a truth of 26.35 (−3.6 %);
single camera 071/645 reads 6.8 against 8.47 (−19.7 %): **fail**. Every other
criterion passes. A first attempt summed only the cells shown at k (site
−24 %); the registered quantity is Σ dwell over the whole grid, so the minute
totals were changed to include hidden cells. This change was made after seeing
that attempt.

Decision: **not accepted as registered**. The miss is single-camera dwell
recall (the E1 mass error), not the viewer. The branch was merged as a defect
fix: the page it replaced showed "652 people" (raw ids, about 25 times the
truth) and a k=5 claim that H3 showed false for about 1 shown cell in 5. Link:
[#41][pr41].

### Improvements plan, phase 7

Two hypotheses registered before any run, one for the k=5 gap and one for the
single-camera miss of H4; H8 was registered after H6's decision, before its own
run.

**H6, k counted on coarser units.** Variants: Z1 display 0.5 m, k on the 1 m
cell containing it; Z2 display and k on 1 m cells; Z3 display and k on 2 m
zones; raw ids, against today's Z0 (k on 0.5 m). Metric: violation rate of the
displayed unit, per camera and fused; target ≤ 5 % on both. Guards: hidden true
dwell mass ≤ +.10 against Z0; dwell TVD full@joint of the shown map not worse
by more than .02. Selection: the finest display unit that meets the target and
the guards; if none does, the best one that holds the guards and whose CI
excludes the current rate.

| Variant | Violation per camera | Violation fused | Hidden mass camera / fused | TVD (Δ) | Decision |
|---|---|---|---|---|---|
| Z0 (today) | .194 [.168, .228] | .178 [.140, .205] | .203 / .211 | .338 | – |
| Z1, on its 0.5 m cell | .346 [.316, .383] | .355 [.310, .391] | .089 / .090 | .282 (−.056) | worse rate |
| Z2 | .158 [.132, .193] | .161 [.115, .202] | .113 / .101 | .383 (+.046) | TVD guard broken |
| Z3 | .142 [.111, .181] | .161 [.119, .202] | .055 / .043 | .430 (+.093) | TVD guard broken |

Decision: **rejected**. No variant reaches ≤ 5 %. A coarser unit gathers more
true people but also more ids per person (1.54 at 0.5 m, 1.81 at 1 m), so the
rate barely moves. k stays on the 0.5 m cell, and the page and README state the
measured rates. Link: [#45][pr45].

**H7, one recall factor for a single camera.** One global factor f = Σ truth /
Σ shown mean occupancy over the 79 tune cameras, applied only to single-camera
readings. Metric: median |error| of single-camera mean occupancy against
visible truth on the 80 sealed cameras, scored once; target ≤ 10 %. Guard: site
readings byte-identical. Fallback if rejected: show "at least N" and the tune
error range.

Result: f = 1.133 (per scene 1.09 to 1.18). Sealed corrected median |error|
13.8 % (p90 37.1 %); uncorrected 14.5 % (p90 43.1 %). Site readings
byte-identical.

Decision: **rejected**; no factor in the product. The registered fallback
"at least N" was not used, because 22 of 79 tune cameras read above their truth
(up to +33 %), so it is not a bound. Deviation from the rule: a single-camera
reading shows "about N" with the sealed error (median 14.5 %, 8 cameras in 10
between 43 % below and 5 % above). Link: [#44][pr44].

**H8, a higher id threshold per 0.5 m cell.** Registered after H6, before any
H8 run: T ∈ {5, 7, 10, 15} distinct raw ids per cell and slot, the same T live.
Metric, target and guards as H6. Selection: the smallest T that meets the
target and both guards, then scored once on sealed.

| T | Violation per camera | Violation fused | Hidden mass camera / fused | TVD (Δ) | Decision |
|---|---|---|---|---|---|
| 5 (today) | .194 [.168, .228] | .178 [.140, .205] | .203 / .211 | .338 | – |
| 7 | .113 [.090, .143] | .096 [.070, .115] | .295 / .301 | .395 (+.057) | TVD guard broken |
| 10 | .058 [.043, .078] | .048 [.033, .061] | .431 / .437 | .502 (+.165) | target missed per camera; both guards broken |
| 15 | .027 [.019, .035] | .020 [.017, .023] | .619 / .636 | .681 (+.343) | both guards broken |

Decision: **rejected**. No T meets the target and both guards; T stays 5, and
sealed was not scored. Across H3, H6 and H8, a strict cell-level k (every shown
cell crossed by at least 5 people) is not reached by tuning the id test without
destroying the map. The measured rates are stated on the page and in the
[README](../README.md#what-k5-does-and-does-not-guarantee). Link: [#45][pr45].

### Improvements plan, phase 8: opening the repository

Not an experiment: licence, dependency and dataset licences, secret scan, and a
README for outside readers. Link: [#47][pr47].

[c-phase1]: https://github.com/xoel-maestu/space-usage-heatmaps/commit/5ef57a5
[pr2]: https://github.com/xoel-maestu/space-usage-heatmaps/pull/2
[pr4]: https://github.com/xoel-maestu/space-usage-heatmaps/pull/4
[pr6]: https://github.com/xoel-maestu/space-usage-heatmaps/pull/6
[pr8]: https://github.com/xoel-maestu/space-usage-heatmaps/pull/8
[pr10]: https://github.com/xoel-maestu/space-usage-heatmaps/pull/10
[pr12]: https://github.com/xoel-maestu/space-usage-heatmaps/pull/12
[pr14]: https://github.com/xoel-maestu/space-usage-heatmaps/pull/14
[pr16]: https://github.com/xoel-maestu/space-usage-heatmaps/pull/16
[pr19]: https://github.com/xoel-maestu/space-usage-heatmaps/pull/19
[pr20]: https://github.com/xoel-maestu/space-usage-heatmaps/pull/20
[pr22]: https://github.com/xoel-maestu/space-usage-heatmaps/pull/22
[pr24]: https://github.com/xoel-maestu/space-usage-heatmaps/pull/24
[pr26]: https://github.com/xoel-maestu/space-usage-heatmaps/pull/26
[pr29]: https://github.com/xoel-maestu/space-usage-heatmaps/pull/29
[pr30]: https://github.com/xoel-maestu/space-usage-heatmaps/pull/30
[pr32]: https://github.com/xoel-maestu/space-usage-heatmaps/pull/32
[pr34]: https://github.com/xoel-maestu/space-usage-heatmaps/pull/34
[pr38]: https://github.com/xoel-maestu/space-usage-heatmaps/pull/38
[pr41]: https://github.com/xoel-maestu/space-usage-heatmaps/pull/41
[pr44]: https://github.com/xoel-maestu/space-usage-heatmaps/pull/44
[pr45]: https://github.com/xoel-maestu/space-usage-heatmaps/pull/45
[pr47]: https://github.com/xoel-maestu/space-usage-heatmaps/pull/47
[b36]: https://github.com/xoel-maestu/space-usage-heatmaps/tree/feat/36-partial-normalisation
[b37]: https://github.com/xoel-maestu/space-usage-heatmaps/tree/feat/37-chord-passage
[b39]: https://github.com/xoel-maestu/space-usage-heatmaps/tree/feat/39-k-on-chains
