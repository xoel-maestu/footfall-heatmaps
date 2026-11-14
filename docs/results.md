# Results

Detailed measurements behind the summary in the [README](../README.md#results).
Every number here comes from a command in this repository; the reports those
commands write go to `reports/`, which is not committed, so you get them by
running the commands.

## Terms

- **Tune and sealed.** The 10 test scenes of NVIDIA MTMC_Tracking_2024 used
  here are split in two (`src/space_usage_heatmaps/evaluation/domain/splits.py`),
  one camera per scene. Every parameter was chosen on the *tune* scenes (071 to
  075). The *sealed* scenes (076 to 080) were held out and scored once, at the
  end, to check that the choices generalise.
- **E1, E2a, E2b.** The three evaluations: E1 is one camera's heatmap; E2a
  fuses all the cameras of a scene into one floor map; E2b links the same
  person across cameras on the floor (multi-camera IDF1).
- **Iteration 0** is the baseline: the Ultralytics reference recipe (YOLOv8n
  and its built-in ByteTrack, box centre as the floor point), measured as is
  (`configs/iter0.yaml`, optional `baseline` group).
- **H1, H2, ...** label the hypotheses tested against it, one config change
  each (`configs/h1-*.yaml` to `configs/h7-*.yaml`). A change was accepted only
  when its paired bootstrap CI on tune showed a gain and the tracking guard
  held.
- **Tracking guard.** An accepted step may not lower HOTA or IDF1 by more than
  1 point against the config it replaced.
- **Experiment log, phases.** Comments in code, configs and scripts cite the
  [experiment log](experiment-log.md), its phases ("phase 9", "improvements
  plan, phase 4") and the rules pre-registered there before each run. The log
  lists every hypothesis, accepted or rejected, with its rule, numbers and
  decision.
- **Metrics.** TVD: total variation distance between the predicted and true
  dwell maps (0 is identical, 1 is disjoint). Mass error: the share of total
  dwell missing or added. Zone WAPE: weighted absolute percentage error on 2 m
  zones. EMD: earth mover's distance, in metres. HOTA and IDF1: standard
  tracking metrics on the camera's boxes.

## E1, one camera

`uv run space-usage eval --suite e1-final` writes the full report to
`reports/e1-final.json` and `reports/e1-final.md`. Everything below is
`configs/h5-best.yaml` (YOLOX-s at 960 on TensorRT fp16, ByteTrack, the
auto-track foot point), scored on NVIDIA MTMC 2024 against floor ground truth:
dwell maps on 0.5 m cells, the people the camera can see, 5 scenes per split,
95 % paired block-bootstrap CI.

| Metric | Tune | Sealed | Degradation (positive = worse) | Target | Sealed meets it |
|---|---|---|---|---|---|
| TVD | 0.308 [0.294, 0.324] | 0.419 [0.402, 0.436] | +36 % | ≤ 0.30 | no |
| Mass error | 0.154 [0.142, 0.166] | 0.297 [0.284, 0.310] | +92 % | ≤ 0.10 | no |
| Zone WAPE (2 m) | 0.231 [0.211, 0.252] | 0.380 [0.351, 0.412] | +65 % | ≤ 0.30 | no |
| EMD | 1.50 m [1.37, 1.64] | 3.19 m [3.03, 3.36] | +113 % | ≤ 1.2 m | no |
| HOTA / IDF1 | 0.237 / 0.182 | 0.218 / 0.189 | +8 % / −4 % | information only | – |

E1 is not accepted. The sealed scenes miss the targets and the rule that
sealed stays within 20 % of tune. The sealed scenes are harder for every
configuration, not only the tuned one: the iteration 0 baseline also degrades,
from TVD 0.629 to 0.671 and mass error 0.273 to 0.458. Two scenes carry most of
the gap: 078 (TVD 0.602) and 080 (0.477). Each step accepted on tune keeps its
effect on sealed, and none of them lowers HOTA or IDF1 on sealed by more than
1 point against the config it replaced (ΔHOTA / ΔIDF1 in points, positive is
better):

| Step | ΔTVD tune | ΔTVD sealed | ΔHOTA / ΔIDF1 sealed | Tracking guard |
|---|---|---|---|---|
| Auto-track foot point vs box centre (iteration 0) | −0.248 [−0.264, −0.232] | −0.191 [−0.203, −0.180] | +0.0 / +0.0 | holds |
| YOLOX-s detector | −0.026 [−0.034, −0.018] | −0.023 [−0.029, −0.016] | +3.3 / +3.6 | holds |
| Input 640 → 960 | −0.047 [−0.056, −0.038] | −0.038 [−0.045, −0.031] | +2.0 / +1.6 | holds |
| TensorRT fp16 (speed, no accuracy loss) | −0.001 [−0.002, 0.000] | −0.001 [−0.003, −0.000] | +0.1 / +0.0 | holds |

What remains is mostly occlusion. One camera does not see people behind
shelves: against every person on the floor, TVD is 0.450 on tune and 0.557 on
sealed. Fusing several cameras of a floor (E2a, `space-usage eval --cameras`,
and the site recorder) covers those gaps.

## External datasets

Evaluation only, never tuned on (`space-usage eval --config ... --dataset`):

| Dataset | Metric | Value |
|---|---|---|
| WILDTRACK camera C5 (1.68 m high) | ground position error, boxes paired in the image | median 21.8 cm, p90 45.1 cm, recall 0.71 (target median ≤ 30 cm: met) |
| WILDTRACK camera C5 | ground position error, people paired on the floor within 1 m | median 21.5 cm, p90 43.0 cm, recall 0.76 |
| CAVIAR, 26 corridor clips | HOTA / IDF1 | 0.670 / 0.779 |
| MOT17-02, 04, 09 | HOTA / IDF1 | 0.404 / 0.476 |

## Label-free metrics

Label-free metrics are computed from the tracks alone, so they can run on a
camera without ground truth. Of the three, the share of tracks born away from
the image border follows the true error most closely: Spearman 0.75 with TVD
over the 10 scenes, and 0.49 over 265 blocks of 30 s. Occupancy conservation
and mean track length correlate weakly (|ρ| ≤ 0.44). With 10 scenes these are
indications, not proof.

## Speed and cameras per GPU

Speed, `space-usage bench`, 30 min on one 1080p camera with nothing else on the
GPU (GTX 1650 Ti, 4 GB): 42.5 fps mean, 40.1 fps p5 (target ≥ 15), VRAM peak
174 MB (target ≤ 3.5 GB).

Cameras per GPU (`scripts/multicam-bench.py`): 8 cameras at stride 10 (3
processed frames a second per 30 fps camera), with one shared TensorRT detector
in one process, 10 min on scene 071's videos, p5 3.87 processed fps per camera,
VRAM peak 174 MB as NVML reports it for the process, which matches nvidia-smi
(device memory used rose by 177 MiB over the idle baseline in a 2 min, 8-camera
rerun), on the same GTX 1650 Ti. Stride 10 is the largest stride whose fused
dwell map holds against stride 2 (`scripts/stride-sweep.py`; TVD +.004 tune,
+.005 sealed). 15 cameras do not fit: p5 2.14 processed fps per camera, in the
report `scripts/multicam-bench.py` writes for 15 cameras at stride 10.

## Occupancy reading

How far the viewer's "people in view on average" is from the truth
(`scripts/occupancy-truth.py`, `scripts/recall-factor.py`). On the held-out
sealed scenes (076-080, 80 cameras, stride 10, uncorrected) a single camera's
average was off by 14.5 % at the median (median error -14 %), between 43 %
below and 5 % above the people visible to it for 8 cameras in 10, and from
100 % below to 43 % above in all. One global recall factor (x1.13, fitted on
the tune scenes 071-075) did not bring the sealed cameras within 10 % (median
|error| 13.8 %), so no factor is applied. The 15-camera site of scene 071 read
3.6 % below.

## k-anonymity threshold

The measured limits of k=5 are in the README's
[privacy section](../README.md#privacy-by-design), with the scripts that
measure them (`scripts/k-zones-sweep.py`, `scripts/k-threshold-sweep.py`).
