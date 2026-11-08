# space-usage-heatmaps

Builds anonymous foot-traffic heatmaps from fixed cameras: positions on a floor
plan, never images or identities. Python, runs locally on a modest NVIDIA GPU.

![One camera's video beside its floor plan: a dot at each person's foot point, and the dwell map building up](docs/media/demo.gif)

Camera 645 of scene 071 at 2x speed (`scripts/demo-gif.py`): the foot point of
each person in the image, the same positions on the floor plan, and the dwell
accumulated since the video started, shown only in cells at least 5 people
have crossed. Synthetic scene, adapted (overlays, resized) from
[NVIDIA PhysicalAI-SmartSpaces](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Results (E1, one camera)

`uv run space-usage eval --suite e1-final` writes the full report to
`reports/e1-final.json` and `reports/e1-final.md`. Everything below is
`configs/h5-best.yaml` (YOLOX-s at 960 on TensorRT fp16, ByteTrack, the
auto-track foot point), scored on NVIDIA MTMC 2024 against floor ground truth:
dwell maps on 0.5 m cells, the people the camera can see, 5 scenes per split,
95 % paired block-bootstrap CI. Every parameter was chosen on the tune scenes
(071 to 075). The sealed scenes (076 to 080) were scored once, at the end.

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
sealed. Phase 9 fuses several cameras to cover those gaps.

External datasets, evaluation only, never tuned on:

| Dataset | Metric | Value |
|---|---|---|
| WILDTRACK camera C5 (1.68 m high) | ground position error, boxes paired in the image | median 21.8 cm, p90 45.1 cm, recall 0.71 (target median ≤ 30 cm: met) |
| WILDTRACK camera C5 | ground position error, people paired on the floor within 1 m | median 21.5 cm, p90 43.0 cm, recall 0.76 |
| CAVIAR, 26 corridor clips | HOTA / IDF1 | 0.670 / 0.779 |
| MOT17-02, 04, 09 | HOTA / IDF1 | 0.404 / 0.476 |

Label-free metrics are computed from the tracks alone, so they can run on a
camera without ground truth. Of the three, the share of tracks born away from
the image border follows the true error most closely: Spearman 0.75 with TVD
over the 10 scenes, and 0.49 over 265 blocks of 30 s. Occupancy conservation
and mean track length correlate weakly (|ρ| ≤ 0.44). With 10 scenes these are
indications, not proof.

Speed, `space-usage bench`, 30 min on one 1080p camera with nothing else on the
GPU (GTX 1650 Ti, 4 GB): 42.5 fps mean, 40.1 fps p5 (target ≥ 15), VRAM peak
174 MB (target ≤ 3.5 GB).

## Live viewer

`space-usage serve` hosts a page with the floor-plan heatmap of one camera: the
live layer (60 s half-life, held in memory only) updating every second, and the
persisted 15-minute slots. With `--source` it records that camera in the same
process, so the live layer never leaves memory; without it, it shows slots only.
Both layers read a cell with fewer than 5 distinct people as empty: a slot over
its quarter, the live layer over the last 2 minutes. It binds to 127.0.0.1
unless `--host` says otherwise, and warns when it binds beyond loopback.

```bash
uv run space-usage serve --config configs/h5-best.yaml --db slots.db \
  --source rtsp://127.0.0.1:8554/s071 --port 8000
# open http://127.0.0.1:8000/
```

API: `GET /api/live?camera=` returns `{cell_m, origin, shape, dwell, passage, t}`,
or 503 once the recorder has failed; `GET /api/slots?camera=&from=&to=` returns
the slots starting in `[from, to)` (ISO 8601 with a zone, or epoch seconds,
clamped to 1970 to 2100), k-suppressed as they are stored.

### A local RTSP stream from a video file

`scripts/rtsp-loop.sh` serves a file in a loop with
[mediamtx](https://github.com/bluenviron/mediamtx) (MIT, not a dependency) and
ffmpeg. Download the `linux_amd64` release and its `checksums.sha256`, check it,
then:

```bash
grep linux_amd64 checksums.sha256 | sha256sum -c -
tar -xzf mediamtx_v*_linux_amd64.tar.gz -C ~/tools/mediamtx
MEDIAMTX=~/tools/mediamtx/mediamtx scripts/rtsp-loop.sh \
  "$SPACE_USAGE_DATA_DIR/nvidia/MTMC_Tracking_2024/test/scene_071/camera_0645/video.mp4" s071
# -> rtsp://127.0.0.1:8554/s071 until Ctrl+C
```
