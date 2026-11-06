# space-usage-heatmaps

Builds anonymous foot-traffic heatmaps from fixed cameras: positions on a floor
plan, never images or identities. Python, runs locally on a modest NVIDIA GPU.

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
