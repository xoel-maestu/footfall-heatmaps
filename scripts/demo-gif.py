"""The README demo GIF: one camera's video next to its floor plan, in sync.

Left, the video frames with a dot at each tracked person's foot point (the
config's foot point, projected back into the image through the floor
homography). Right, the scene's floor grid with the camera's dwell accumulated
from the start of the video up to the shown frame (a cell shows once k distinct
ids have been in it, as the viewer's maps), the same people's floor positions
and the outline of the cells the camera sees.

Tracks come from the config's track source (the cache when it has them), so no
detector runs. Usage:

    uv run --no-sync python scripts/demo-gif.py --scene 071 --camera 645 \\
        --start 602 --seconds 12 --speed 2 --fps 10 --out docs/media/demo.gif
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from space_usage_heatmaps.occupancy.domain.grid import Grid, GridSpec
from space_usage_heatmaps.occupancy.domain.slots import K_MIN_PEOPLE, suppress

# The viewer's light theme (DESIGN.md, web/static/index.html).
RAMP = ((0xDD, 0xEF, 0xE6), (0x7C, 0xC4, 0xA0), (0x08, 0x7A, 0x50), (0x03, 0x31, 0x1F))
INK, MUTED, LINE, FLOOR, BACKGROUND = "#12151A", "#545B66", "#D5D9DF", "#F7F8FA", "#FFFFFF"
STEADY = 12  # per-channel change below which a video pixel keeps its last value
MARGIN, GAP, CAPTION_H = 16, 16, 74
CAPTION = (
    "Left: camera video, one dot per person at the estimated foot point. Right: floor plan in "
    "0.5 m cells, the camera's",
    "dwell since the video started (a cell shows once 5 people have been in it), the current "
    "positions and its field of view (outline). {speed:g}x speed.",
    "Synthetic scene, adapted (overlays, resized) from NVIDIA PhysicalAI-SmartSpaces, CC BY 4.0.",
)


def frame_indices(start_s, seconds, speed, fps, video_fps):
    """The video frames of the GIF's frames: seconds * fps of them from start_s,
    each speed / fps video seconds after the last."""
    n = round(seconds * fps)
    step = speed * video_fps / fps
    return np.round(start_s * video_fps + np.arange(n) * step).astype(int)


def cumulative_dwell(spec, track_ids, frames, xs, ys, video_fps, cutoffs, k=K_MIN_PEOPLE):
    """Per cutoff, the dwell grid (person-seconds) of every sample up to and
    including that frame, empty where fewer than k distinct ids have been."""
    out = []
    for cutoff in cutoffs:
        keep = frames <= cutoff
        g = Grid.from_samples(spec, track_ids[keep], xs[keep], ys[keep], video_fps)
        out.append(suppress(g.dwell, g.passage, k)[0])
    return out


def steady(previous, current, threshold):
    """current, with the pixels that differ from previous by less than
    threshold in every channel kept at previous: video noise on a still
    background then repeats the last frame's pixels, which a GIF stores cheaply."""
    diff = np.abs(current.astype(int) - previous.astype(int)).max(axis=-1)
    return np.where((diff < threshold)[..., None], previous, current)


def colour_cap(grid):
    """The viewer's colour cap: the nearest-rank 99th percentile of the cells
    above zero (0 when none is)."""
    values = np.sort(np.asarray(grid, float)[np.asarray(grid) > 0])
    return float(values[math.ceil(0.99 * len(values)) - 1]) if len(values) else 0.0


def ramp_rgb(f):
    """The viewer's four-stop green ramp at f in [0, 1] (clamped)."""
    x = min(1.0, max(0.0, f)) * (len(RAMP) - 1)
    i = min(len(RAMP) - 2, math.floor(x))
    a, b, w = RAMP[i], RAMP[i + 1], x - i
    return tuple(round(a[c] + (b[c] - a[c]) * w) for c in range(3))


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    panel_h: int
    video_x: int
    video_w: int
    video_h: int
    floor_x: int
    cell_px: float


def layout(width, video_w, video_h, nx, ny):
    """Video and floor side by side, both as tall as the floor grid, which keeps
    square cells; a caption strip below."""
    inner = width - 2 * MARGIN - GAP
    cell = inner / (ny * video_w / video_h + nx)
    panel_h = math.floor(cell * ny)
    cell = panel_h / ny
    vw = math.floor(panel_h * video_w / video_h)
    return Layout(
        width=width,
        height=MARGIN + panel_h + CAPTION_H,
        panel_h=panel_h,
        video_x=MARGIN,
        video_w=vw,
        video_h=round(vw * video_h / video_w),
        floor_x=MARGIN + vw + GAP,
        cell_px=cell,
    )


def _to_image(world_to_image, xs, ys):
    u, v, w = world_to_image @ np.vstack([xs, ys, np.ones(len(xs))])
    return u / w, v / w


def _font(size):
    from matplotlib import font_manager
    from PIL import ImageFont

    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)


def _dot(draw, x, y, r):
    draw.ellipse((x - r - 2, y - r - 2, x + r + 2, y + r + 2), fill=BACKGROUND)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=INK)


def render(args):
    import cv2
    import yaml
    from PIL import Image, ImageDraw

    from space_usage_heatmaps.cli import data_dir, load_config, track_source
    from space_usage_heatmaps.evaluation.application.evaluate_oracle import camera_floor_points
    from space_usage_heatmaps.evaluation.infrastructure.nvidia_mtmc_reader import NvidiaMtmcReader
    from space_usage_heatmaps.floorplan.domain.homography import Homography

    run = yaml.safe_load(Path(args.config).read_text())["run"]
    config = load_config(args.config)
    reader = NvidiaMtmcReader(data_dir())
    cam = reader.camera(args.scene, args.camera)
    t = track_source(config, reader).tracks(args.scene, args.camera)
    xs, ys, _ = camera_floor_points(
        cam,
        t.boxes,
        config.foot_point,
        config.person_height_m,
        config.min_implied_height_m,
        t.track_ids,
        t.frames,
        config.track_heights(),
    )
    spec = GridSpec.covering(*run["extent_m"], cell_m=run["cell_m"])
    coverage = Homography(cam.world_to_image).sees(*spec.cell_centres(), cam.width, cam.height)
    shown = frame_indices(args.start, args.seconds, args.speed, args.fps, cam.fps)
    grids = cumulative_dwell(spec, t.track_ids, t.frames, xs, ys, cam.fps, shown)
    lay = layout(args.width, cam.width, cam.height, spec.nx, spec.ny)
    sx, c = lay.video_w / cam.width, lay.cell_px
    us, vs = _to_image(cam.world_to_image, xs, ys)

    def floor_xy(x, y):  # metres to canvas pixels; y grows up the screen
        return (
            lay.floor_x + (x - spec.origin_x) / spec.cell_m * c,
            MARGIN + lay.panel_h - (y - spec.origin_y) / spec.cell_m * c,
        )

    base = Image.new("RGB", (lay.width, lay.height), BACKGROUND)
    d = ImageDraw.Draw(base)
    font = _font(13)
    for i, line in enumerate(CAPTION):
        text = line.format(speed=args.speed)
        d.text(
            (MARGIN, MARGIN + lay.panel_h + 8 + i * 20),
            text,
            fill=MUTED if i == len(CAPTION) - 1 else INK,
            font=font,
        )

    capture = cv2.VideoCapture(str(reader.video(args.scene, args.camera)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(shown[0]))
    position, frames, still = int(shown[0]), [], None
    for frame_index, dwell in zip(shown, grids, strict=True):
        while position <= frame_index:
            ok, bgr = capture.read()
            if not ok:
                raise SystemExit(f"video ends before frame {frame_index}")
            position += 1
        img = base.copy()
        video = cv2.cvtColor(
            cv2.resize(bgr, (lay.video_w, lay.video_h), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        still = video if still is None else steady(still, video, STEADY)
        img.paste(Image.fromarray(still), (MARGIN, MARGIN))
        draw = ImageDraw.Draw(img)
        x0, y0 = lay.floor_x, MARGIN
        draw.rectangle((x0, y0, x0 + spec.nx * c, y0 + spec.ny * c), fill=FLOOR)
        cap = colour_cap(dwell)
        for i, j in zip(*np.nonzero(dwell), strict=True):
            left, top = floor_xy(
                spec.origin_x + j * spec.cell_m, spec.origin_y + (i + 1) * spec.cell_m
            )
            draw.rectangle(
                (round(left), round(top), round(left + c) - 1, round(top + c) - 1),
                fill=ramp_rgb(dwell[i, j] / cap),
            )
        per = round(2 / spec.cell_m)  # grid lines every 2 m, as the viewer
        for j in range(0, spec.nx + 1, per):
            draw.line(
                (round(x0 + j * c), y0, round(x0 + j * c), round(y0 + spec.ny * c)), fill=LINE
            )
        for i in range(0, spec.ny + 1, per):
            draw.line(
                (x0, round(y0 + i * c), round(x0 + spec.nx * c), round(y0 + i * c)), fill=LINE
            )
        _outline(draw, coverage, spec, floor_xy)
        now = t.frames == frame_index
        for u, v, x, y in zip(us[now], vs[now], xs[now], ys[now], strict=True):
            if 0 <= u < cam.width and 0 <= v < cam.height:
                _dot(draw, MARGIN + u * sx, MARGIN + v * sx, 4)
            _dot(draw, *floor_xy(x, y), 3)
        frames.append(img)
    capture.release()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # One palette for every frame and no dithering: unchanged pixels stay the same
    # index from frame to frame, which keeps the file small.
    sample = Image.new("RGB", (lay.width, lay.height * 3))
    for k, f in enumerate((frames[0], frames[len(frames) // 2], frames[-1])):
        sample.paste(f, (0, k * lay.height))
    shared = sample.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    palette = [f.quantize(palette=shared, dither=Image.Dither.NONE) for f in frames]
    palette[0].save(
        out,
        save_all=True,
        append_images=palette[1:],
        duration=round(1000 / args.fps),
        loop=0,
        optimize=True,
    )
    print(f"{out}: {len(frames)} frames, {lay.width}x{lay.height}, {out.stat().st_size} bytes")


def _outline(draw, coverage, spec, floor_xy):
    """The edges between a covered cell and an uncovered one (or the grid's border)."""
    ny, nx = coverage.shape
    padded = np.pad(coverage, 1)
    m = spec.cell_m
    for i in range(ny):
        for j in range(nx):
            if not coverage[i, j]:
                continue
            x, y = spec.origin_x + j * m, spec.origin_y + i * m
            edges = (
                (padded[i, j + 1], (x, y, x + m, y)),  # the row below (smaller y)
                (padded[i + 2, j + 1], (x, y + m, x + m, y + m)),  # the row above
                (padded[i + 1, j], (x, y, x, y + m)),  # left
                (padded[i + 1, j + 2], (x + m, y, x + m, y + m)),  # right
            )
            for neighbour, (ax, ay, bx, by) in edges:
                if not neighbour:
                    draw.line((*floor_xy(ax, ay), *floor_xy(bx, by)), fill=MUTED, width=2)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="configs/h5-best.yaml")
    p.add_argument("--scene", default="071")
    p.add_argument("--camera", default="645")
    p.add_argument("--start", type=float, required=True, help="video second of the first frame")
    p.add_argument("--seconds", type=float, default=12, help="GIF length in seconds")
    p.add_argument("--speed", type=float, default=2, help="playback speed")
    p.add_argument("--fps", type=float, default=10, help="GIF frames per second")
    p.add_argument("--width", type=int, default=1200)
    p.add_argument("--out", default="docs/media/demo.gif")
    render(p.parse_args())


if __name__ == "__main__":
    main()
