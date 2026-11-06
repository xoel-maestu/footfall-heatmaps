"""A dataset shipped as image files (WILDTRACK, MOT17) as one video file: the
perception pipeline reads videos. FFV1 is lossless, so the detector sees the
dataset's own pixels. The video is derived data under the data dir, rebuilt
when the frame list or rate changes (a JSON note beside it)."""

import json
from pathlib import Path

import cv2


def sequence_video(frames, fps, out):
    """The FFV1 video of the image files frames, in that order, at fps."""
    out = Path(out)
    note = out.with_suffix(".json")
    expected = {"frames": [Path(f).name for f in frames], "fps": fps}
    if out.exists() and note.exists() and json.loads(note.read_text()) == expected:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise OSError(f"cannot read frame {frames[0]}")
    height, width = first.shape[:2]
    tmp = out.with_name(out.stem + ".tmp" + out.suffix)
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"cannot write {tmp}: OpenCV has no FFV1 encoder")
    try:
        for f in frames:
            image = cv2.imread(str(f))
            if image is None or image.shape[:2] != (height, width):
                raise OSError(f"frame {f} is missing or not {width}x{height}")
            writer.write(image)
    finally:
        writer.release()
    tmp.replace(out)
    note.write_text(json.dumps(expected))
    return out
