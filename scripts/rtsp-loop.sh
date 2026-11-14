#!/usr/bin/env bash
# Serve a video file in a loop as a local RTSP stream, for footfall run/serve.
#
#   MEDIAMTX=/path/to/mediamtx scripts/rtsp-loop.sh <video.mp4> [name]
#   -> rtsp://127.0.0.1:8554/<name> (default name: s071), until Ctrl+C
#
# mediamtx (MIT, https://github.com/bluenviron/mediamtx/releases) is not a
# dependency of the package: download the linux_amd64 release, check it against
# the release's checksums.sha256 (sha256sum -c), and point MEDIAMTX at the binary.
# Only RTSP over TCP on 127.0.0.1 is enabled, every other listener is off.
# ffmpeg re-sends the file at its own rate (-re), without re-encoding, forever
# (-stream_loop -1). mediamtx runs in a temp dir: nothing lands in the repo.
set -euo pipefail

video=${1:?usage: MEDIAMTX=/path/to/mediamtx $0 <video.mp4> [name]}
name=${2:-s071}
mediamtx=$(realpath "$(command -v "${MEDIAMTX:-mediamtx}")")
port=${RTSP_PORT:-8554}

conf=$(mktemp --suffix=.yml)
cat >"$conf" <<EOF
logLevel: warn
rtspAddress: 127.0.0.1:$port
rtspTransports: [tcp]
rtmp: false
hls: false
webrtc: false
srt: false
moq: false
paths:
  all_others:
EOF

conf_dir=$(dirname "$conf")
(cd "$conf_dir" && exec "$mediamtx" "$conf") &
server=$!
trap 'kill "$server" 2>/dev/null; rm -f "$conf"' EXIT
sleep 1 # mediamtx opens its listener

echo "rtsp://127.0.0.1:$port/$name"
ffmpeg -hide_banner -loglevel warning -re -stream_loop -1 -i "$video" -c copy \
  -rtsp_transport tcp -f rtsp "rtsp://127.0.0.1:$port/$name"
