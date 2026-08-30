#!/usr/bin/env bash
# SVG is the deliverable. PNGs exist only so the figure can be looked at.
#   bash build.sh          -> render all iterations
#   bash build.sh iter03   -> render just one
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out

targets=()
if [ $# -gt 0 ]; then
  for a in "$@"; do targets+=("iterations/${a%.svg}.svg"); done
else
  for f in iterations/*.svg; do targets+=("$f"); done
fi

for f in "${targets[@]}"; do
  b="$(basename "${f%.svg}")"
  rsvg-convert -w 1600 "$f" -o "out/$b.png"
  echo "built  out/$b.png"
done
