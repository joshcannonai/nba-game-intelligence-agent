#!/usr/bin/env bash
# Render README.md and docs/REPORT.md to self-contained HTML for reading or sharing.
#
# The markdown files are the source of truth -- GitHub renders them fine. This exists
# for the cases GitHub does not cover: reading offline, printing the report to PDF
# (the stylesheet has a print block), or handing someone a single file that needs no
# clone and no network.
#
#     ./scripts/render_docs.sh            # -> docs/html/
#     ./scripts/render_docs.sh /tmp/out   # -> somewhere else
#
# Output is gitignored; regenerate rather than commit it.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/docs/html}"
CSS="$ROOT/docs/assets/doc.css"

command -v pandoc >/dev/null || { echo "pandoc not found: brew install pandoc"; exit 1; }

mkdir -p "$OUT"

render() {
  local src="$1" dest="$2" title="$3"
  pandoc "$src" -f gfm -t html5 --standalone --self-contained \
    -c "$CSS" --metadata title="$title" -o "$dest"
  echo "  $(basename "$dest")"
}

echo "Rendering to $OUT"
render "$ROOT/README.md"       "$OUT/explainer.html" "NBA Game Intelligence Agent — How it works"
render "$ROOT/docs/REPORT.md"  "$OUT/report.html"    "NBA Game Intelligence Agent — Final Report (Draft v1)"
