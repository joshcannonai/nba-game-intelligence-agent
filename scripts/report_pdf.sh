#!/usr/bin/env bash
# Render docs/REPORT.md to a submittable PDF.
#
# The assignment requires a PDF upload, so this is the artifact that actually gets
# handed in. Chrome headless is used rather than LaTeX because no TeX distribution
# is installed; the stylesheet's @media print block handles the light-mode switch.
#
#     ./scripts/report_pdf.sh [output.pdf]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$HOME/Desktop/cecs499-docs/CECS499-Final-Report-Draft.pdf}"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

[ -x "$CHROME" ] || { echo "Chrome not found at $CHROME"; exit 1; }

"$ROOT/scripts/render_docs.sh" >/dev/null
mkdir -p "$(dirname "$OUT")"

"$CHROME" --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$OUT" "file://$ROOT/docs/html/report.html" 2>/dev/null

echo "$OUT"
