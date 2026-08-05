#!/usr/bin/env bash
# Render docs/REPORT.md into the shared Google Doc, IN PLACE.
#
# The doc ID is fixed on purpose: the link is already circulated to the group,
# so this must always update that document and never create a new one.
#
#     ./scripts/to_gdoc.sh
#
# Needs `gws` authenticated as the doc owner. gws only accepts uploads from the
# current directory, hence the temp file in the repo root.

set -euo pipefail

DOC_ID="1kr87GpueIulw8N_Fo4lmUSEs7ju0XdH_8dYRlzckv7w"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$ROOT/_gdoc_upload.html"

cd "$ROOT"
trap 'rm -f "$TMP"' EXIT

pandoc docs/REPORT.md -f gfm -t html5 --standalone --metadata title="" -o "$TMP"

# Docs partially honours pandoc's default stylesheet and inflates every table
# header into a near-empty row. Stripping it lets Docs apply its own table style.
python3 - "$TMP" <<'PY'
import re, sys
p = sys.argv[1]
h = open(p).read()
open(p, "w").write(re.sub(r"<style>.*?</style>", "", h, flags=re.S))
PY

gws drive files update \
  --params "{\"fileId\":\"$DOC_ID\",\"fields\":\"id,name,modifiedTime\"}" \
  --upload "$(basename "$TMP")" --upload-content-type "text/html" --format json

echo "https://docs.google.com/document/d/$DOC_ID/edit"
