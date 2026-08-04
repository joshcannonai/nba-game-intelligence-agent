#!/usr/bin/env bash
# Put the app on a public URL, still running entirely on this laptop.
#
#     ./scripts/share.sh            # report UI
#     ./scripts/share.sh chat       # conversational UI
#
# Everything -- Streamlit, the data, the gate, and Gemma via Ollama -- runs here.
# Cloudflare just forwards a public hostname to localhost. Nothing is deployed and
# no data leaves the machine except what the page renders.
#
# WHY THIS RATHER THAN HOSTING IT PROPERLY. The scored results depend on a model
# whose knowledge cutoff predates the season we test on. A hosted API model would
# almost certainly have the 2025-26 results in its weights, so the numbers would
# stop meaning anything. Tunnelling keeps Gemma, which keeps the guarantee.
#
# WHAT YOU ARE ACCEPTING:
#   * No authentication. Anyone with the URL can use the app. The URL is random
#     and unguessable, but it is not a password. Do not post it publicly.
#   * It lives as long as this terminal and this laptop do. Close the lid and the
#     site goes down. `caffeinate -s ./scripts/share.sh` keeps the Mac awake.
#   * The URL changes every restart. Quick tunnels are ephemeral by design.
#
# Ctrl-C stops the tunnel and the Streamlit server it started.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-report}"
case "$MODE" in
  report) APP="ui/app.py";  PORT=8501 ;;
  chat)   APP="ui/chat.py"; PORT=8502 ;;
  *) echo "usage: $0 [report|chat]"; exit 1 ;;
esac

command -v cloudflared >/dev/null || { echo "need cloudflared: brew install cloudflared"; exit 1; }
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "no .venv -- run: python3 -m venv .venv && pip install -r requirements.txt"; exit 1; }

started_streamlit=""
cleanup() {
  [ -n "$started_streamlit" ] && kill "$started_streamlit" 2>/dev/null || true
  [ -n "${tunnel_pid:-}" ] && kill "$tunnel_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Reuse an already-running server rather than fighting it for the port.
if curl -s -o /dev/null "http://localhost:$PORT"; then
  echo "using the Streamlit server already on :$PORT"
else
  echo "starting $APP on :$PORT"
  "$PY" -m streamlit run "$APP" --server.port "$PORT" --server.headless true >/dev/null 2>&1 &
  started_streamlit=$!
  for _ in $(seq 1 30); do
    curl -s -o /dev/null "http://localhost:$PORT" && break
    sleep 1
  done
fi

# Ollama is only needed for the agent path; the report UI is deterministic without it.
if curl -s -o /dev/null http://localhost:11434/api/tags; then
  echo "ollama is up -- the agent path will work"
else
  echo "note: ollama is not running, so the LLM path is unavailable (run: ollama serve)"
fi

LOG="$(mktemp -t nbatunnel)"
cloudflared tunnel --url "http://localhost:$PORT" >"$LOG" 2>&1 &
tunnel_pid=$!

echo -n "opening tunnel"
URL=""
for _ in $(seq 1 40); do
  URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | head -1 || true)"
  [ -n "$URL" ] && break
  echo -n "."
  sleep 1
done
echo

[ -n "$URL" ] || { echo "tunnel did not come up; see $LOG"; exit 1; }

cat <<EOF

  $URL

  Running on this laptop. No auth -- treat the URL as semi-private.
  Ctrl-C to stop.

EOF

wait "$tunnel_pid"
