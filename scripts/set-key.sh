#!/bin/sh
# Write an API key into .env by READING THE CLIPBOARD.
#
# Never type or paste a key into a shell command -- it lands in your history and
# in any terminal scrollback you later share. Copy the key, then run this.
#
#   ./scripts/set-key.sh google      # after copying from aistudio.google.com/apikey
#   ./scripts/set-key.sh anthropic
#
# The key is never printed, only its length and last 4 characters.

set -eu

PROVIDER="${1:-}"
case "$PROVIDER" in
  google)    VAR=GOOGLE_API_KEY ;;
  anthropic) VAR=ANTHROPIC_API_KEY ;;
  *) echo "usage: $0 {google|anthropic}" >&2; exit 1 ;;
esac

ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENV_FILE="$ROOT/.env"

KEY=$(pbpaste | tr -d '[:space:]')

if [ -z "$KEY" ]; then
  echo "Clipboard is empty. Copy the key first, then re-run." >&2
  exit 1
fi
case "$KEY" in
  *' '*|*'	'*) echo "Clipboard does not look like a key." >&2; exit 1 ;;
esac
if [ ${#KEY} -lt 20 ]; then
  echo "Clipboard is only ${#KEY} chars -- that is not a key. Did you copy it?" >&2
  exit 1
fi

touch "$ENV_FILE"
chmod 600 "$ENV_FILE"
# Drop any existing line for this var, then append the new one.
if grep -q "^${VAR}=" "$ENV_FILE" 2>/dev/null; then
  grep -v "^${VAR}=" "$ENV_FILE" > "$ENV_FILE.tmp" && mv "$ENV_FILE.tmp" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
fi
printf '%s=%s\n' "$VAR" "$KEY" >> "$ENV_FILE"

echo "$VAR written to .env  (${#KEY} chars, ends ...$(printf '%s' "$KEY" | tail -c 4))"
echo ".env is gitignored. Clearing the clipboard."
printf '' | pbcopy
