"""Keep the B/C full-season eval alive. Do not start D/E.

CECS 499 is A/B/C. D/E waits until there is a real self-improvement loop.

1. Restarts ``eval.resume_bc --workers 4`` if it dies before both arms finish.
2. Waits until the checkpoint has 1,322 unique B games and 1,322 unique C games.
3. Waits for that eval process to exit.
4. Texts Josh that B/C finished. Granted 2026-08-12: one unattended send at 2644.

    PYTHONPATH=vendor python -m eval.watch_bc
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "eval" / "results_actual_ui_full_season.jsonl"
PIDFILE = ROOT / "eval" / ".watch_bc.pid"
NOTIFY_FLAG = ROOT / "eval" / ".bc_notified"
NOTIFY_DIR = (
    Path(os.environ["LIFEOS_NOTIFY_DIR"])
    if os.environ.get("LIFEOS_NOTIFY_DIR")
    else None
)
NOTIFY_ENV = (NOTIFY_DIR / ".env.local") if NOTIFY_DIR else None
NOTIFY_PY = (NOTIFY_DIR / "notify.py") if NOTIFY_DIR else None
TARGET_PER_ARM = 1322
POLL_SECONDS = 30
PYTHON = ROOT / ".venv" / "bin" / "python"
ENV = {
    **os.environ,
    "PYTHONPATH": f"vendor{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
}


def _log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def _counts(path: Path) -> Counter:
    arms: Counter = Counter()
    seen: set[tuple[str, str]] = set()
    if not path.exists():
        return arms
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row.get("arm")), str(row.get("game_id")))
        if key in seen:
            continue
        seen.add(key)
        arms[key[0]] += 1
    return arms


def _pgrep(*needles: str) -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-f", needles[0]], text=True)
    except subprocess.CalledProcessError:
        return []
    pids = []
    for raw in out.split():
        try:
            pid = int(raw)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        cmdline = Path(f"/proc/{pid}/cmdline")
        # macOS has no /proc; fall back to ps
        try:
            cmd = subprocess.check_output(
                ["ps", "-p", str(pid), "-o", "command="], text=True
            ).strip()
        except subprocess.CalledProcessError:
            continue
        if all(n in cmd for n in needles) and "eval.watch_bc" not in cmd:
            pids.append(pid)
    return pids


def _bc_running() -> bool:
    return bool(_pgrep("eval.resume_bc"))


def _start_bc() -> None:
    _log("starting eval.resume_bc --workers 4")
    subprocess.Popen(
        [
            str(PYTHON),
            "-m",
            "eval.resume_bc",
            "--full-season",
            "--arms",
            "BC",
            "--checkpoint",
            str(CHECKPOINT),
            "--out",
            str(ROOT / "eval" / "results_actual_ui_full_season.csv"),
            "--workers",
            "4",
        ],
        cwd=ROOT,
        env=ENV,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def _notify_josh() -> None:
    """One iMessage + Telegram when B/C hits 2644. Does not print secrets."""
    if NOTIFY_FLAG.exists():
        _log("notify already sent; skipping")
        return
    if (
        not NOTIFY_PY
        or not NOTIFY_ENV
        or not NOTIFY_PY.is_file()
        or not NOTIFY_ENV.is_file()
    ):
        _log("notify skipped: set LIFEOS_NOTIFY_DIR to the LifeOS notify service")
        return
    text = (
        "NBA B and C finished (2644). D and E are not starting. "
        "You do not need to reply."
    )
    try:
        proc = subprocess.run(
            [
                "/bin/bash",
                "-lc",
                'set -a; source "$1"; set +a; python3 "$2" --imessage --telegram "$3"',
                "_",
                str(NOTIFY_ENV),
                str(NOTIFY_PY),
                text,
            ],
            cwd=str(NOTIFY_DIR),
            capture_output=True,
            text=True,
            timeout=90,
        )
        NOTIFY_FLAG.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
        out = (proc.stdout or "").strip() or (proc.stderr or "").strip()
        _log(f"notify exit={proc.returncode} {out[:200]}")
    except Exception as exc:  # noqa: BLE001 — B/C finish must still be recorded
        _log(f"notify failed: {type(exc).__name__}: {exc}")


def _claim_pidfile() -> None:
    PIDFILE.write_text(str(os.getpid()), encoding="utf-8")


def main() -> None:
    _claim_pidfile()
    _log(f"watch_bc pid={os.getpid()} checkpoint={CHECKPOINT}")
    while True:
        arms = _counts(CHECKPOINT)
        b, c = arms["B"], arms["C"]
        done = b >= TARGET_PER_ARM and c >= TARGET_PER_ARM
        _log(f"B={b}/{TARGET_PER_ARM} C={c}/{TARGET_PER_ARM} resume_bc={_bc_running()}")
        if done:
            break
        if not _bc_running():
            _start_bc()
            time.sleep(5)
            if not _bc_running():
                _log("resume_bc failed to stay up; retrying next poll")
        time.sleep(POLL_SECONDS)

    _log("B/C checkpoint complete; waiting for resume_bc to exit")
    while _bc_running():
        time.sleep(5)
    _notify_josh()
    _log("B/C complete. Not starting D/E.")


if __name__ == "__main__":
    main()
