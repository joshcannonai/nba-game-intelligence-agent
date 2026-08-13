"""Resume the B/C full-season eval without invalidating the checkpoint.

Arm B has no predict_win_probability tool. Gemma sometimes invents that call
anyway; the UI returns status=invalid_json and ui_agent_eval treats any bad
tool_result as a hard fail. Editing ui_agent_eval.py would change
evaluator_sha256 and discard the checkpoint. This wrapper only drops
tool_results for tools the arm was not given, then uses the same scoring path.

``--workers N`` (or ``BC_EVAL_WORKERS``) runs N games against the UI at once.
Ollama must be started with ``OLLAMA_NUM_PARALLEL`` >= N or the extra games
queue. The UI, prompt, tools, and evaluator file are unchanged.

    PYTHONPATH=vendor python -m eval.resume_bc --full-season --arms BC --workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eval import ui_agent_eval as m

_orig = m.assert_real_ui_result
_write_lock = threading.Lock()
_log_lock = threading.Lock()
_orig_append = m.append_checkpoint


def _assert_real_ui_result(
    result: dict,
    game_id: str,
    arm: str,
    cutoff: str,
    expected_prompt_sha256: str | None = None,
) -> None:
    required = set(m.REQUIRED_CALLS.get(arm, {}))
    patched = dict(result)
    patched["tool_results"] = [
        event
        for event in result.get("tool_results", [])
        if event.get("name") in required
    ]
    _orig(patched, game_id, arm, cutoff, expected_prompt_sha256)


def _append_checkpoint(path: Path, record: dict) -> None:
    with _write_lock:
        _orig_append(path, record)


def _log(message: str) -> None:
    with _log_lock:
        print(message, flush=True)


m.assert_real_ui_result = _assert_real_ui_result
m.append_checkpoint = _append_checkpoint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", default="http://127.0.0.1:8000")
    parser.add_argument("--arms", default="ABC")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--full-season",
        action="store_true",
        help="Run every canonical 2025-26 game with the previous day as cutoff.",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--errors", type=Path)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument(
        "--csv-every",
        type=int,
        default=50,
        help="Refresh the human-readable CSV after this many new checkpoint rows.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("BC_EVAL_WORKERS", "4")),
        help="Concurrent /api/run games. Keep this <= Ollama's OLLAMA_NUM_PARALLEL.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.full_season:
        args.checkpoint = args.checkpoint or m.DEFAULT_FULL_CHECKPOINT
        args.out = args.out or m.DEFAULT_FULL_CSV
        args.errors = args.errors or m.DEFAULT_FULL_ERRORS
    else:
        args.checkpoint = args.checkpoint or m.DEFAULT_SAMPLE_CHECKPOINT
        args.out = args.out or m.DEFAULT_SAMPLE_CSV
        args.errors = args.errors or m.DEFAULT_SAMPLE_ERRORS
    if not args.arms or any(arm not in "ABC" for arm in args.arms):
        parser.error("--arms must contain only A, B, and/or C")
    return args


def _setup(args: argparse.Namespace) -> dict:
    with (m.ROOT / "data" / "samples" / "game_logs_2026.csv").open(
        newline="", encoding="utf-8"
    ) as fh:
        season_games = tuple(row["game_id"] for row in csv.DictReader(fh))
    selected = season_games if args.full_season else m.SHARED_GAMES
    games = selected[: args.limit] if args.limit else selected
    eval_contract = m.eval_contract_for(args.full_season)
    with urllib.request.urlopen(f"{args.bridge}/api/health", timeout=30) as response:
        server_fingerprint = json.load(response)["runtime_fingerprint"]
    if (
        any(arm in args.arms for arm in "BC")
        and server_fingerprint["llm"]["digest"] == "unavailable"
    ):
        raise RuntimeError(
            "Ollama gemma4 digest is unavailable; start Ollama before evaluating B/C"
        )
    runtime_contracts = m.expected_runtime_contracts(eval_contract, server_fingerprint)
    done = _read_checkpoint(args.checkpoint, eval_contract, runtime_contracts)
    return {
        "games": games,
        "truth": m.game_truth(games),
        "fixed_cutoff": None if args.full_season else m.CUTOFF,
        "eval_contract": eval_contract,
        "server_fingerprint": server_fingerprint,
        "runtime_contracts": runtime_contracts,
        "done": done,
        "odds": m.load_odds(),
        "total": len(games) * len(args.arms),
    }


def _read_checkpoint(
    path: Path,
    eval_contract: str,
    runtime_contracts: dict[str, dict[str, str]],
) -> dict[tuple[str, str], dict]:
    """Load completed games even if Ollama was restarted for more parallel slots.

    ui_agent_eval hashes the live /api/show digest into runtime_contract_sha256.
    Restarting Ollama can change that wrapper hash without changing gemma4 weights,
    which would otherwise make the strict reader treat 800+ scored games as missing.
    Prompt, evaluator file, and eval_contract still have to match.
    """
    strict = m.read_checkpoint(path, eval_contract, runtime_contracts)
    if not path.exists():
        return strict
    relaxed: dict[tuple[str, str], dict] = {}
    hashes: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("eval_contract") != eval_contract:
            continue
        arm = row.get("arm")
        expected = runtime_contracts.get(arm)
        if not expected:
            continue
        if row.get("system_prompt_sha256") != expected["system_prompt_sha256"]:
            continue
        relaxed[(arm, row["game_id"])] = row
        hashes.add(str(row.get("server_runtime_sha256") or ""))
    if len(relaxed) != len(strict):
        _log(
            f"checkpoint: strict={len(strict)} relaxed={len(relaxed)} "
            f"server_hashes={sorted(hashes)}"
        )
    return relaxed if relaxed else strict


def _run_one(job: dict) -> dict:
    result, attempt_count = m.run_with_retry(
        job["bridge"],
        job["game_id"],
        job["arm"],
        job["cutoff"],
        max_attempts=job["max_attempts"],
        retry_delay=job["retry_delay"],
        errors_path=job["errors_path"],
        expected_prompt_sha256=job["expected_prompt_sha256"],
    )
    if result["server_runtime_sha256"] != job["server_runtime_sha256"]:
        raise RuntimeError(
            "server runtime changed during evaluation; refusing to mix rows"
        )
    record = m.score_record(
        result,
        job["row"],
        job["arm"],
        job["cutoff"],
        job["odds"],
        job["eval_contract"],
        job["runtime_contract_sha256"],
    )
    record["attempt_count"] = attempt_count
    m.append_checkpoint(job["checkpoint"], record)
    return record


def main() -> None:
    args = _parse_args()
    ctx = _setup(args)
    games = ctx["games"]
    done = ctx["done"]
    total = ctx["total"]
    pending: list[dict] = []
    skipped = 0
    for arm in args.arms:
        for game_id in games:
            key = (arm, game_id)
            if key in done:
                skipped += 1
                continue
            row = ctx["truth"][game_id]
            pending.append(
                {
                    "index": skipped + len(pending) + 1,
                    "arm": arm,
                    "game_id": game_id,
                    "row": row,
                    "cutoff": m.cutoff_for_game(row, fixed_cutoff=ctx["fixed_cutoff"]),
                    "bridge": args.bridge,
                    "max_attempts": args.max_attempts,
                    "retry_delay": args.retry_delay,
                    "errors_path": args.errors,
                    "expected_prompt_sha256": ctx["runtime_contracts"][arm][
                        "system_prompt_sha256"
                    ],
                    "server_runtime_sha256": ctx["server_fingerprint"]["sha256"],
                    "odds": ctx["odds"],
                    "eval_contract": ctx["eval_contract"],
                    "runtime_contract_sha256": ctx["runtime_contracts"][arm][
                        "runtime_contract_sha256"
                    ],
                    "checkpoint": args.checkpoint,
                }
            )
    _log(
        f"resume_bc workers={args.workers} skipped={skipped} pending={len(pending)} "
        f"total={total}"
    )
    newly_completed = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, job): job for job in pending}
        for fut in as_completed(futures):
            job = futures[fut]
            record = fut.result()
            key = (job["arm"], job["game_id"])
            with _write_lock:
                done[key] = record
                newly_completed += 1
                snapshot = newly_completed
                rows = list(done.values())
            if args.csv_every > 0 and snapshot % args.csv_every == 0:
                m.write_csv(args.out, rows)
            elapsed = time.time() - started
            rate = snapshot / elapsed if elapsed else 0.0
            remaining = len(pending) - snapshot
            eta_min = (remaining / rate) / 60 if rate else 0.0
            _log(
                f"[{skipped + snapshot}/{total}] {job['arm']} {job['game_id']} "
                f"p_home={record['home_win_prob']:.3f} pick={record['predicted_winner']} "
                f"correct={record['correct']} tools={record['tool_call_count']} "
                f"run={record['elapsed_seconds']:.1f}s wall={elapsed / 60:.1f}m "
                f"{rate * 60:.1f}/10m eta={eta_min:.1f}m"
            )
    ordered = [done[(arm, game)] for arm in args.arms for game in games]
    m.write_csv(args.out, ordered)
    _log(f"wrote {args.out} ({len(ordered)} actual UI rows)")


if __name__ == "__main__":
    main()
