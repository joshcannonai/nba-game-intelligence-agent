"""Evaluate the actual UI endpoints on a shared set or the complete season.

This runner does not reproduce the agent or its prompt rule in Python. Arm A calls
``/api/predict``. Arms B and C call ``/api/run`` and consume
the same SSE stream the React UI consumes, including the configured Ollama model,
SKILL.md system-prompt injection, tool calls, and server-generated gate receipts.

The ten-game comparison uses the shared 2026-04-05 cutoff specified by Sarvesh.
The full-season mode gives each game the previous calendar day as its cutoff.
Results and betting prices are joined only after the UI has returned its final
prediction. The JSONL checkpoint is append-only so a long local-model run can be
resumed without silently replacing completed generations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

from eval.betting import fair_home_prob, load_odds, odds_for_game, priced

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_CHECKPOINT = ROOT / "eval" / "results_actual_ui_10.jsonl"
DEFAULT_SAMPLE_CSV = ROOT / "eval" / "results_actual_ui_10.csv"
DEFAULT_SAMPLE_ERRORS = ROOT / "eval" / "results_actual_ui_10_errors.jsonl"
DEFAULT_FULL_CHECKPOINT = ROOT / "eval" / "results_actual_ui_full_season.jsonl"
DEFAULT_FULL_CSV = ROOT / "eval" / "results_actual_ui_full_season.csv"
DEFAULT_FULL_ERRORS = ROOT / "eval" / "results_actual_ui_full_season_errors.jsonl"
CUTOFF = "2026-04-05"
MODEL_BACKEND = "ollama"
AGENT_REASONING_MODE = "disabled"
FULL_SEASON_CONTRACT = "actual-ui-full-season-previous-day-v3"
SAMPLE_CONTRACT = "actual-ui-shared-10-fixed-2026-04-05-v3"

# Exact games sent by Sarvesh, resolved against the canonical 2025-26 game log.
SHARED_GAMES = (
    "CHO-BOS-2026-04-07",
    "ATL-CLE-2026-04-08",
    "IND-BRK-2026-04-09",
    "ORL-CHI-2026-04-10",
    "ATL-MIA-2026-04-12",
    "MIA-CHO-2026-04-14",
    "GSW-LAC-2026-04-15",
    "CHO-ORL-2026-04-17",
    "ATL-NYK-2026-04-18",
    "MIN-DEN-2026-04-20",
)

REQUIRED_CALLS = {
    "B": {
        "retrieve_matchup_context": 1,
        "retrieve_team_form": 2,
        "retrieve_injuries": 2,
    },
    "C": {
        "retrieve_matchup_context": 1,
        "retrieve_team_form": 2,
        "retrieve_injuries": 2,
        "predict_win_probability": 1,
    },
}


def eval_contract_for(full_season: bool) -> str:
    return FULL_SEASON_CONTRACT if full_season else SAMPLE_CONTRACT


def _files_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def expected_runtime_contracts(
    eval_contract: str, server_fingerprint: dict | None = None
) -> dict[str, dict[str, str]]:
    """Fingerprint the exact evaluator, prompt, server, datasets, and model."""
    from agent.run import system_prompt_for
    from agent.sources import get_source
    from agent.tools import build_tools

    common = [
        ROOT / "eval" / "ui_agent_eval.py",
    ]
    source = get_source("real")
    contracts = {}
    for arm in "ABC":
        prompt_sha = "not_applicable"
        if arm in "BC":
            include_model = arm == "C"
            names = [
                tool.name for tool in build_tools(source, include_model=include_model)
            ]
            prompt = system_prompt_for(names, include_model)
            prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        payload = {
            "arm": arm,
            "eval_contract": eval_contract,
            "model_backend": "none" if arm == "A" else MODEL_BACKEND,
            "language_model": "none" if arm == "A" else "gemma4",
            "agent_reasoning_mode": (
                "not_applicable" if arm == "A" else AGENT_REASONING_MODE
            ),
            "system_prompt_sha256": prompt_sha,
            "evaluator_sha256": _files_sha256(common),
            "server_runtime_fingerprint": server_fingerprint or {},
        }
        contracts[arm] = {
            "system_prompt_sha256": prompt_sha,
            "runtime_contract_sha256": hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest(),
        }
    return contracts


def game_truth(game_ids: tuple[str, ...] | list[str]) -> dict[str, dict[str, str]]:
    path = ROOT / "data" / "samples" / "game_logs_2026.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        rows = {row["game_id"]: row for row in csv.DictReader(fh)}
    missing = set(game_ids) - set(rows)
    if missing:
        raise RuntimeError(
            f"shared games missing from canonical log: {sorted(missing)}"
        )
    return rows


def post_json(url: str, payload: dict, *, timeout: int = 900):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=timeout)


def parse_final_json(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.S)
        if not match:
            raise ValueError(f"agent final did not contain JSON: {text[:300]}")
        payload = json.loads(match.group(0))
    return payload


def cutoff_for_game(truth: dict[str, str], *, fixed_cutoff: str | None) -> str:
    if fixed_cutoff:
        return fixed_cutoff
    game_date = date.fromisoformat(truth["game_date"])
    return (game_date - timedelta(days=1)).isoformat()


def run_a(bridge: str, game_id: str, cutoff: str) -> dict:
    with post_json(
        f"{bridge}/api/predict",
        {"matchup_id": game_id, "as_of_date": cutoff},
    ) as response:
        payload = json.load(response)
    with urllib.request.urlopen(f"{bridge}/api/health", timeout=30) as response:
        health = json.load(response)
    return {
        "final": payload,
        "tool_calls": [],
        "gate_receipts": [],
        "usage": {},
        "elapsed_seconds": 0.0,
        "system_prompt_sha256": "not_applicable",
        "system_prompt_chars": 0,
        "server_runtime_sha256": health["runtime_fingerprint"]["sha256"],
    }


def run_agent(bridge: str, game_id: str, arm: str, cutoff: str) -> dict:
    include_model = arm == "C"
    events: list[dict] = []
    current_event = "message"
    prompt_text = ""
    final_text = ""
    server_runtime_sha256 = ""
    with post_json(
        f"{bridge}/api/run",
        {
            "matchup_id": game_id,
            "as_of_date": cutoff,
            "include_model": include_model,
            "model_backend": MODEL_BACKEND,
        },
    ) as response:
        for raw in response:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("event: "):
                current_event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                events.append({"event": current_event, "data": data})
                if current_event == "context_start":
                    messages = data.get("messages") or []
                    if messages and messages[0].get("role") == "system":
                        prompt_text = str(messages[0].get("content") or "")
                elif current_event == "start":
                    server_runtime_sha256 = str(
                        data.get("runtime_fingerprint", {}).get("sha256") or ""
                    )
                elif current_event == "final":
                    final_text = str(data.get("content") or "")
                elif current_event == "error":
                    raise RuntimeError(data.get("message") or "UI agent failed")

    final = parse_final_json(final_text)
    calls = [e["data"] for e in events if e["event"] == "tool_call"]
    tool_results = [e["data"] for e in events if e["event"] == "tool_result"]
    receipts = [e["data"] for e in events if e["event"] == "gate_receipt"]
    context = [e["data"] for e in events if e["event"] == "context_message"]
    usage = context[-1].get("usage", {}) if context else {}
    elapsed = max((float(e["data"].get("elapsed", 0)) for e in events), default=0.0)
    return {
        "final": final,
        "tool_calls": calls,
        "tool_results": tool_results,
        "gate_receipts": receipts,
        "usage": usage,
        "elapsed_seconds": elapsed,
        "system_prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
        "system_prompt_chars": len(prompt_text),
        "server_runtime_sha256": server_runtime_sha256,
    }


def _expected_required_calls(
    game_id: str, arm: str, cutoff: str
) -> list[tuple[str, dict]]:
    away, home = game_id.split("-")[:2]
    calls = [
        ("retrieve_matchup_context", {"matchup_id": game_id, "as_of_date": cutoff}),
        ("retrieve_team_form", {"team_abbr": away, "as_of_date": cutoff, "last_n": 10}),
        ("retrieve_team_form", {"team_abbr": home, "as_of_date": cutoff, "last_n": 10}),
        ("retrieve_injuries", {"team_abbr": away, "as_of_date": cutoff}),
        ("retrieve_injuries", {"team_abbr": home, "as_of_date": cutoff}),
    ]
    if arm == "C":
        calls.append(
            ("predict_win_probability", {"matchup_id": game_id, "as_of_date": cutoff})
        )
    return calls


def assert_real_ui_result(
    result: dict,
    game_id: str,
    arm: str,
    cutoff: str,
    expected_prompt_sha256: str | None = None,
) -> None:
    final = result["final"]
    if final.get("status") not in (None, "ok"):
        raise AssertionError(f"non-ok result for {arm} {game_id}: {final}")
    if final.get("matchup_id") != game_id or final.get("as_of_date") != cutoff:
        raise AssertionError(f"identity drift for {arm} {game_id}: {final}")
    home = float(final["home_win_prob"])
    away = float(final["away_win_prob"])
    if not (0 <= home <= 1 and 0 <= away <= 1 and abs(home + away - 1) < 1e-6):
        raise AssertionError(
            f"invalid probabilities for {arm} {game_id}: {home}, {away}"
        )
    if arm == "A":
        return
    if (
        expected_prompt_sha256
        and result["system_prompt_sha256"] != expected_prompt_sha256
    ):
        raise AssertionError(f"runtime prompt drift for {arm} {game_id}")

    bad_results = []
    for result_event in result.get("tool_results", []):
        status = result_event.get("status")
        allowed = status == "ok" or (
            result_event.get("name") == "retrieve_team_form"
            and status == "awaiting_input"
        )
        if not allowed:
            bad_results.append((result_event.get("name"), status))
    if bad_results:
        raise AssertionError(
            f"required tool result unavailable for {arm} {game_id}: {bad_results}"
        )

    expected_calls = _expected_required_calls(game_id, arm, cutoff)
    for name in REQUIRED_CALLS[arm]:
        actual_args = [
            dict(call.get("args", {}))
            for call in result["tool_calls"]
            if call.get("name") == name
        ]
        if name == "retrieve_team_form":
            for args in actual_args:
                args.setdefault("last_n", 10)
        wanted_args = [args for tool, args in expected_calls if tool == name]
        if sorted(
            actual_args, key=lambda item: json.dumps(item, sort_keys=True)
        ) != sorted(wanted_args, key=lambda item: json.dumps(item, sort_keys=True)):
            raise AssertionError(
                f"wrong {name} calls for {arm} {game_id}: {actual_args}; expected {wanted_args}"
            )

        receipts = [
            receipt
            for receipt in result["gate_receipts"]
            if receipt.get("tool") == name
            and receipt.get("requested_cutoff") == cutoff
            and receipt.get("tool_cutoff") == cutoff
            and receipt.get("status") == "passed"
        ]
        if len(receipts) != len(wanted_args):
            raise AssertionError(
                f"missing passed gate receipts for {arm} {game_id} {name}: "
                f"{len(receipts)}/{len(wanted_args)}"
            )


def score_record(
    result: dict,
    truth: dict[str, str],
    arm: str,
    cutoff: str,
    odds: dict[str, dict],
    eval_contract: str,
    runtime_contract_sha256: str,
) -> dict:
    final = result["final"]
    home_probability = float(final["home_win_prob"])
    predicted = truth["home"] if home_probability >= 0.5 else truth["away"]
    odds_row = odds_for_game(truth["game_id"], odds)
    market_home = fair_home_prob(odds_row) if odds_row else None
    selected_decimal = None
    pnl = None
    if market_home is not None:
        selected_decimal = priced(
            market_home if predicted == truth["home"] else 1 - market_home
        )
        pnl = (
            round(100 * (selected_decimal - 1), 2)
            if predicted == truth["winner"]
            else -100.0
        )
    return {
        "arm": arm,
        "eval_contract": eval_contract,
        "runtime_contract_sha256": runtime_contract_sha256,
        "game_id": truth["game_id"],
        "game_date": truth["game_date"],
        "away": truth["away"],
        "home": truth["home"],
        "cutoff": cutoff,
        "language_model": "none" if arm == "A" else "gemma4:latest via Ollama",
        "agent_reasoning_mode": "not_applicable"
        if arm == "A"
        else AGENT_REASONING_MODE,
        "home_win_prob": home_probability,
        "predicted_winner": predicted,
        "actual_winner": truth["winner"],
        "correct": int(predicted == truth["winner"]),
        "tool_call_count": len(result["tool_calls"]),
        "tool_calls": [call["name"] for call in result["tool_calls"]],
        "gate_passed": sum(
            r.get("status") == "passed" for r in result["gate_receipts"]
        ),
        "gate_failed": sum(
            r.get("status") == "failed" for r in result["gate_receipts"]
        ),
        "system_prompt_sha256": result["system_prompt_sha256"],
        "system_prompt_chars": result["system_prompt_chars"],
        "input_tokens": int(result["usage"].get("input_tokens", 0) or 0),
        "output_tokens": int(result["usage"].get("output_tokens", 0) or 0),
        "elapsed_seconds": result["elapsed_seconds"],
        "market_home_prob": market_home,
        "reconstructed_decimal_odds": selected_decimal,
        "odds_provenance": "derived from pre-tip closing spread; not a quoted moneyline",
        "stake": 100 if selected_decimal is not None else None,
        "net_pnl": pnl,
        "key_factors": final.get("key_factors", []),
        "narrative": final.get("narrative", ""),
        "gate_receipts": result["gate_receipts"],
        "server_runtime_sha256": result["server_runtime_sha256"],
        "final_json": final,
    }


def read_checkpoint(
    path: Path,
    eval_contract: str,
    runtime_contracts: dict[str, dict[str, str]],
) -> dict[tuple[str, str], dict]:
    done = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("eval_contract") != eval_contract:
                continue
            arm = row.get("arm")
            expected = runtime_contracts.get(arm)
            if (
                not expected
                or row.get("runtime_contract_sha256")
                != expected["runtime_contract_sha256"]
            ):
                continue
            if row.get("system_prompt_sha256") != expected["system_prompt_sha256"]:
                continue
            done[(arm, row["game_id"])] = row
    return done


def append_checkpoint(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def run_with_retry(
    bridge: str,
    game_id: str,
    arm: str,
    cutoff: str,
    *,
    max_attempts: int,
    retry_delay: float,
    errors_path: Path,
    expected_prompt_sha256: str,
) -> tuple[dict, int]:
    for attempt in range(1, max_attempts + 1):
        try:
            result = (
                run_a(bridge, game_id, cutoff)
                if arm == "A"
                else run_agent(bridge, game_id, arm, cutoff)
            )
            assert_real_ui_result(
                result,
                game_id,
                arm,
                cutoff,
                expected_prompt_sha256,
            )
            return result, attempt
        except Exception as exc:
            append_checkpoint(
                errors_path,
                {
                    "arm": arm,
                    "game_id": game_id,
                    "cutoff": cutoff,
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            if attempt == max_attempts:
                raise
            print(
                f"  attempt {attempt}/{max_attempts} rejected: {type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(retry_delay)
    raise AssertionError("retry loop exhausted without returning or raising")


def write_csv(path: Path, records: list[dict]) -> None:
    fields = [
        "arm",
        "eval_contract",
        "runtime_contract_sha256",
        "game_id",
        "game_date",
        "away",
        "home",
        "cutoff",
        "language_model",
        "agent_reasoning_mode",
        "home_win_prob",
        "predicted_winner",
        "actual_winner",
        "correct",
        "tool_call_count",
        "tool_calls",
        "gate_passed",
        "gate_failed",
        "system_prompt_sha256",
        "system_prompt_chars",
        "input_tokens",
        "output_tokens",
        "elapsed_seconds",
        "market_home_prob",
        "reconstructed_decimal_odds",
        "odds_provenance",
        "stake",
        "net_pnl",
        "key_factors",
        "narrative",
        "gate_receipts",
        "server_runtime_sha256",
        "final_json",
        "attempt_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: json.dumps(record[key], ensure_ascii=False)
                    if isinstance(record.get(key), (dict, list))
                    else record.get(key)
                    for key in fields
                }
            )


def main() -> None:
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
    args = parser.parse_args()

    if args.full_season:
        args.checkpoint = args.checkpoint or DEFAULT_FULL_CHECKPOINT
        args.out = args.out or DEFAULT_FULL_CSV
        args.errors = args.errors or DEFAULT_FULL_ERRORS
    else:
        args.checkpoint = args.checkpoint or DEFAULT_SAMPLE_CHECKPOINT
        args.out = args.out or DEFAULT_SAMPLE_CSV
        args.errors = args.errors or DEFAULT_SAMPLE_ERRORS

    if not args.arms or any(arm not in "ABC" for arm in args.arms):
        parser.error("--arms must contain only A, B, and/or C")

    with (ROOT / "data" / "samples" / "game_logs_2026.csv").open(
        newline="", encoding="utf-8"
    ) as fh:
        season_games = tuple(row["game_id"] for row in csv.DictReader(fh))
    selected = season_games if args.full_season else SHARED_GAMES
    games = selected[: args.limit] if args.limit else selected
    truth = game_truth(games)
    fixed_cutoff = None if args.full_season else CUTOFF
    eval_contract = eval_contract_for(args.full_season)
    with urllib.request.urlopen(f"{args.bridge}/api/health", timeout=30) as response:
        server_fingerprint = json.load(response)["runtime_fingerprint"]
    if (
        any(arm in args.arms for arm in "BC")
        and server_fingerprint["llm"]["digest"] == "unavailable"
    ):
        raise RuntimeError(
            "Ollama gemma4 digest is unavailable; start Ollama before evaluating B/C"
        )
    runtime_contracts = expected_runtime_contracts(eval_contract, server_fingerprint)
    done = read_checkpoint(args.checkpoint, eval_contract, runtime_contracts)
    odds = load_odds()
    total = len(games) * len(args.arms)
    completed = 0
    newly_completed = 0
    started = time.time()
    for arm in args.arms:
        for game_id in games:
            completed += 1
            row = truth[game_id]
            cutoff = cutoff_for_game(row, fixed_cutoff=fixed_cutoff)
            key = (arm, game_id)
            if key in done:
                print(f"[{completed}/{total}] resume {arm} {game_id}", flush=True)
                continue
            print(f"[{completed}/{total}] UI run {arm} {game_id}", flush=True)
            result, attempt_count = run_with_retry(
                args.bridge,
                game_id,
                arm,
                cutoff,
                max_attempts=args.max_attempts,
                retry_delay=args.retry_delay,
                errors_path=args.errors,
                expected_prompt_sha256=runtime_contracts[arm]["system_prompt_sha256"],
            )
            if result["server_runtime_sha256"] != server_fingerprint["sha256"]:
                raise RuntimeError(
                    "server runtime changed during evaluation; refusing to mix rows"
                )
            record = score_record(
                result,
                row,
                arm,
                cutoff,
                odds,
                eval_contract,
                runtime_contracts[arm]["runtime_contract_sha256"],
            )
            record["attempt_count"] = attempt_count
            append_checkpoint(args.checkpoint, record)
            done[key] = record
            newly_completed += 1
            if args.csv_every > 0 and newly_completed % args.csv_every == 0:
                write_csv(args.out, list(done.values()))
            elapsed = time.time() - started
            print(
                f"  p_home={record['home_win_prob']:.3f} pick={record['predicted_winner']} "
                f"correct={record['correct']} tools={record['tool_call_count']} "
                f"run={record['elapsed_seconds']:.1f}s wall={elapsed / 60:.1f}m",
                flush=True,
            )
    ordered = [done[(arm, game)] for arm in args.arms for game in games]
    write_csv(args.out, ordered)
    print(f"wrote {args.out} ({len(ordered)} actual UI rows)")


if __name__ == "__main__":
    main()
