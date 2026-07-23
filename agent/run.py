"""LangChain tool-calling analyst agent for a single NBA matchup.

Two chat model backends, picked with --model:

  anthropic (default)  Claude, personal API credits. Fast iteration while
                       building -- this is "build mode" from the README.
  ollama               Local Gemma 4 via Ollama, no API key, no cost. This is
                       the leakage-safe path: Claude's training cutoff isn't
                       something we can pin to a date the way an open model's
                       release date is, so replay/production runs (testing
                       against real past games) need a model whose cutoff we
                       can actually verify predates the test window. Requires
                       `ollama pull gemma4` once and the Ollama server running.

    python -m agent.run --dry-run                      # no API key, mock data
    python -m agent.run --dry-run --source real \
        --matchup LAL-BOS-2024-12-25 --as-of 2024-12-24
    python -m agent.run --source real --matchup ... --as-of ...            # Claude
    python -m agent.run --model ollama --source real --matchup ... --as-of ...  # Gemma 4
    python -m agent.run --status --source real         # which tools have data, and where the rest comes from
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Allow `python -m agent.run` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.sources import get_source  # noqa: E402
from agent.tools import build_tools  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

SYSTEM = """You are the NBA Game Intelligence analyst agent.
Given a matchup_id and as_of_date, use your tools to gather context and a win
probability, then write a short structured pregame report.

Rules:
- Always call retrieve_matchup_context first.
- If either team is on a back-to-back, call retrieve_player_splits with
  back_to_back=true for that team's key players.
- Always call predict_win_probability.
- The report we owe the user is: who wins, the likely best player, a narrative,
  statistics, and the betting line. So also attempt retrieve_betting_line,
  predict_best_player, and retrieve_news, plus predict_stat_line for a key
  player. Attempt them even though some are not built yet -- a tool that
  answers "awaiting_input" is how we learn what is still blocking the report.
- SOME TOOLS HAVE NO DATA YET. A tool may return {"status": "awaiting_input",
  "needs_from": ..., "needs": ...}. That is not an error and not an empty result.
  It means the data layer for it does not exist. When that happens, add a line to
  "missing" naming the tool and where its input comes from, then carry on.
- Tool output may also contain nulls with an "unavailable" or "warnings" note.
  Same rule. Never fill a gap with a guess. Never treat a null or a missing tool
  as zero. An unknown injury list is not "nobody is hurt".
- Final answer must be valid JSON with keys:
  matchup_id, as_of_date, home_win_prob, away_win_prob, key_factors (list of
  short strings), missing (list of "tool_name -- needs_from -- what it needs"),
  narrative (2-4 sentences).
- Do not invent stats that tools did not return.
"""


def build_agent(source, model_backend: str = "anthropic"):
    from langchain.agents import create_agent

    if model_backend == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit(
                "ANTHROPIC_API_KEY missing. Copy .env.example to .env and set your "
                "key, or run with --dry-run (no LLM), or --model ollama for the "
                "local Gemma 4 path (no API key needed)."
            )
        model = ChatAnthropic(model="claude-sonnet-4-5", api_key=api_key, temperature=0)
    elif model_backend == "ollama":
        from langchain_ollama import ChatOllama

        # Local, free, and -- the actual point -- a training cutoff we can
        # verify predates the games we're testing on. See module docstring.
        model = ChatOllama(model="gemma4", temperature=0)
    else:
        raise ValueError(f"unknown model backend: {model_backend!r}")

    return create_agent(model, build_tools(source), system_prompt=SYSTEM)


def run_matchup(
    matchup_id: str, as_of_date: str, source, model_backend: str = "anthropic"
) -> str:
    agent = build_agent(source, model_backend)
    user = (
        f"Produce a pregame report for matchup_id={matchup_id} as_of_date={as_of_date}."
    )
    result = agent.invoke({"messages": [{"role": "user", "content": user}]})
    messages = result.get("messages", [])
    if not messages:
        return json.dumps({"error": "no agent messages"}, indent=2)
    content = getattr(messages[-1], "content", messages[-1])
    if isinstance(content, list):
        parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
        content = "\n".join(p for p in parts if p)
    return str(content)


def _probe_args(matchup_id: str, as_of_date: str) -> dict[str, dict]:
    """Representative arguments for every tool, so each one can be called once."""
    away, home = matchup_id.split("-")[0], matchup_id.split("-")[1]
    player = "LeBron James"
    return {
        "retrieve_matchup_context": {
            "matchup_id": matchup_id,
            "as_of_date": as_of_date,
        },
        "retrieve_player_splits": {"player_name": player, "back_to_back": True},
        "retrieve_schedule": {"as_of_date": as_of_date, "days_ahead": 1},
        "retrieve_team_form": {
            "team_abbr": home,
            "as_of_date": as_of_date,
            "last_n": 10,
        },
        "retrieve_injuries": {"team_abbr": home, "as_of_date": as_of_date},
        "retrieve_news": {"team_abbr": home, "as_of_date": as_of_date, "limit": 5},
        "retrieve_betting_line": {"matchup_id": matchup_id, "as_of_date": as_of_date},
        "predict_win_probability": {
            "home_abbr": home,
            "away_abbr": away,
            "as_of_date": as_of_date,
        },
        "predict_stat_line": {
            "player_name": player,
            "matchup_id": matchup_id,
            "as_of_date": as_of_date,
        },
        "predict_best_player": {"matchup_id": matchup_id, "as_of_date": as_of_date},
    }


def status_board(matchup_id: str, as_of_date: str, source) -> str:
    """Which tools return data, which are stubs, and where the rest is waiting on.

    Deterministic and free: calls every tool once and reports what came back.
    All ten tools are the agent lane's; what varies is whether the data or model
    behind each one exists. So this doubles as the project's input-blocking list.
    """
    tools = {t.name: t for t in build_tools(source)}
    args = _probe_args(matchup_id, as_of_date)

    built, stubbed, missing = [], [], []
    for name, tool in tools.items():
        try:
            payload = json.loads(tool.invoke(args[name]))
        except Exception as exc:  # a tool that raises is a gap too, not a crash
            missing.append((name, "unknown", f"raised {type(exc).__name__}: {exc}"))
            continue

        if payload.get("status") == "awaiting_input":
            missing.append(
                (name, payload.get("needs_from", "?"), payload.get("needs", ""))
            )
        elif payload.get("warning") or str(payload.get("model", "")).startswith(
            "stub_"
        ):
            stubbed.append((name, payload.get("warning", "placeholder logic")))
        else:
            built.append(name)

    total = len(tools)
    lines = [
        "NBA Game Intelligence Agent -- tool status",
        f"source={source.name}  matchup={matchup_id}  as_of={as_of_date}",
        "",
        f"All {total} tools are written and callable (agent lane). What follows is",
        "whether the data or model behind each one exists yet.",
        "",
        f"RETURNING REAL DATA ({len(built)}/{total})",
    ]
    lines += [f"  {n}" for n in built] or ["  (none)"]

    lines += [
        "",
        f"PLACEHOLDER LOGIC ({len(stubbed)}/{total}) -- runs, but not the real model",
    ]
    for name, warning in stubbed:
        lines += [f"  {name}", f"      {warning}"]
    if not stubbed:
        lines.append("  (none)")

    lines += [
        "",
        f"AWAITING INPUT ({len(missing)}/{total}) -- tool is ready, data is not:",
    ]
    for name, needs_from, needs in missing:
        lines += [f"  {name}  --  input from: {needs_from}", f"      needs: {needs}"]
    if not missing:
        lines.append("  (none)")

    by_source: dict[str, list[str]] = {}
    for name, needs_from, _ in missing:
        # "Josh (have the data)" and "Josh (scope-cut candidate)" are one person
        by_source.setdefault(needs_from.split(" (")[0], []).append(name)
    if by_source:
        lines += ["", "INPUTS WE ARE WAITING ON"]
        for who, names in sorted(by_source.items()):
            lines.append(f"  {who}: {', '.join(names)}")

    return "\n".join(lines)


def dry_run(matchup_id: str, as_of_date: str, source) -> str:
    """No API key needed. Exercises the real tool contracts and data gating.

    This is the deterministic path: same tools, same date filtering, no LLM. It
    proves the data layer without spending a token.
    """
    tools = {t.name: t for t in build_tools(source)}

    ctx = json.loads(
        tools["retrieve_matchup_context"].invoke(
            {"matchup_id": matchup_id, "as_of_date": as_of_date}
        )
    )
    pred = json.loads(
        tools["predict_win_probability"].invoke(
            {
                "home_abbr": ctx["home_team"]["abbr"],
                "away_abbr": ctx["away_team"]["abbr"],
                "as_of_date": as_of_date,
            }
        )
    )

    key_factors: list[str] = []
    home, away = ctx["home_team"], ctx["away_team"]
    if home.get("off_rating") and away.get("off_rating"):
        home_net = home["off_rating"] - home["def_rating"]
        away_net = away["off_rating"] - away["def_rating"]
        key_factors.append(
            f"{home['abbr']} net rating {home_net:+.1f} vs {away['abbr']} {away_net:+.1f}"
            f" ({home.get('basis', 'mock')})"
        )

    rest = ctx.get("rest", {})
    if rest.get("unavailable"):
        key_factors.append("Rest/back-to-back: UNAVAILABLE (no game-log dataset yet)")
    else:
        b2b = [
            abbr
            for abbr, flag in (
                (home["abbr"], rest.get("home_back_to_back")),
                (away["abbr"], rest.get("away_back_to_back")),
            )
            if flag
        ]

        # None means no prior game in the log (season opener), not zero days rest
        def _rest(days) -> str:
            return f"{days}d" if days is not None else "no prior game"

        key_factors.append(
            f"Rest: {home['abbr']} {_rest(rest.get('home_days_rest'))}, "
            f"{away['abbr']} {_rest(rest.get('away_days_rest'))}"
            + (f" -- {', '.join(b2b)} on a back-to-back" if b2b else "")
        )

    h2h = ctx.get("h2h_last_5", [])
    if h2h:
        wins = {}
        for g in h2h:
            wins[g["winner"]] = wins.get(g["winner"], 0) + 1
        tally = ", ".join(f"{abbr} {n}" for abbr, n in wins.items())
        key_factors.append(f"H2H last {len(h2h)} (before {as_of_date}): {tally}")

    injuries = ctx.get("injuries", [])
    if injuries:
        shown = ", ".join(f"{i['player']} ({i['team']})" for i in injuries[:3])
        key_factors.append(f"{len(injuries)} out as of {as_of_date}: {shown}")
    else:
        key_factors.append(f"No players listed out as of {as_of_date}")

    for w in ctx.get("warnings", []):
        key_factors.append(f"WARNING: {w}")

    prob = pred.get("home_win_prob")
    narrative = (
        f"{away['abbr']} at {home['abbr']} on {ctx['game_date']}, as of {as_of_date}. "
        + (
            f"Home win probability {prob:.1%} from the net-rating stub (not XGBoost yet). "
            if prob is not None
            else "No win probability: the stub could not find ratings for both teams. "
        )
        + f"Injury list is date-gated: {len(injuries)} player(s) known out that morning. "
        + (
            "Rest and head-to-head are missing because the schedule/game-log dataset "
            "is not on main yet."
            if rest.get("unavailable")
            else "Rest and head-to-head computed from game logs."
        )
    )

    # Same "missing" contract the LLM path is held to (see SYSTEM), computed
    # deterministically: every tool that is not built, and who owns it.
    tools = {t.name: t for t in build_tools(source)}
    probes = _probe_args(matchup_id, as_of_date)
    missing = []
    for name, tool in tools.items():
        try:
            payload = json.loads(tool.invoke(probes[name]))
        except Exception:
            continue
        if payload.get("status") == "awaiting_input":
            missing.append(
                f"{name} -- {payload.get('needs_from', '?')} -- {payload.get('needs', '')}"
            )

    return json.dumps(
        {
            "matchup_id": ctx["matchup_id"],
            "as_of_date": as_of_date,
            "source": ctx["source"],
            "home_win_prob": pred.get("home_win_prob"),
            "away_win_prob": pred.get("away_win_prob"),
            "key_factors": key_factors,
            "missing": missing,
            "narrative": narrative,
            "mode": "dry_run_no_llm",
        },
        indent=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="NBA Game Intelligence agent")
    parser.add_argument(
        "--matchup",
        default="LAL-BOS-2026-01-15",
        help="Matchup id AWAY-HOME-YYYY-MM-DD (default: the mock fixture)",
    )
    parser.add_argument(
        "--as-of",
        default="2026-01-14",
        help="As-of date YYYY-MM-DD (default: day before tip)",
    )
    parser.add_argument(
        "--source",
        choices=["mock", "real"],
        default="mock",
        help="mock = fixture (deterministic); real = date-gated CSVs on main",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exercise tools + print a report without calling an LLM",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print which tools have data, and where the missing inputs come from",
    )
    parser.add_argument(
        "--model",
        choices=["anthropic", "ollama"],
        default="anthropic",
        help=(
            "anthropic = Claude, needs ANTHROPIC_API_KEY (build mode); "
            "ollama = local Gemma 4, no API key, needs `ollama pull gemma4` "
            "and the Ollama server running (the leakage-safe path -- see "
            "module docstring). Ignored with --dry-run, which never calls an LLM."
        ),
    )
    args = parser.parse_args()

    source = get_source(args.source)
    if args.status:
        print(status_board(args.matchup, args.as_of, source))
    elif args.dry_run:
        print(dry_run(args.matchup, args.as_of, source))
    else:
        print(run_matchup(args.matchup, args.as_of, source, args.model))


if __name__ == "__main__":
    main()
