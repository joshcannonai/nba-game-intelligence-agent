"""LangChain tool-calling analyst agent for a single NBA matchup.

Build mode uses Anthropic (personal credits). Replay/production path will swap
the chat model for a local Ollama model with a known cutoff.

    python -m agent.run --dry-run                      # no API key, mock data
    python -m agent.run --dry-run --source real \
        --matchup LAL-BOS-2024-12-25 --as-of 2024-12-24
    python -m agent.run --source real --matchup ... --as-of ...   # live LLM
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
- Tool output may contain nulls with an "unavailable" or "warnings" note. Those
  mean the data does not exist yet. Say so in key_factors. Never fill a gap with
  a guess, and never treat a null as zero.
- Final answer must be valid JSON with keys:
  matchup_id, as_of_date, home_win_prob, away_win_prob, key_factors (list of
  short strings), narrative (2-4 sentences).
- Do not invent stats that tools did not return.
"""


def build_agent(source):
    from langchain.agents import create_agent
    from langchain_anthropic import ChatAnthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY missing. Copy .env.example to .env and set your key, "
            "or run with --dry-run (no LLM, exercises the same tools and data)."
        )
    model = ChatAnthropic(model="claude-sonnet-4-5", api_key=api_key, temperature=0)
    return create_agent(model, build_tools(source), system_prompt=SYSTEM)


def run_matchup(matchup_id: str, as_of_date: str, source) -> str:
    agent = build_agent(source)
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
        key_factors.append(
            f"Rest: {home['abbr']} {rest.get('home_days_rest')}d, "
            f"{away['abbr']} {rest.get('away_days_rest')}d"
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

    return json.dumps(
        {
            "matchup_id": ctx["matchup_id"],
            "as_of_date": as_of_date,
            "source": ctx["source"],
            "home_win_prob": pred.get("home_win_prob"),
            "away_win_prob": pred.get("away_win_prob"),
            "key_factors": key_factors,
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
    args = parser.parse_args()

    source = get_source(args.source)
    if args.dry_run:
        print(dry_run(args.matchup, args.as_of, source))
    else:
        print(run_matchup(args.matchup, args.as_of, source))


if __name__ == "__main__":
    main()
