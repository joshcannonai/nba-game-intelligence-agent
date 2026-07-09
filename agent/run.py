"""LangChain tool-calling analyst agent for a single NBA matchup.

Build mode uses Anthropic (personal credits). Replay/production path will swap
the chat model for a local Ollama model with a known cutoff.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

# Allow `python -m agent.run` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.tools import TOOLS  # noqa: E402

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
- Final answer must be valid JSON with keys:
  matchup_id, as_of_date, home_win_prob, away_win_prob, key_factors (list of
  short strings), narrative (2-4 sentences).
- Do not invent stats that tools did not return.
"""


def build_agent():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY missing. Copy .env.example to .env and set your key."
        )
    model = ChatAnthropic(
        model="claude-sonnet-4-5",
        api_key=api_key,
        temperature=0,
    )
    return create_agent(model, TOOLS, system_prompt=SYSTEM)


def run_matchup(matchup_id: str, as_of_date: str) -> str:
    agent = build_agent()
    user = (
        f"Produce a pregame report for matchup_id={matchup_id} as_of_date={as_of_date}."
    )
    result = agent.invoke({"messages": [{"role": "user", "content": user}]})
    messages = result.get("messages", [])
    if not messages:
        return json.dumps({"error": "no agent messages"}, indent=2)
    last = messages[-1]
    content = getattr(last, "content", last)
    if isinstance(content, list):
        text_parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ]
        content = "\n".join(p for p in text_parts if p)
    return str(content)


def dry_run(matchup_id: str, as_of_date: str) -> str:
    """No API key needed. Exercises the tool contracts and prints a sample report."""
    from agent.tools import (
        predict_win_probability,
        retrieve_matchup_context,
        retrieve_player_splits,
    )

    ctx = json.loads(
        retrieve_matchup_context.invoke(
            {"matchup_id": matchup_id, "as_of_date": as_of_date}
        )
    )
    lebron = json.loads(
        retrieve_player_splits.invoke(
            {"player_name": "LeBron James", "back_to_back": True}
        )
    )
    pred = json.loads(
        predict_win_probability.invoke(
            {
                "home_abbr": ctx["home_team"]["abbr"],
                "away_abbr": ctx["away_team"]["abbr"],
                "as_of_date": as_of_date,
            }
        )
    )
    injury_line = (
        f"Injury watch: {ctx['injuries'][0]['player']} ({ctx['injuries'][0]['status']})"
        if ctx["injuries"]
        else "No injuries in mock as-of window"
    )
    report = {
        "matchup_id": ctx["matchup_id"],
        "as_of_date": as_of_date,
        "home_win_prob": pred["home_win_prob"],
        "away_win_prob": pred["away_win_prob"],
        "key_factors": [
            f"{ctx['home_team']['abbr']} net rating edge vs {ctx['away_team']['abbr']}",
            f"{ctx['away_team']['abbr']} on a back-to-back (0 days rest)",
            injury_line,
            (
                f"{lebron['name']} b2b pts avg {lebron['b2b_pts_avg']} "
                f"vs season {lebron['pts_avg']}"
            ),
        ],
        "narrative": (
            f"Stub report for {ctx['away_team']['abbr']} @ {ctx['home_team']['abbr']} "
            f"on {ctx['game_date']}. Home win probability {pred['home_win_prob']:.1%} "
            f"from the net-rating stub (not XGBoost yet). Away side is on a back-to-back, "
            f"so fatigue splits matter more than season averages."
        ),
        "mode": "dry_run_no_llm",
    }
    return json.dumps(report, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="NBA Game Intelligence agent (mock)")
    parser.add_argument(
        "--matchup",
        default="LAL-BOS-2026-01-15",
        help="Matchup id (default: LAL-BOS-2026-01-15)",
    )
    parser.add_argument(
        "--as-of",
        default="2026-01-14",
        help="As-of date YYYY-MM-DD (default: day before tip)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exercise tools + print a sample report without calling an LLM",
    )
    args = parser.parse_args()
    if args.dry_run:
        print(dry_run(args.matchup, args.as_of))
    else:
        print(run_matchup(args.matchup, args.as_of))


if __name__ == "__main__":
    main()
