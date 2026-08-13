"""30 live Gemma passes for Models D and E.

D and E are Gemma 4 agents that may see the closing line. A/B/C must not.
This runner builds that extra tool here instead of in agent/tools.py because
the B/C full-season eval fingerprints agent/run.py, agent/tools.py, and
skills/*.md — editing those files would discard the in-progress B/C checkpoint.

    python -m eval.gemma_de              # 30 passes on a gated 16-game sample
    python -m eval.gemma_de --smoke      # one game, one pass (path check)

Each pass tweaks the prompt, scores the same sample, keeps the change only if
the objective moved the right way (D: winners, E: money), then writes the log.
Do not stop until 30. Resume is safe: per-pass jsonl is append-only.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import timedelta
from pathlib import Path

from langchain_core.tools import tool

from agent.run import _AGENT_REASONING_CORE, _MODEL_C_PREDICTOR_ADDITION
from agent.skills import skills_block
from agent.sources import get_source
from agent.tools import build_tools
from eval.betting import STAKE, fair_home_prob, load_odds, odds_for_matchup, settle
from eval.ui_agent_eval import parse_final_json
from models.features import build_season

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "docs" / "evaluation" / "gemma-de-live-log.json"
GAMES_JSONL = ROOT / "eval" / "results_gemma_de.jsonl"
N_GAMES = 16
SCORE_COLUMNS = {
    "score_away",
    "score_home",
    "q1_away",
    "q2_away",
    "q3_away",
    "q4_away",
    "ot_away",
    "q1_home",
    "q2_home",
    "q3_home",
    "q4_home",
    "ot_home",
    "winner",
}

_MARKET_RULES = """
- Always call retrieve_matchup_context first.
- Call retrieve_team_form for BOTH teams, retrieve_injuries for both, and
  retrieve_betting_line once. After matchup context returns, request the two
  form calls, two injury calls, and the betting line together in one turn.
- Always call predict_win_probability. It is Model A, which cannot see the
  market. Compare it with the line. You may agree or disagree.
- A run is complete only after matchup context, two team-form results, two
  injury results, one betting line, and one Model A probability.
- You HAVE a betting-line tool. Call it. Do not recall a spread from memory
  if the tool returns not_found or gated. The 2025-26 file has spreads and
  favorites, not moneylines.
- Injuries are not equal names. Rank by importance when it is not None.
  A high-importance absence (star minutes/points) can move a pick; a
  low-importance rotation piece should not.
- Final answer must be valid JSON with keys:
  matchup_id, as_of_date, home_win_prob, away_win_prob, pick, key_factors
  (list of short strings), missing, narrative (2-4 sentences).
- home_win_prob and away_win_prob must be numbers between 0 and 1 that sum to 1.
  They are your true belief that the home team wins.
- pick must be "HOME", "AWAY", or "NONE". NONE means no bet (P&L 0).
- Do not invent stats that tools did not return.
"""

_D_OBJECTIVE = """You are Model D, a Gemma NBA analyst whose job is to pick WINNERS.
You may see the closing spread. Maximize accuracy, not profit. `pick` should be
the team you think will win. Use the market as one signal among form, rest,
injuries (by importance), and Model A. Do not blindly copy Vegas. Do not ignore
a double-digit favorite. Start from the 55% NBA home-team base rate.

"""

_E_OBJECTIVE = """You are Model E, a Gemma NBA analyst whose job is to MAKE MONEY
on reconstructed moneylines ($100 a game, 3.75% hold). `home_win_prob` is your
true belief. `pick` is the side you would bet — it may differ from the likely
winner when the price is bad. You may return pick=NONE to skip a game.
A 75%+ market favorite is expensive to fade. Juice on huge favorites is a
reason not to bet them at all if your edge is small.

"""

BETTING_SKILL = """
--- SKILL FILE: eval/gemma_de.py (D/E only; A/B/C do not have this tool) ---
---
tool: retrieve_betting_line
use_when: Always, once, after matchup context. Models D and E only.
---

## What it gives you

The closing spread, who is favored, and signed home/away spreads. Moneylines
are empty for 2025-26; do not invent them.

## Rules

- This is the market's number. D uses it to pick winners. E uses it as the price.
- It is not the result. Score columns are not in this payload.
- If status is not_found or gated, say so in missing and reason without a line.
"""

SYSTEM_D = (
    _D_OBJECTIVE + _AGENT_REASONING_CORE + _MODEL_C_PREDICTOR_ADDITION + _MARKET_RULES
)
SYSTEM_E = (
    _E_OBJECTIVE + _AGENT_REASONING_CORE + _MODEL_C_PREDICTOR_ADDITION + _MARKET_RULES
)

# Sequential hypotheses. After each pass we keep the addendum only if the
# sample score moved the right way. Later passes stack on kept text.
PASSES: list[dict] = [
    {
        "arm": "D",
        "tweak": "baseline: Gemma + line, pick winners",
        "why": "Need a Gemma D number. Same retrieval as C, plus the closing spread.",
        "addendum": "",
        "next": "Start from the market implied probability, then adjust.",
    },
    {
        "arm": "D",
        "tweak": "start from market implied p (spread/14)",
        "why": "CPU D's only keep was log-odds of the close. Make Gemma start there.",
        "addendum": (
            "\n- Convert the spread to a probability with a normal margin model "
            "(sigma 14 points). Start from that number, then adjust with form "
            "and injuries. Do not start from 50/50 and wander toward the line."
        ),
        "next": "Never fade a 7-point favorite unless a star is out.",
    },
    {
        "arm": "D",
        "tweak": "never fade a 7+ point favorite without a star out",
        "why": "NBA 7-point favorites win often. Gemma should need a star absence to fade.",
        "addendum": (
            "\n- If the spread is 7 or more points, pick the favorite unless the "
            "favorite is missing a high-importance player (importance clearly "
            "above a typical starter). A backup center is not enough to fade."
        ),
        "next": "Spell out Kyrie-vs-Lively injury ranking.",
    },
    {
        "arm": "D",
        "tweak": "rank injuries Kyrie vs Lively",
        "why": "Equal-name injury lists are how you miss a star and overreact to a backup.",
        "addendum": (
            "\n- Rank absences by importance. A 20-point creator out is a larger "
            "availability concern than a low-minute big. Do not treat every name "
            "on the injury list as equal. If importance is None, say unknown, "
            "do not treat it as zero."
        ),
        "next": "When form is missing, trust the market more.",
    },
    {
        "arm": "D",
        "tweak": "missing form → trust market more",
        "why": "Opening week has no rolling form. The line is the sharpest gated signal then.",
        "addendum": (
            "\n- If retrieve_team_form is awaiting_input for either team, weight "
            "the betting line more than prior-season ratings. Do not emit 0.48/0.52."
        ),
        "next": "When form is deep, allow disagreement with the market.",
    },
    {
        "arm": "D",
        "tweak": "deep form may disagree with the market",
        "why": "If both teams have 10 games of form, independent evidence can beat a 3-point line.",
        "addendum": (
            "\n- If both teams have a full 10-game form window and Model A disagrees "
            "with the favorite, you may fade a spread of 3 points or less. Do not "
            "fade larger lines on form alone."
        ),
        "next": "Rest and B2B only as a tie-breaker on short lines.",
    },
    {
        "arm": "D",
        "tweak": "rest/B2B only on |spread| <= 3",
        "why": "Rest is a small NBA effect. It should not overturn a 8-point favorite.",
        "addendum": (
            "\n- Rest and back-to-backs are tie-breakers only when the spread is "
            "3 points or less. They do not overturn a clear favorite."
        ),
        "next": "If Model A and the market agree, lock that side.",
    },
    {
        "arm": "D",
        "tweak": "lock when Model A and market agree",
        "why": "Two independent systems pointing the same way is the high-confidence case.",
        "addendum": (
            "\n- If Model A and the betting line pick the same winner, pick that "
            "side. Do not override agreement with a narrative."
        ),
        "next": "On disagreement, inspect injuries then default to the market.",
    },
    {
        "arm": "D",
        "tweak": "disagreement: injuries, else market",
        "why": "When A and Vegas split, the usual reason a line is wrong is a star out.",
        "addendum": (
            "\n- If Model A and the market disagree, look at high-importance injuries. "
            "If those do not explain the gap, pick the market's side."
        ),
        "next": "Playoffs: trust the market more.",
    },
    {
        "arm": "D",
        "tweak": "playoffs: trust market more",
        "why": "Playoff lines are sharper and rotations shrink. Copying chalk is less lazy there.",
        "addendum": (
            "\n- If matchup context or the schedule indicates a playoff game, "
            "weight the betting line more than regular-season form."
        ),
        "next": "October-November: trust market more because form is noisy.",
    },
    {
        "arm": "D",
        "tweak": "Oct-Nov: trust market more",
        "why": "Early-season form is small-sample. The book has more than 10 games of priors.",
        "addendum": (
            "\n- For games in October or November, weight the betting line more "
            "than rolling form. Form windows are short then."
        ),
        "next": "Do not copy coin-flip lines; use 55% home plus form.",
    },
    {
        "arm": "D",
        "tweak": "do not copy coin-flip lines",
        "why": "A 1-point spread is not information. Home base rate plus form should decide those.",
        "addendum": (
            "\n- If the spread is 1.5 points or less, do not copy the favorite. "
            "Use 55% home plus form, rest, and injuries."
        ),
        "next": "Blend 30% Model A / 70% market in log-odds.",
    },
    {
        "arm": "D",
        "tweak": "blend 30% A / 70% market",
        "why": "CPU search: the close dominates. Make Gemma's probability a 70% market blend.",
        "addendum": (
            "\n- Set home_win_prob to a log-odds blend: 30% Model A, 70% market "
            "implied probability, then apply only injury adjustments that the "
            "blend could not see."
        ),
        "next": "Try the opposite blend: 70% A / 30% market.",
    },
    {
        "arm": "D",
        "tweak": "blend 70% A / 30% market",
        "why": "If 70% market did not help, maybe Gemma should stay closer to Model A.",
        "addendum": (
            "\n- Set home_win_prob to a log-odds blend: 70% Model A, 30% market "
            "implied probability, then apply injury adjustments."
        ),
        "next": "Fade only when a star is out on the favorite AND A already faded.",
    },
    {
        "arm": "D",
        "tweak": "fade only if star-out and A faded",
        "why": "Two conditions to fade: the book may be slow on a star, and A already noticed.",
        "addendum": (
            "\n- Fade the market favorite only if (1) that team is missing a "
            "high-importance player and (2) Model A already picks the other side. "
            "Otherwise pick the favorite."
        ),
        "next": "Home dogs with a rest edge: pick home more often.",
    },
    {
        "arm": "D",
        "tweak": "home dogs with rest edge",
        "why": "Home + rest is the classic small NBA overlay on a short dog line.",
        "addendum": (
            "\n- If the home team is the underdog by 3.5 points or less AND has "
            "more rest than the away team, pick home unless a high-importance "
            "home player is out."
        ),
        "next": "Away favorites on a B2B: shade toward home.",
    },
    {
        "arm": "D",
        "tweak": "away favorite on a B2B: shade home",
        "why": "Road B2B favorites are a known tired-leg spot. Small, legal, gated.",
        "addendum": (
            "\n- If the away team is favored and is on a back-to-back, shade "
            "toward home. Fade only if the spread is 4 points or less."
        ),
        "next": "Always take a 10-point favorite.",
    },
    {
        "arm": "D",
        "tweak": "always take a 10-point favorite",
        "why": "Double-digit NBA favorites almost always win. D is about winners.",
        "addendum": (
            "\n- If the spread is 10 points or more, pick the favorite. Do not fade "
            "double-digit lines on form, rest, or a low-importance injury."
        ),
        "next": "Tighten the injury rule one more time on the kept stack.",
    },
    {
        "arm": "D",
        "tweak": "star-out can fade a 6-9 point line",
        "why": "The 10-point lock may be right; 6-9 point lines are where a Kyrie-level out matters.",
        "addendum": (
            "\n- A high-importance absence on the favorite may fade a 6 to 9 point "
            "spread. It may not fade 10+. A low-importance absence may not fade 3+."
        ),
        "next": "Freeze D. Switch to E: money, not winners.",
    },
    {
        "arm": "D",
        "tweak": "freeze D kept stack, no new rule",
        "why": "Re-score the accumulated D prompt with no new text so we know the stack is stable.",
        "addendum": "",
        "next": "E baseline: same tools, pick the +EV moneyline.",
    },
    {
        "arm": "E",
        "tweak": "baseline: Gemma + line, maximize money",
        "why": "E's job is P&L. Belief and bet can differ. NONE is allowed.",
        "addendum": "",
        "next": "Never fade a 70%+ market favorite.",
    },
    {
        "arm": "E",
        "tweak": "never fade a 70%+ favorite",
        "why": "CPU E's best keep was never-fade-70%. Try it in the Gemma prompt.",
        "addendum": (
            "\n- If the market implied probability of the favorite is 70% or higher, "
            "pick that favorite. Do not fade 70%+ chalk for price."
        ),
        "next": "Never fade 75%+ (the CPU E11 rule).",
    },
    {
        "arm": "E",
        "tweak": "never fade a 75%+ favorite",
        "why": "E11 on the full season. See if Gemma plus 75% never-fade beats 70%.",
        "addendum": (
            "\n- If the market implied probability of the favorite is 75% or higher, "
            "pick that favorite. Below 75%, bet the side with higher expected value."
        ),
        "next": "Skip huge favorites (no-bet) instead of betting them.",
    },
    {
        "arm": "E",
        "tweak": "no-bet when market p >= 80%",
        "why": "Huge favorites pay -400. Even when they win, juice eats the bankroll.",
        "addendum": (
            "\n- If the market implied probability of the favorite is 80% or higher, "
            "return pick=NONE. Do not bet -400 chalk."
        ),
        "next": "Only bet when |belief - market| >= 0.05.",
    },
    {
        "arm": "E",
        "tweak": "only bet with a 5-point edge",
        "why": "Vig is 3.75%. A 5-point belief gap is the smallest plausible overlay.",
        "addendum": (
            "\n- Return pick=NONE unless your home_win_prob differs from the market "
            "implied probability by at least 0.05. Then bet the +EV side."
        ),
        "next": "Only bet dogs when the favorite has a high-importance out.",
    },
    {
        "arm": "E",
        "tweak": "dogs only if favorite star is out",
        "why": "The +EV dog needs a reason the line is wrong. A star out is the gated reason.",
        "addendum": (
            "\n- Only pick an underdog if the favorite is missing a high-importance "
            "player. Otherwise pick the favorite or NONE."
        ),
        "next": "No-bet opening week (form missing).",
    },
    {
        "arm": "E",
        "tweak": "no-bet when form is missing",
        "why": "Without form, E's independent belief is weak. Passing is a money decision.",
        "addendum": (
            "\n- If retrieve_team_form is awaiting_input for either team, return "
            "pick=NONE. Do not bet opening week on prior-season ratings."
        ),
        "next": "Bet home dogs that have a rest advantage.",
    },
    {
        "arm": "E",
        "tweak": "bet home dogs with rest",
        "why": "Home + rest + plus-money is the overlay E can actually cash.",
        "addendum": (
            "\n- If the home team is a 3.5-point-or-less underdog and has more rest "
            "than the away team, pick HOME. This is a money bet, not a winner bet."
        ),
        "next": "Only bet when EV on $100 is at least $8.",
    },
    {
        "arm": "E",
        "tweak": "only bet if EV >= $8 per $100",
        "why": "Need a buffer above vig. Small edges are noise on 16 games.",
        "addendum": (
            "\n- Compute expected value of each moneyline from your belief and the "
            "spread-implied price. Return pick=NONE unless EV is at least $8 on $100."
        ),
        "next": "Combine the kept E rules into one short stack and re-score.",
    },
    {
        "arm": "E",
        "tweak": "freeze E kept stack, no new rule",
        "why": "Re-score accumulated E prompt with no new text so the stack is stable.",
        "addendum": "",
        "next": "Stop. Review all 30. LLM stays in D/E only if it beats the CPU policy on this sample.",
    },
]


def retrieve_betting_line_tool(
    source, required_as_of_date: str, required_matchup_id: str
):
    """D/E-only market tool. Not registered on A/B/C."""

    @tool
    def retrieve_betting_line(matchup_id: str, as_of_date: str) -> str:
        """Closing spread and favorite for this matchup. D/E only.

        Args:
            matchup_id: AWAY-HOME-YYYY-MM-DD.
            as_of_date: ISO date. Must equal the authorized cutoff.
        """
        if as_of_date != required_as_of_date or matchup_id != required_matchup_id:
            return json.dumps(
                {
                    "status": "error",
                    "tool": "retrieve_betting_line",
                    "error": "matchup_id and as_of_date must equal the authorized request",
                    "data_source_read": False,
                },
                indent=2,
            )
        payload = source.betting_line(matchup_id, as_of_date)
        leaked = SCORE_COLUMNS & payload.keys()
        if leaked:
            raise RuntimeError(
                f"retrieve_betting_line leaked score fields: {sorted(leaked)}"
            )
        if payload.get("status") == "ok":
            payload["caveat"] = (
                "Closing spread for Models D and E. A/B/C do not receive this tool. "
                "2025-26 moneylines are empty; use spread and favorite."
            )
        return json.dumps(payload, indent=2)

    return retrieve_betting_line


def system_prompt_for_de(objective: str, tool_names: list[str], addendum: str) -> str:
    base = SYSTEM_E if objective == "money" else SYSTEM_D
    names = [n for n in tool_names if n != "retrieve_betting_line"]
    return base + skills_block(names) + BETTING_SKILL + (addendum or "")


def build_de_agent(
    source,
    *,
    objective: str,
    prompt_addendum: str,
    required_as_of_date: str,
    required_matchup_id: str,
):
    from langchain.agents import create_agent
    from langchain_ollama import ChatOllama

    tools = list(
        build_tools(
            source,
            include_model=True,
            required_as_of_date=required_as_of_date,
            required_matchup_id=required_matchup_id,
        )
    )
    tools.append(
        retrieve_betting_line_tool(source, required_as_of_date, required_matchup_id)
    )
    if any(
        t.name == "retrieve_betting_line"
        for t in build_tools(source, include_model=True)
    ):
        raise RuntimeError("A/B/C tool surface must not include retrieve_betting_line")
    prompt = system_prompt_for_de(objective, [t.name for t in tools], prompt_addendum)
    model = ChatOllama(model="gemma4", temperature=0, reasoning=False)
    return create_agent(model, tools, system_prompt=prompt)


def _message_text(content) -> str:
    if isinstance(content, list):
        parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
        return "\n".join(p for p in parts if p)
    return str(content)


def run_matchup_de(
    matchup_id: str,
    as_of_date: str,
    *,
    objective: str,
    prompt_addendum: str,
    source=None,
) -> dict:
    source = source or get_source("real")
    agent = build_de_agent(
        source,
        objective=objective,
        prompt_addendum=prompt_addendum,
        required_as_of_date=as_of_date,
        required_matchup_id=matchup_id,
    )
    user = (
        f"Produce a pregame report for matchup_id={matchup_id} as_of_date={as_of_date}."
    )
    result = agent.invoke({"messages": [{"role": "user", "content": user}]})
    messages = result.get("messages", [])
    if not messages:
        raise RuntimeError("no agent messages")
    text = _message_text(getattr(messages[-1], "content", messages[-1]))
    return parse_final_json(text)


def sample_games(n: int = N_GAMES) -> list[dict]:
    """Evenly spaced 2025-26 games with a result and a closing line."""
    odds = load_odds("2026")
    rows, _ = build_season(2026)
    eligible = []
    for row in rows:
        if row.home_won is None:
            continue
        odds_row = odds_for_matchup(row.away, row.home, row.game_date, odds)
        p_market = fair_home_prob(odds_row) if odds_row else None
        if p_market is None:
            continue
        eligible.append(
            {
                "game_id": row.game_id,
                "game_date": row.game_date.isoformat(),
                "home": row.home,
                "away": row.away,
                "home_won": bool(row.home_won),
                "playoffs": bool(row.playoffs),
                "p_market": p_market,
                "cutoff": (row.game_date - timedelta(days=1)).isoformat(),
            }
        )
    if len(eligible) < n:
        raise RuntimeError(f"only {len(eligible)} eligible games, need {n}")
    step = len(eligible) / n
    picked = [eligible[int(i * step)] for i in range(n)]
    # De-dupe in case of collision; fill from the back of the season.
    seen = {g["game_id"] for g in picked}
    fill = [g for g in reversed(eligible) if g["game_id"] not in seen]
    unique = []
    for g in picked:
        if g["game_id"] in {u["game_id"] for u in unique}:
            unique.append(fill.pop(0))
        else:
            unique.append(g)
    return unique


def pick_from_final(
    final: dict, home: str | None = None, away: str | None = None
) -> bool | None:
    """True=home bet, False=away bet, None=no bet.

    Gemma often emits a team abbreviation instead of HOME/AWAY. Accept both.
    """
    from agent.teams import abbr_from_nickname, normalize_abbr

    raw = str(final.get("pick") or final.get("predicted_winner") or "").strip()
    pick = raw.upper()
    if pick in {"NONE", "SKIP", "NO_BET", "PASS"}:
        return None
    if pick in {"HOME", "H"}:
        return True
    if pick in {"AWAY", "A"}:
        return False
    token = None
    if raw:
        token = abbr_from_nickname(raw) or abbr_from_nickname(raw.split()[-1])
        try:
            token = normalize_abbr(token or raw.replace(" ", ""))
        except Exception:
            token = raw.upper()
    if home and token == normalize_abbr(home):
        return True
    if away and token == normalize_abbr(away):
        return False
    home_p = final.get("home_win_prob")
    if isinstance(home_p, (int, float)):
        return float(home_p) >= 0.5
    raise ValueError(f"no pick in agent output: {final!r}"[:300])


def score_games(results: list[dict]) -> dict:
    n = len(results)
    bets = [r for r in results if r["pick_home"] is not None]
    correct = sum(1 for r in bets if r["correct"])
    pnl = sum(r["pnl"] for r in results)
    staked = STAKE * len(bets)
    return {
        "n_games": n,
        "n_bets": len(bets),
        "correct": correct,
        "accuracy": (correct / len(bets)) if bets else 0.0,
        "net_pnl": round(pnl, 2),
        "roi_pct": round(100.0 * pnl / staked, 2) if staked else 0.0,
        "vegas_accuracy": sum(1 for r in results if r["vegas_correct"]) / n,
        "parse_failures": sum(1 for r in results if r.get("error")),
    }


def _settle_one(
    pick_home: bool | None, home_won: bool, p_market: float
) -> tuple[int | None, float]:
    if pick_home is None:
        return None, 0.0
    correct = int(pick_home == home_won)
    return correct, settle(pick_home, home_won, p_market)


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def _load_done_passes(path: Path) -> dict[int, list[dict]]:
    done: dict[int, list[dict]] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done.setdefault(int(row["pass"]), []).append(row)
    return done


def run_pass(
    pass_n: int,
    spec: dict,
    games: list[dict],
    addendum: str,
    *,
    done_rows: list[dict] | None = None,
    source=None,
) -> list[dict]:
    objective = "money" if spec["arm"] == "E" else "winner"
    source = source or get_source("real")
    have = {r["game_id"]: r for r in (done_rows or []) if not r.get("error")}
    out = []
    for i, game in enumerate(games, start=1):
        if game["game_id"] in have:
            out.append(have[game["game_id"]])
            print(
                f"    resume {spec['arm']} {game['game_id']}",
                flush=True,
            )
            continue
        t0 = time.time()
        error = None
        final = {}
        pick_home = None
        try:
            final = run_matchup_de(
                game["game_id"],
                game["cutoff"],
                objective=objective,
                prompt_addendum=addendum,
                source=source,
            )
            pick_home = pick_from_final(final, home=game["home"], away=game["away"])
        except Exception as exc:  # one game must not kill the 30-pass run
            error = f"{type(exc).__name__}: {exc}"
            final = {"error": error}
        correct, pnl = _settle_one(pick_home, game["home_won"], game["p_market"])
        vegas_home = game["p_market"] >= 0.5
        row = {
            "pass": pass_n,
            "arm": spec["arm"],
            "tweak": spec["tweak"],
            "game_id": game["game_id"],
            "cutoff": game["cutoff"],
            "p_home": final.get("home_win_prob"),
            "pick": final.get("pick"),
            "pick_home": pick_home,
            "home_won": game["home_won"],
            "correct": correct,
            "pnl": round(pnl, 2),
            "p_market": game["p_market"],
            "vegas_correct": int(vegas_home == game["home_won"]),
            "elapsed_seconds": round(time.time() - t0, 2),
            "error": error,
            "final_json": final,
        }
        _append_jsonl(GAMES_JSONL, row)
        out.append(row)
        mark = "skip" if pick_home is None else ("ok" if correct else "miss")
        print(
            f"    [{i}/{len(games)}] {spec['arm']} {game['game_id']} "
            f"pick={final.get('pick')} {mark} {row['elapsed_seconds']:.0f}s",
            flush=True,
        )
    return out


def _better(arm: str, score: dict, best: dict) -> bool:
    if arm == "D":
        if score["accuracy"] > best["accuracy"] + 1e-12:
            return True
        if abs(score["accuracy"] - best["accuracy"]) <= 1e-12:
            return score["net_pnl"] > best["net_pnl"]
        return False
    if score["net_pnl"] > best["net_pnl"] + 1e-9:
        return True
    if abs(score["net_pnl"] - best["net_pnl"]) <= 1e-9:
        return score["accuracy"] > best["accuracy"]
    return False


def write_log(payload: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_search(games: list[dict], *, start_pass: int = 1) -> dict:
    source = get_source("real")
    done = _load_done_passes(GAMES_JSONL)
    passes_out: list[dict] = []
    if LOG_PATH.exists():
        prev = json.loads(LOG_PATH.read_text(encoding="utf-8"))
        passes_out = list(prev.get("passes") or [])

    kept_d: list[str] = []
    kept_e: list[str] = []
    best_d = {"accuracy": -1.0, "net_pnl": -1e18}
    best_e = {"accuracy": -1.0, "net_pnl": -1e18}
    for rec in passes_out:
        if rec["arm"] == "D" and rec.get("kept"):
            if rec.get("addendum"):
                kept_d.append(rec["addendum"])
            best_d = {
                "accuracy": rec["accuracy"],
                "net_pnl": rec["net_pnl"],
            }
        if rec["arm"] == "E" and rec.get("kept"):
            if rec.get("addendum"):
                kept_e.append(rec["addendum"])
            best_e = {
                "accuracy": rec["accuracy"],
                "net_pnl": rec["net_pnl"],
            }

    finished = {int(r["pass"]) for r in passes_out}
    vegas_acc = sum(
        g["p_market"] >= 0.5
        and g["home_won"]
        or g["p_market"] < 0.5
        and not g["home_won"]
        for g in games
    ) / len(games)

    for i, spec in enumerate(PASSES, start=1):
        if i < start_pass or i in finished:
            continue
        stack = kept_e if spec["arm"] == "E" else kept_d
        trial = "".join(stack) + (spec["addendum"] or "")
        print(f"\n== pass {i:02d}/30 {spec['arm']} {spec['tweak']} ==", flush=True)
        rows = run_pass(i, spec, games, trial, done_rows=done.get(i), source=source)
        score = score_games(rows)
        best = best_e if spec["arm"] == "E" else best_d
        if best["accuracy"] < 0:
            kept = True
        elif not spec["addendum"]:
            kept = True
        else:
            kept = _better(spec["arm"], score, best)
        if kept:
            if spec["addendum"]:
                stack.append(spec["addendum"])
            if spec["addendum"] or best["accuracy"] < 0:
                if spec["arm"] == "D":
                    best_d = {
                        "accuracy": score["accuracy"],
                        "net_pnl": score["net_pnl"],
                    }
                else:
                    best_e = {
                        "accuracy": score["accuracy"],
                        "net_pnl": score["net_pnl"],
                    }
        rec = {
            "pass": i,
            "arm": spec["arm"],
            "tweak": spec["tweak"],
            "why_this_pass": spec["why"],
            "addendum": spec["addendum"],
            "win_rate": round(score["accuracy"], 4),
            "accuracy": score["accuracy"],
            "correct": score["correct"],
            "n_bets": score["n_bets"],
            "n_games": score["n_games"],
            "net_pnl": score["net_pnl"],
            "roi_pct": score["roi_pct"],
            "vegas_accuracy": round(score["vegas_accuracy"], 4),
            "parse_failures": score["parse_failures"],
            "kept": kept,
            "running_best_win_rate": round(
                (best_d if spec["arm"] == "D" else best_e)["accuracy"], 4
            ),
            "running_best_pnl": (best_d if spec["arm"] == "D" else best_e)["net_pnl"],
            "next_change_and_why": spec["next"],
            "win_rate_2026": f"{score['accuracy']:.1%} on {score['n_bets']} bets / {score['n_games']} games",
            "correct_2026": f"{score['correct']}/{score['n_bets']}",
            "pnl_2026": score["net_pnl"],
            "running_best_win_rate_2026": f"{(best_d if spec['arm'] == 'D' else best_e)['accuracy']:.1%}",
        }
        passes_out.append(rec)
        payload = {
            "model": "gemma4",
            "n_games": len(games),
            "game_ids": [g["game_id"] for g in games],
            "vegas_accuracy": round(vegas_acc, 4),
            "best_d_win_rate": best_d["accuracy"] if best_d["accuracy"] >= 0 else None,
            "best_d_pnl": best_d["net_pnl"] if best_d["accuracy"] >= 0 else None,
            "best_e_win_rate": best_e["accuracy"] if best_e["accuracy"] >= 0 else None,
            "best_e_pnl": best_e["net_pnl"] if best_e["accuracy"] >= 0 else None,
            "kept_d_rules": len([a for a in kept_d if a]),
            "kept_e_rules": len([a for a in kept_e if a]),
            "passes": passes_out,
            "note": (
                "Gemma 4 via Ollama. D/E may see retrieve_betting_line. "
                "A/B/C still cannot. Sample is date-gated (previous calendar day). "
                "Closing spread is the market feature; moneylines are reconstructed."
            ),
        }
        write_log(payload)
        print(
            f"  pass {i:02d} {spec['arm']} {spec['tweak']:<44} "
            f"win {score['accuracy']:.1%}  P&L {score['net_pnl']:+.0f}  "
            f"{'KEEP' if kept else 'revert'}  vegas {score['vegas_accuracy']:.1%}",
            flush=True,
        )

    return json.loads(LOG_PATH.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-games", type=int, default=N_GAMES)
    ap.add_argument("--smoke", action="store_true", help="One game, pass 1 only")
    ap.add_argument("--start-pass", type=int, default=1)
    args = ap.parse_args()

    from agent.tools import build_tools as _bt
    from agent.sources import get_source as _gs

    abc = {t.name for t in _bt(_gs("mock"), include_model=True)}
    if "retrieve_betting_line" in abc:
        raise SystemExit("refusing to run: A/B/C can see the market")

    games = sample_games(1 if args.smoke else args.n_games)
    print(
        f"Gemma D/E  model=gemma4  games={len(games)}  "
        f"vegas={sum(g['p_market'] >= 0.5 and g['home_won'] or g['p_market'] < 0.5 and not g['home_won'] for g in games) / len(games):.1%}"
    )
    for g in games:
        print(
            f"  {g['game_id']}  cutoff={g['cutoff']}  market_p_home={g['p_market']:.3f}"
        )
    if args.smoke:
        spec = PASSES[0]
        rows = run_pass(1, spec, games, "", source=get_source("real"))
        print(json.dumps(score_games(rows), indent=2))
        return
    payload = run_search(games, start_pass=args.start_pass)
    print(
        f"\nwrote {LOG_PATH.relative_to(ROOT)}  "
        f"D best {payload.get('best_d_win_rate')}  "
        f"E best P&L {payload.get('best_e_pnl')}"
    )


if __name__ == "__main__":
    main()
