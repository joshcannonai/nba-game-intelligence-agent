"""Replays old games day by day and scores each prediction method.
 
For every game, it looks at what was knowable the morning before tip-off,
asks each method to predict who wins, then checks the real result.
 
Prints 3 numbers per method:
    accuracy    % of games called correctly
    log loss    punishes confident WRONG guesses (lower = better)
    brier       how far off the probability was (lower = better)
 
Also compares against 2 baselines: always guessing the home team wins,
and the actual Vegas betting line.
 
Usage:
    python -m eval.replay --playoffs
    python -m eval.replay --limit 200 --out eval/results.csv
"""
 
from __future__ import annotations
 
import argparse
import csv
import json
import math
import sys
from datetime import timedelta
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
 
from agent.sources import get_source, parse_date  # noqa: E402
from agent.tools import build_tools  # noqa: E402
from agent.run import SYSTEM, run_matchup  # noqa: E402
 
ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-15
 
# run_matchup (the full agent) always calls predict_win_probability -- that's
# the hybrid arm. llm_direct is the same agent minus that one tool, so it has
# to guess a probability on its own instead of reading the classifier's answer.
LLM_DIRECT_SYSTEM = SYSTEM.replace(
    "- Always call predict_win_probability.\n", ""
).replace(
    "predict_best_player, and retrieve_news, plus predict_stat_line for a key\n"
    "  player.",
    "predict_best_player, and retrieve_news, plus predict_stat_line for a key\n"
    "  player. Do NOT call predict_win_probability -- decide the win probability\n"
    "  yourself, from context alone.",
)
 
 
def american_to_prob(ml: str) -> float | None:
    try:
        v = float(ml)
    except (TypeError, ValueError):
        return None
    return 100.0 / (v + 100.0) if v > 0 else abs(v) / (abs(v) + 100.0)
 
 
# How spread-out real game margins are around the betting spread (measured
# from last season's games). Used to turn a point spread into a probability.
MARGIN_SIGMA = 14.0
 
 
def _phi(x: float) -> float:
    """Standard normal CDF -- turns a z-score into a probability."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
 
 
def spread_home_prob(row: dict) -> float | None:
    """Turn a point spread into a home-win probability."""
    try:
        spread = float(row.get("spread", ""))
    except (TypeError, ValueError):
        return None
    favored = (row.get("whos_favored") or "").strip().lower()
    if favored not in {"home", "away"}:
        return None
    expected_home_margin = spread if favored == "home" else -spread
    return _phi(expected_home_margin / MARGIN_SIGMA)
 
 
def vegas_home_prob(row: dict) -> float | None:
    """The market's home-win probability, from the two moneylines (falls
    back to the spread if moneylines aren't there)."""
    h, a = (
        american_to_prob(row.get("moneyline_home")),
        american_to_prob(row.get("moneyline_away")),
    )
    if h is None or a is None or (h + a) == 0:
        return spread_home_prob(row)
    return h / (h + a)
 
 
def metrics(preds: list[tuple[float, int]]) -> dict:
    """preds = [(predicted_home_win_prob, actual_home_win 0/1), ...]"""
    if not preds:
        return {"n": 0}
    n = len(preds)
    acc = sum((p >= 0.5) == bool(y) for p, y in preds) / n
    ll = (
        -sum(
            y * math.log(max(p, EPS)) + (1 - y) * math.log(max(1 - p, EPS))
            for p, y in preds
        )
        / n
    )
    brier = sum((p - y) ** 2 for p, y in preds) / n
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "log_loss": round(ll, 4),
        "brier": round(brier, 4),
    }
 
 
def _extract_home_prob(agent_output: str) -> float | None:
    """Pull home_win_prob out of the agent's final JSON. Returns None if it
    can't -- never a guessed number."""
    try:
        data = json.loads(agent_output)
        return float(data.get("home_win_prob"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
 
 
def _build_llm_direct_agent(source, model_backend: str):
    """Same agent as agent.run.build_agent, minus predict_win_probability."""
    from langchain.agents import create_agent
 
    if model_backend == "anthropic":
        from langchain_anthropic import ChatAnthropic
        import os
 
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY missing")
        model = ChatAnthropic(model="claude-sonnet-4-5", api_key=api_key, temperature=0)
    elif model_backend == "ollama":
        from langchain_ollama import ChatOllama
 
        model = ChatOllama(model="gemma4", temperature=0)
    else:
        raise ValueError(f"unknown model backend: {model_backend!r}")
 
    tools = [t for t in build_tools(source) if t.name != "predict_win_probability"]
    return create_agent(model, tools, system_prompt=LLM_DIRECT_SYSTEM)
 
 
def run_llm_direct(matchup_id: str, as_of_date: str, source, model_backend: str) -> str:
    agent = _build_llm_direct_agent(source, model_backend)
    user = f"Produce a pregame report for matchup_id={matchup_id} as_of_date={as_of_date}."
    result = agent.invoke({"messages": [{"role": "user", "content": user}]})
    messages = result.get("messages", [])
    if not messages:
        return json.dumps({"error": "no agent messages"})
    content = getattr(messages[-1], "content", messages[-1])
    if isinstance(content, list):
        content = "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)
 
 
def run_agent_arms(games: list[dict], sample_n: int, model_backend: str) -> tuple[dict, dict, str | None]:
    """Score llm_direct and hybrid on a small SAMPLE of games -- the agent
    loop is slow (~30s/game), unlike the classifier arm which runs on all of
    them. If the LLM backend isn't available, stop after the first game and
    say why, instead of silently returning zeros or retrying every game."""
    source = get_source("real")
    llm_preds, hybrid_preds = [], []
 
    for i, g in enumerate(games[:sample_n]):
        as_of = (parse_date(g["game_date"]) - timedelta(days=1)).isoformat()
        actual = 1 if g["winner"] == g["home"] else 0
 
        try:
            llm_out = run_llm_direct(g["game_id"], as_of, source, model_backend)
            hybrid_out = run_matchup(g["game_id"], as_of, source, model_backend)
        except (Exception, SystemExit) as e:
            if i == 0:
                return {}, {}, f"{type(e).__name__}: {e}"
            continue
 
        lp, hp = _extract_home_prob(llm_out), _extract_home_prob(hybrid_out)
        if lp is not None:
            llm_preds.append((lp, actual))
        if hp is not None:
            hybrid_preds.append((hp, actual))
 
    return metrics(llm_preds), metrics(hybrid_preds), None
 
 
def load_rows(playoffs_only: bool, limit: int | None) -> tuple[list[dict], dict]:
    games_path = ROOT / "data/samples/game_logs_2026.csv"
    odds_path = ROOT / "data/samples/odds_2026.csv"
    if not games_path.exists():
        raise SystemExit(
            f"missing {games_path.name}. Run: python scripts/build_2026_testset.py"
        )
 
    games = list(csv.DictReader(open(games_path)))
    if playoffs_only:
        games = [g for g in games if g.get("playoffs") == "1"]
    if limit:
        games = games[:limit]
 
    odds = {}
    if odds_path.exists():
        odds = {r["matchup_id"]: r for r in csv.DictReader(open(odds_path))}
    return games, odds
 
 
def run(
    playoffs_only: bool,
    limit: int | None,
    out_path: str | None,
    agent_arms: bool = False,
    agent_sample: int = 25,
    model_backend: str = "anthropic",
) -> dict:
    games, odds = load_rows(playoffs_only, limit)
    source = get_source("real")
    tools = {t.name: t for t in build_tools(source)}
    predict = tools["predict_win_probability"]
 
    model_preds, home_preds, vegas_preds, rows = [], [], [], []
    skipped = 0
 
    for g in games:
        tip = parse_date(g["game_date"])
        as_of = (tip - timedelta(days=1)).isoformat()
        actual = 1 if g["winner"] == g["home"] else 0
 
        try:
            res = json.loads(
                predict.invoke(
                    {
                        "home_abbr": g["home"],
                        "away_abbr": g["away"],
                        "as_of_date": as_of,
                    }
                )
            )
        except Exception:
            skipped += 1
            continue
 
        p = res.get("home_win_prob")
        if p is None:
            skipped += 1
            continue
 
        model_preds.append((p, actual))
        home_preds.append(
            (0.55, actual)
        )  # naive prior, not 1.0 -- log loss needs a probability
 
        vp = vegas_home_prob(odds.get(g["game_id"], {}))
        if vp is not None:
            vegas_preds.append((vp, actual))
 
        rows.append(
            {
                "matchup_id": g["game_id"],
                "as_of": as_of,
                "home": g["home"],
                "away": g["away"],
                "model_home_prob": round(p, 4),
                "vegas_home_prob": round(vp, 4) if vp is not None else "",
                "actual_home_win": actual,
                "playoffs": g.get("playoffs", "0"),
            }
        )
 
    report = {
        "scope": "2026 playoffs" if playoffs_only else "2025-26 season",
        "games_scored": len(model_preds),
        "skipped": skipped,
        "classifier": metrics(model_preds),
        "baseline_always_home": metrics(home_preds),
        "baseline_vegas": metrics(vegas_preds),
    }
 
    if agent_arms:
        llm_metrics, hybrid_metrics, error = run_agent_arms(games, agent_sample, model_backend)
        if error:
            report["llm_direct"] = {"status": "not_available", "reason": error}
            report["hybrid"] = {"status": "not_available", "reason": error}
        else:
            report["llm_direct"] = llm_metrics
            report["hybrid"] = hybrid_metrics
    else:
        report["llm_direct"] = {"status": "not_run", "reason": "pass --agent-arms to run this"}
        report["hybrid"] = {"status": "not_run", "reason": "pass --agent-arms to run this"}
 
    if out_path and rows:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        report["per_game_csv"] = str(p)
 
    return report
 
 
def main() -> None:
    ap = argparse.ArgumentParser(description="Replay a season and score predictions")
    ap.add_argument(
        "--playoffs", action="store_true", help="Only the 2026 playoffs (the test set)"
    )
    ap.add_argument("--limit", type=int, help="Cap games (for a quick run)")
    ap.add_argument("--out", help="Write per-game predictions to this CSV")
    ap.add_argument(
        "--agent-arms",
        action="store_true",
        help="Also score llm_direct and hybrid (slow, runs on --agent-sample games only)",
    )
    ap.add_argument("--agent-sample", type=int, default=25, help="Games to run agent arms on")
    ap.add_argument(
        "--model",
        choices=["anthropic", "ollama"],
        default="anthropic",
        help="LLM backend for the agent arms",
    )
    args = ap.parse_args()
 
    report = run(
        args.playoffs,
        args.limit,
        args.out,
        agent_arms=args.agent_arms,
        agent_sample=args.agent_sample,
        model_backend=args.model,
    )
    print(json.dumps(report, indent=2))
 
    c, v = report["classifier"], report["baseline_vegas"]
    if c.get("n") and v.get("n"):
        print(
            f"\nclassifier {c['accuracy']:.1%} vs vegas {v['accuracy']:.1%} "
            f"({c['accuracy'] - v['accuracy']:+.1%})  |  "
            f"log loss {c['log_loss']:.3f} vs {v['log_loss']:.3f}"
        )
    for arm in ("llm_direct", "hybrid"):
        a = report[arm]
        if a.get("n"):
            print(f"{arm} {a['accuracy']:.1%} on {a['n']} sampled games "
                  f"(log loss {a['log_loss']:.3f}, brier {a['brier']:.3f})")
        elif a.get("status"):
            print(f"{arm}: {a['status']} -- {a.get('reason', '')}")
 
 
if __name__ == "__main__":
    main()
