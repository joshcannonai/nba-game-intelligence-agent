"""Score the three arms against each other on the same games.

This is the experiment the whole project exists to run:

    arm A   model only     logistic regression, no language model involved
    arm B   agent only     the LLM reasons from the retrieval tools, no model
    arm C   agent + model  the LLM gets the model's number as one input

    hypothesis: C beats both A and B.

If C wins, an agent that can explain itself costs nothing in accuracy. If A
wins, the explanation is costing us something real and we should say so. Either
result is worth writing up; the failure mode is a fuzzy comparison that supports
whichever conclusion we already wanted.

Two reference points, because 66% accuracy means nothing on its own:

    always-home  home teams win ~55% of NBA games. Below this we have negative
                 information.
    Vegas        the de-vigged closing line. The real bar, and not one we expect
                 to clear -- the market has money and injury beat reporters and
                 we have a laptop.

    python -m eval.three_arms                        # arm A + baselines, all 1,322
    python -m eval.three_arms --arms abc --sample 40 # all three, 40 games
    python -m eval.three_arms --arms abc --playoffs --model ollama

WHY THE SAMPLE. Arms B and C call a language model once per game, about 40
seconds each locally. The full season in all three arms is roughly 30 hours. So
B and C run on a sample -- and every arm is then scored on THAT SAME SAMPLE, not
on its own convenient subset. A paired comparison on 40 games beats an unpaired
one on 1,322, because the arms differ by exactly one tool and nothing else.

Small n is a real limit and the report has to own it: 40 games is roughly a
+/-8% band on accuracy, which is wider than the gap we are looking for. Treat a
sample run as directional, and say "n=40" out loud next to any number from it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.sources import closing_line, get_source, parse_matchup_id  # noqa: E402
from eval.replay import metrics, vegas_home_prob  # noqa: E402
from models.features import build_season  # noqa: E402
from models.predict import model_available, score_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Home-court base rate, used as arm "always-home". Not 1.0: log loss needs a
# probability, and a baseline that is certain every time scores infinitely badly
# the first time it is wrong.
HOME_PRIOR = 0.55


def load_test_games(playoffs_only: bool) -> list:
    rows, report = build_season(2026)
    rows = [r for r in rows if r.home_won is not None]
    if playoffs_only:
        rows = [r for r in rows if r.playoffs]
    print(
        f"test set: {len(rows)} games "
        f"({'2026 playoffs' if playoffs_only else '2025-26 season'}), "
        f"{report['games_with_injury_signal']} with injury signal"
    )
    return rows


def vegas_for(rows) -> dict:
    """Closing line per game, read through the same gated accessor the eval uses."""
    out = {}
    for r in rows:
        away, home, game_date = parse_matchup_id(r.game_id)
        line = closing_line(away, home, game_date, game_date)
        if line.get("status") == "ok":
            p = vegas_home_prob(line)
            if p is not None:
                out[r.game_id] = p
    return out


_PROB = re.compile(r'"home_win_prob"\s*:\s*([0-9.]+)')


def extract_prob(text: str) -> float | None:
    """Pull home_win_prob out of whatever the LLM actually returned.

    Models wrap JSON in prose, in ``` fences, or emit it slightly malformed.
    Try strict parsing first, then a regex, then give up -- an unparseable
    answer is recorded as a skip, never silently coerced to 0.5, which would
    quietly drag an arm toward the baseline and flatter it.
    """
    try:
        obj = json.loads(text)
        v = obj.get("home_win_prob")
        if isinstance(v, (int, float)):
            return float(v)
    except Exception:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        try:
            v = json.loads(fence.group(1)).get("home_win_prob")
            if isinstance(v, (int, float)):
                return float(v)
        except Exception:
            pass
    m = _PROB.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def run_agent_arm(rows, *, include_model: bool, backend: str, label: str) -> tuple:
    """Run the LLM agent over the sample. Returns (preds, per_game, failures)."""
    from agent.run import run_matchup

    source = get_source("real")
    preds, per_game, failures = [], {}, 0
    t0 = time.time()

    for i, r in enumerate(rows, 1):
        as_of = (r.game_date - timedelta(days=1)).isoformat()
        try:
            text = run_matchup(
                r.game_id, as_of, source, backend, include_model=include_model
            )
            p = extract_prob(text)
        except Exception as exc:
            print(f"  [{label}] {r.game_id} raised {type(exc).__name__}: {exc}")
            p = None

        if p is None or not (0.0 <= p <= 1.0):
            failures += 1
        else:
            preds.append((p, r.home_won))
            per_game[r.game_id] = p

        rate = (time.time() - t0) / i
        print(
            f"  [{label}] {i}/{len(rows)} {r.game_id} "
            f"p={p if p is None else round(p, 3)} "
            f"({rate:.1f}s/game, ~{rate * (len(rows) - i) / 60:.0f}m left)",
            flush=True,
        )

    return preds, per_game, failures


def _sign_test(successes: int, trials: int) -> float:
    """One-sided binomial p under 'the override is a coin flip'."""
    if trials == 0:
        return 1.0
    return sum(math.comb(trials, k) for k in range(successes, trials + 1)) / 2**trials


def _override_analysis(rows, per_game_probs) -> None:
    """When arm C disagrees with the model it was handed, who was right?

    This is the measurement that actually survives a small sample. Comparing
    headline accuracies asks whether these 40 games flattered one arm; this asks
    a paired question -- on the games where the agent overruled the model, how
    often did overruling help? Every game contributes only when the two differ,
    so a sample that happens to be easy or hard for both cancels out.
    """
    a_probs, c_probs = per_game_probs.get("A"), per_game_probs.get("C")
    if not a_probs or not c_probs:
        return

    agree = overrides = agent_right = model_right = 0
    worst = []
    for r in rows:
        a, c = a_probs.get(r.game_id), c_probs.get(r.game_id)
        if a is None or c is None:
            continue
        if (a >= 0.5) == (c >= 0.5):
            agree += 1
            continue
        overrides += 1
        agent_right += (c >= 0.5) == bool(r.home_won)
        model_right += (a >= 0.5) == bool(r.home_won)
        worst.append((abs(a - c), r.game_id, a, c, r.home_won))

    print(f"\n{'-' * 66}\nwhen the agent overruled the model")
    print(f"  agreed on          {agree}")
    print(f"  overruled on       {overrides}")
    if not overrides:
        return
    print(f"    model was right  {model_right}")
    print(f"    agent was right  {agent_right}")
    p = _sign_test(max(agent_right, model_right), overrides)
    verdict = "not distinguishable from chance" if p > 0.05 else f"p = {p:.3f}"
    print(f"  overruling helped {agent_right}/{overrides} times -- {verdict}")

    worst.sort(reverse=True)
    if worst:
        print("  biggest reversals (model -> agent, actual):")
        for _, game_id, a, c, y in worst[:3]:
            who = "home won" if y else "home lost"
            print(f"    {game_id:<24} {a:.3f} -> {c:.3f}   {who}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Three-arm comparison")
    ap.add_argument(
        "--arms",
        default="a",
        help="Which arms to run: 'a' (model only, fast) or 'abc' (adds the LLM arms)",
    )
    ap.add_argument(
        "--sample", type=int, help="Score this many games (required for b/c)"
    )
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed")
    ap.add_argument("--playoffs", action="store_true", help="Playoffs only")
    ap.add_argument("--model", default="ollama", choices=["ollama", "anthropic"])
    ap.add_argument("--out", help="Write per-game predictions here")
    args = ap.parse_args()

    arms = set(args.arms.lower())
    needs_llm = bool(arms & {"b", "c"})

    if not model_available():
        raise SystemExit(
            "No models/win_probability.json. Run `python -m models.train` first."
        )

    rows = load_test_games(args.playoffs)

    if args.sample and args.sample < len(rows):
        rng = random.Random(args.seed)
        rows = sorted(rng.sample(rows, args.sample), key=lambda r: r.game_date)
        print(f"sampled {len(rows)} games (seed={args.seed})")
    elif needs_llm:
        raise SystemExit(
            f"Refusing to run {len(rows)} games through a language model -- that is "
            f"roughly {len(rows) * 40 / 3600:.0f} hours per arm.\n"
            "Pass --sample N (try 40) to score a fixed random subset."
        )

    vegas = vegas_for(rows)
    print(f"vegas closing line available for {len(vegas)}/{len(rows)} games\n")

    results: dict[str, dict] = {}
    per_game_probs: dict[str, dict] = {}

    # ---- arm A and the two baselines are free, so they always run.
    a_preds = [(score_features(r.features), r.home_won) for r in rows]
    results["A_model_only"] = metrics(a_preds)
    per_game_probs["A"] = {r.game_id: p for r, (p, _) in zip(rows, a_preds)}

    results["baseline_always_home"] = metrics([(HOME_PRIOR, r.home_won) for r in rows])
    v_preds = [(vegas[r.game_id], r.home_won) for r in rows if r.game_id in vegas]
    results["baseline_vegas"] = metrics(v_preds)

    if "b" in arms:
        print("arm B -- agent alone, no model tool")
        preds, pg, fails = run_agent_arm(
            rows, include_model=False, backend=args.model, label="B"
        )
        results["B_agent_only"] = metrics(preds) | {"unparseable": fails}
        per_game_probs["B"] = pg

    if "c" in arms:
        print("arm C -- agent plus the model's number")
        preds, pg, fails = run_agent_arm(
            rows, include_model=True, backend=args.model, label="C"
        )
        results["C_agent_plus_model"] = metrics(preds) | {"unparseable": fails}
        per_game_probs["C"] = pg

    print("\n" + "=" * 66)
    print(f"{'arm':<26}{'n':>5}{'acc':>9}{'logloss':>10}{'brier':>9}")
    print("-" * 66)
    for name, m in results.items():
        if not m.get("n"):
            continue
        print(
            f"{name:<26}{m['n']:>5}{m['accuracy']:>9.1%}"
            f"{m['log_loss']:>10.4f}{m['brier']:>9.4f}"
        )
    print("=" * 66)

    a = results.get("A_model_only", {})
    c = results.get("C_agent_plus_model", {})
    b = results.get("B_agent_only", {})
    if a.get("n") and c.get("n"):
        d = c["accuracy"] - a["accuracy"]
        print(f"\nhypothesis (C > A): {d:+.1%} accuracy, n={c['n']}")
        # +/-1 se on a proportion. At n=40 this is ~8 points, which is the
        # point: print it so nobody reads a 3-point gap as a finding.
        se = math.sqrt(0.25 / c["n"])
        print(
            f"  one standard error at this n is +/-{se:.1%} -- "
            + ("inside the noise." if abs(d) < se else "outside one SE.")
        )
    if b.get("n") and c.get("n"):
        print(
            f"C vs B (does the model help the agent): {c['accuracy'] - b['accuracy']:+.1%}"
        )

    _override_analysis(rows, per_game_probs)

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as fh:
            cols = ["game_id", "game_date", "home", "away", "actual_home_win", "vegas"]
            cols += [f"arm_{k}" for k in sorted(per_game_probs)]
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                row = {
                    "game_id": r.game_id,
                    "game_date": r.game_date.isoformat(),
                    "home": r.home,
                    "away": r.away,
                    "actual_home_win": r.home_won,
                    # 6dp, not 4. At 4 a probability of 0.499967 writes as 0.5000 and the
# scorer then reads it as a home pick, which cost exactly one game and
# made this file disagree with models/win_probability.json by 0.08%.
                    "vegas": round(vegas[r.game_id], 6) if r.game_id in vegas else "",
                }
                for k, probs in per_game_probs.items():
                    row[f"arm_{k}"] = (
                        round(probs[r.game_id], 6) if r.game_id in probs else ""
                    )
                w.writerow(row)
        print(f"\nwrote {p}")

    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
