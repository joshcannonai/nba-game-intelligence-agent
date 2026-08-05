"""Remove one tool at a time and measure what it was worth.

    python -m eval.ablation --sample 50 --model ollama
    python -m eval.ablation --sample 50 --tools retrieve_injuries retrieve_team_form

The advisor asked for this on 2026-08-04: run the agent on a fixed set of games,
then run it again with a tool withheld, and the accuracy change is that tool's
contribution. It is also the sanity check he described -- if you strip the tools
and accuracy barely moves, the tools were never doing the work and the headline
number is coming from somewhere you have not accounted for.

WHY IT IS PAIRED. Every arm runs the SAME games in the SAME order with the SAME
seed. A tool's effect is small and 50 games is a wide band, so an unpaired
comparison would mostly measure which games each arm happened to draw. Paired,
the games cancel and what is left is the tool.

WHY IT IS SMALL. Each game is one local LLM call, roughly 40 seconds. One
baseline plus two ablations over 50 games is about 100 minutes. That is the
reason the advisor said to do this on a smaller set than the headline run, and
the reason n is printed next to every number here.

WHICH ARM TO ABLATE, AND WHY IT IS B BY DEFAULT. A first run on arm C returned
byte-identical probabilities with and without each tool. That is the skills layer
doing its job, not a broken harness: `skills/predict_win_probability.md` tells the
agent to treat the fitted model's number as the answer unless it has a concrete
reason to move, so pulling a retrieval tool does not move the output. Arm C
therefore measures the model, not the tools. Arm B has no model tool, so the
retrieval tools are the only thing it has, and the accuracy change when one is
removed is that tool's actual contribution. Run C only to demonstrate the
deference itself.

WHAT IT CANNOT TELL YOU. Removing a tool changes two things at once: the data
the agent can reach, and the paragraph of rules that came with it (the skills
block is composed from the tools actually granted). We do not separate those,
and the write-up should not claim otherwise.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.sources import get_source  # noqa: E402
from agent.tools import build_tools  # noqa: E402
from eval.replay import metrics  # noqa: E402
from eval.three_arms import extract_prob, load_test_games  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "results_ablation.csv"

# Defaults chosen because they are the two the report makes claims about: the
# injury list drove the negative result in §7.4, and current form is what the
# prior-season ratings fall back from.
DEFAULT_TOOLS = ("retrieve_injuries", "retrieve_team_form")


def run_arm(
    rows, *, without: tuple[str, ...], backend: str, label: str, include_model: bool
) -> dict:
    """One pass over the sample with `without` withheld. Returns game_id -> prob."""
    from agent.run import run_matchup

    source = get_source("real")
    per_game: dict[str, float] = {}
    t0 = time.time()

    for i, r in enumerate(rows, 1):
        as_of = (r.game_date - timedelta(days=1)).isoformat()
        try:
            text = run_matchup(
                r.game_id,
                as_of,
                source,
                backend,
                include_model=include_model,
                without=without,
            )
            p = extract_prob(text)
        except Exception as exc:
            print(f"  [{label}] {r.game_id} raised {type(exc).__name__}: {exc}")
            p = None

        if p is not None and 0.0 <= p <= 1.0:
            per_game[r.game_id] = p

        rate = (time.time() - t0) / i
        print(
            f"  [{label}] {i}/{len(rows)} {r.game_id} "
            f"p={p if p is None else round(p, 3)} "
            f"({rate:.0f}s/game, ~{rate * (len(rows) - i) / 60:.0f}m left)",
            flush=True,
        )
    return per_game


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="ollama")
    ap.add_argument("--tools", nargs="*", default=list(DEFAULT_TOOLS))
    ap.add_argument(
        "--arm",
        choices=["B", "C"],
        default="B",
        help="B (no model tool) is the meaningful one; see the module docstring",
    )
    args = ap.parse_args()

    available = {t.name for t in build_tools(get_source("mock"))}
    unknown = set(args.tools) - available
    if unknown:
        raise SystemExit(
            f"unknown tools: {sorted(unknown)}\navailable: {sorted(available)}"
        )

    # Same sampling as eval/three_arms.py, so an ablation run is comparable to
    # the headline run rather than drawing its own convenient subset.
    rows = load_test_games(False)
    rng = random.Random(args.seed)
    rows = sorted(rng.sample(rows, min(args.sample, len(rows))), key=lambda r: r.game_date)
    use_model = args.arm == "C"
    n_arms = 1 + len(args.tools)
    print(
        f"arm {args.arm}, {len(rows)} games, seed {args.seed}, {n_arms} passes "
        f"(~{n_arms * len(rows) * 40 / 60:.0f} minutes)\n"
    )

    arms: dict[str, dict[str, float]] = {}
    print("baseline: every tool")
    arms["all tools"] = run_arm(
        rows, without=(), backend=args.model, label="all", include_model=use_model
    )
    for tool in args.tools:
        print(f"\nwithout {tool}")
        arms[f"without {tool}"] = run_arm(
            rows,
            without=(tool,),
            backend=args.model,
            label=tool.replace("retrieve_", "-"),
            include_model=use_model,
        )

    # Score every arm on the games ALL of them answered, so the comparison is
    # paired. An arm that failed on a game must not be scored on a different set.
    common = set.intersection(*(set(a) for a in arms.values())) if arms else set()
    truth = {r.game_id: r.home_won for r in rows}
    print(f"\n{len(common)} games answered by every arm\n")

    base = None
    print(f"{'arm':<34} {'accuracy':>9} {'log loss':>9} {'vs baseline':>12}")
    results = []
    for label, per_game in arms.items():
        pairs = [(per_game[g], truth[g]) for g in sorted(common)]
        m = metrics(pairs)
        if base is None:
            base = m["accuracy"]
            delta = ""
        else:
            delta = f"{(m['accuracy'] - base) * 100:+.1f} pts"
        print(f"{label:<34} {m['accuracy']:>8.1%} {m['log_loss']:>9.3f} {delta:>12}")
        results.append({"arm": label, "n": len(pairs), **m})

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print(
        "\nA tool that matters makes accuracy DROP when removed. A negative "
        "'vs baseline' is the tool earning its place; a positive one means the "
        "agent did better without it, which is a finding, not a bug."
    )


if __name__ == "__main__":
    main()
