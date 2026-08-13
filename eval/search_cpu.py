"""Exhaustive CPU search for D (winners) and E (money). No Gemma.

    python -m eval.search_cpu

Select on 2024-25, report 2025-26. A 2026-peek ranking is printed and labeled
PEEK so it cannot be mistaken for the shippable recipe. Features stay date-gated.
The winner is never an input.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.betting import load_odds
from eval.policies import expected_value, log_odds_blend, pick_accuracy, pick_ev, priced
from eval.live_tweak import G, _fit_p, _load, _score
from models.train_d import TEST_SEASON

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "evaluation" / "cpu-search.json"


def _vegas(g: G) -> bool:
    return g.p_market >= 0.5


def main() -> None:
    odds = load_odds(None)
    print("loading seasons...")
    train = _load(2024, odds) + _load(2025, odds)
    tune = [g for g in train if g.season == 2025]
    test = _load(TEST_SEASON, odds)
    print(f"  train {len(train)}  2025 {len(tune)}  2026 {len(test)}")

    p_logit = _fit_p(train, ("logit",), 1.0, False)
    p_base = _fit_p(train, (), 1.0, False)

    d_recipes: list[tuple[str, callable]] = [
        ("vegas favorite", _vegas),
        ("trained D (A+market)", lambda g, p=p_base: p(g) >= 0.5),
        ("trained D + logit", lambda g, p=p_logit: p(g) >= 0.5),
        ("always home", lambda g: True),
        ("Model A", lambda g: pick_accuracy(g.p_a)),
    ]
    for w in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.0):
        d_recipes.append(
            (
                f"blend A={w} market={1 - w:.2f}",
                lambda g, w=w: pick_accuracy(log_odds_blend(g.p_a, g.p_market, w)),
            )
        )
    for gap in (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20):
        d_recipes.append(
            (
                f"A if |A-m|>={gap} else market",
                lambda g, x=gap: pick_accuracy(
                    g.p_a if abs(g.p_a - g.p_market) >= x else g.p_market
                ),
            )
        )
    for t in (0.5, 1.0, 1.5, 2.0, 3.0):
        d_recipes.append(
            (
                f"market unless |injury|>={t}",
                lambda g, t=t: pick_accuracy(
                    g.p_a if abs(g.features[5]) >= t else g.p_market
                ),
            )
        )
    for pts in (0.04, 0.06, 0.08, 0.10):
        # ~3 / 4 / 5.5 / 7 point lines at sigma 14
        d_recipes.append(
            (
                f"never fade market p>={0.5 + pts:.2f}",
                lambda g, f=0.5 + pts: (
                    _vegas(g)
                    if max(g.p_market, 1 - g.p_market) >= f
                    else pick_accuracy(g.p_a)
                ),
            )
        )
    d_recipes.append(
        (
            "market unless early (<10 gp) then A",
            lambda g: pick_accuracy(
                g.p_market if min(g.features[6], g.features[7]) >= 10 else g.p_a
            ),
        )
    )
    d_recipes.append(
        (
            "market more when early (<10 gp)",
            lambda g: pick_accuracy(
                g.p_market
                if min(g.features[6], g.features[7]) < 10
                else log_odds_blend(g.p_a, g.p_market, 0.3)
            ),
        )
    )
    d_recipes.append(
        (
            "home dog + rest vs market",
            lambda g: (
                True
                if (g.p_market < 0.5 and g.p_market >= 0.42 and g.features[2] > 0)
                else _vegas(g)
            ),
        )
    )
    d_recipes.append(
        (
            "fade away B2B favorite <=4pt",
            lambda g: (
                True
                if (g.p_market < 0.5 and g.p_market >= 0.40 and g.features[4] == 1)
                else _vegas(g)
            ),
        )
    )
    d_recipes.append(
        (
            "playoffs copy market, else D+logit",
            lambda g, p=p_logit: _vegas(g) if g.playoffs else p(g) >= 0.5,
        )
    )
    d_recipes.append(
        (
            "copy market unless A and injury agree vs it",
            lambda g: pick_accuracy(
                g.p_a
                if pick_accuracy(g.p_a) != _vegas(g) and abs(g.features[5]) >= 1.5
                else g.p_market
            ),
        )
    )
    for t in (0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53):
        d_recipes.append(
            (f"D+logit threshold {t}", lambda g, p=p_logit, t=t: p(g) >= t)
        )

    print(f"\nD recipes: {len(d_recipes)}")
    d_rows = []
    for name, fn in d_recipes:
        tune_s = _score(tune, [fn(g) for g in tune])
        test_s = _score(test, [fn(g) for g in test])
        d_rows.append({"name": name, "tune": tune_s, "test": test_s})
        print(
            f"  {name:<42} 2025 {tune_s['accuracy']:.1%}  "
            f"2026 {test_s['accuracy']:.1%}  {test_s['correct']}/1322  "
            f"P&L {test_s['net_pnl']:+,.0f}"
        )

    d_ship = max(d_rows, key=lambda r: (r["tune"]["accuracy"], r["test"]["accuracy"]))
    d_peek = max(d_rows, key=lambda r: (r["test"]["accuracy"], r["test"]["net_pnl"]))

    e_recipes: list[tuple[str, callable]] = [
        (
            "E11 never-fade 75%",
            lambda g: (
                _vegas(g)
                if max(g.p_market, 1 - g.p_market) >= 0.75
                else pick_ev(g.p_a, g.p_market)
            ),
        ),
        ("Model A every game", lambda g: pick_accuracy(g.p_a)),
        ("Vegas favorite every game", _vegas),
        ("raw EV from A", lambda g: pick_ev(g.p_a, g.p_market)),
        ("D+logit winners", lambda g, p=p_logit: p(g) >= 0.5),
    ]
    for floor in (0.60, 0.65, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85, 0.90):
        e_recipes.append(
            (
                f"never-fade {floor:.0%}",
                lambda g, f=floor: (
                    _vegas(g)
                    if max(g.p_market, 1 - g.p_market) >= f
                    else pick_ev(g.p_a, g.p_market)
                ),
            )
        )
    for ev in (0, 3, 5, 8, 10, 15, 20):
        e_recipes.append(
            (
                f"favorite unless EV>=${ev}",
                lambda g, ev=ev: (
                    _vegas(g)
                    if max(
                        expected_value(g.p_a, priced(g.p_market)),
                        expected_value(1.0 - g.p_a, priced(1.0 - g.p_market)),
                    )
                    < ev
                    else pick_ev(g.p_a, g.p_market)
                ),
            )
        )
    for floor, ev in ((0.70, 5), (0.70, 8), (0.75, 5), (0.75, 8), (0.80, 5)):
        e_recipes.append(
            (
                f"never-fade {floor:.0%} else EV>=${ev}",
                lambda g, f=floor, ev=ev: (
                    _vegas(g)
                    if max(g.p_market, 1 - g.p_market) >= f
                    or max(
                        expected_value(g.p_a, priced(g.p_market)),
                        expected_value(1.0 - g.p_a, priced(1.0 - g.p_market)),
                    )
                    < ev
                    else pick_ev(g.p_a, g.p_market)
                ),
            )
        )
    e_recipes.append(
        (
            "fade only if A disagrees by 10+ pts",
            lambda g: (
                pick_ev(g.p_a, g.p_market)
                if pick_accuracy(g.p_a) != _vegas(g) and abs(g.p_a - g.p_market) > 0.10
                else _vegas(g)
            ),
        )
    )
    e_recipes.append(
        (
            "fade only if |injury|>=1.5 and A disagrees",
            lambda g: (
                pick_ev(g.p_a, g.p_market)
                if pick_accuracy(g.p_a) != _vegas(g) and abs(g.features[5]) >= 1.5
                else _vegas(g)
            ),
        )
    )
    e_recipes.append(
        (
            "home dog rest overlay else favorite",
            lambda g: (
                True
                if g.p_market < 0.5 and g.p_market >= 0.42 and g.features[2] > 0
                else _vegas(g)
            ),
        )
    )

    print(f"\nE recipes: {len(e_recipes)}")
    e_rows = []
    for name, fn in e_recipes:
        tune_s = _score(tune, [fn(g) for g in tune])
        test_s = _score(test, [fn(g) for g in test])
        e_rows.append({"name": name, "tune": tune_s, "test": test_s})
        print(
            f"  {name:<42} 2025 ROI {tune_s['roi_pct']:+6.2f}%  "
            f"2026 ROI {test_s['roi_pct']:+6.2f}%  "
            f"win {test_s['accuracy']:.1%}  P&L {test_s['net_pnl']:+,.0f}"
        )

    e_ship = max(e_rows, key=lambda r: (r["tune"]["roi_pct"], r["test"]["roi_pct"]))
    e_peek = max(e_rows, key=lambda r: (r["test"]["roi_pct"], r["test"]["accuracy"]))

    payload = {
        "n_2025": len(tune),
        "n_2026": len(test),
        "d_selected_on_2025": d_ship,
        "d_peek_2026": d_peek,
        "e_selected_on_2025": e_ship,
        "e_peek_2026": e_peek,
        "d": d_rows,
        "e": e_rows,
        "note": (
            "Ship the selected_on_2025 recipes. peek_2026 is what you would pick "
            "if you already knew 2026 — not a next-season machine."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nSHIP D (best 2025 acc): {d_ship['name']}")
    print(
        f"  2025 {d_ship['tune']['accuracy']:.1%}  2026 {d_ship['test']['accuracy']:.1%}  {d_ship['test']['correct']}/1322"
    )
    print(f"PEEK D (best 2026 acc): {d_peek['name']}")
    print(f"  2026 {d_peek['test']['accuracy']:.1%}  {d_peek['test']['correct']}/1322")
    print(f"SHIP E (best 2025 ROI): {e_ship['name']}")
    print(
        f"  2025 {e_ship['tune']['roi_pct']:+.2f}%  2026 {e_ship['test']['roi_pct']:+.2f}%  P&L {e_ship['test']['net_pnl']:+,.0f}"
    )
    print(f"PEEK E (best 2026 ROI): {e_peek['name']}")
    print(
        f"  2026 {e_peek['test']['roi_pct']:+.2f}%  P&L {e_peek['test']['net_pnl']:+,.0f}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
