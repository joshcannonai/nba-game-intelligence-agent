"""Fifteen gated D experiments and fifteen gated E experiments.

    python -m eval.search_de

WHY THIS IS NOT 15 FULL-SEASON GEMMA LOOPS. Each Gemma game is ~40s. Fifteen
D runs plus fifteen E runs on 1,322 games would be months. These experiments
are the thing that can actually search: every completed 2025-26 game, date-gated
features, market visible only to D/E, outcomes joined after the pick.

HOW WE AVOID FITTING THE TEST SEASON. Each recipe is a fixed rule or a model
fit on 2024-25 only. We *select* the winner on 2024-25 (D: accuracy, E: ROI)
and *report* 2025-26. That is the "would you follow this next season" number.
Picking the best of 15 on 2026 itself would be the cheat.

A/B/C never see these recipes. They still have no betting-line tool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from eval.betting import fair_home_prob, load_odds, odds_for_matchup
from eval.policies import (
    expected_value,
    log_odds_blend,
    pick_accuracy,
    priced,
    score_d_features,
    settle_pick,
)
from models.features import FEATURE_NAMES, build_season
from models.predict import score_features
from models.train_d import TEST_SEASON, TRAIN_SEASONS, fit_logistic

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "docs" / "evaluation" / "de-search-log.json"
INJURY_I = FEATURE_NAMES.index("injury_weight_diff")
GP_HOME_I = FEATURE_NAMES.index("home_games_played")
GP_AWAY_I = FEATURE_NAMES.index("away_games_played")
FORM_I = FEATURE_NAMES.index("form_margin_diff")
WINPCT_I = FEATURE_NAMES.index("win_pct_diff")


@dataclass
class Game:
    game_id: str
    season: int
    playoffs: bool
    home: str
    away: str
    cutoff: str
    features: tuple[float, ...]
    p_a: float
    p_market: float
    home_won: bool


def _load_season(season: int, odds) -> list[Game]:
    rows, _ = build_season(season)
    out = []
    for row in rows:
        if row.home_won is None:
            continue
        odds_row = odds_for_matchup(row.away, row.home, row.game_date, odds)
        p_market = fair_home_prob(odds_row) if odds_row else None
        if p_market is None:
            continue
        out.append(
            Game(
                game_id=row.game_id,
                season=season,
                playoffs=row.playoffs,
                home=row.home,
                away=row.away,
                cutoff=(row.game_date - timedelta(days=1)).isoformat(),
                features=tuple(row.features),
                p_a=score_features(row.features),
                p_market=p_market,
                home_won=bool(row.home_won),
            )
        )
    return out


def _tweak(features: tuple[float, ...], *, injury_scale=1.0, drop_gp=False):
    f = list(features)
    f[INJURY_I] *= injury_scale
    if drop_gp:
        f[GP_HOME_I] = 0.0
        f[GP_AWAY_I] = 0.0
    return tuple(f)


def _score_a(features) -> float:
    return score_features(features)


def _fit_market_model(games: list[Game], *, injury_scale=1.0, drop_gp=False):
    import numpy as np

    X, y = [], []
    for g in games:
        feats = _tweak(g.features, injury_scale=injury_scale, drop_gp=drop_gp)
        X.append(tuple(feats) + (g.p_market,))
        y.append(int(g.home_won))
    coef, intercept, mean, scale = fit_logistic(np.array(X, float), np.array(y, float))
    spec = {
        "coefficients": [float(c) for c in coef],
        "intercept": float(intercept),
        "scaler_mean": [float(m) for m in mean],
        "scaler_scale": [float(s) for s in scale],
    }
    return spec


def _p_from_spec(g: Game, spec, *, injury_scale=1.0, drop_gp=False) -> float:
    feats = _tweak(g.features, injury_scale=injury_scale, drop_gp=drop_gp)
    return score_d_features(tuple(feats) + (g.p_market,), spec)


def _summarize(games: list[Game], picks: list[bool]) -> dict:
    n = len(games)
    correct = sum(int(p == g.home_won) for p, g in zip(picks, games))
    pnl = 0.0
    for pick_home, g in zip(picks, games):
        pnl += settle_pick(pick_home, g.home_won, g.p_market)["net_pnl"]
    staked = n * 100.0
    return {
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "net_pnl": round(pnl, 2),
        "roi_pct": round(100.0 * pnl / staked, 2) if staked else None,
        "ending_cash_if_funded": round(staked + pnl, 2),
    }


# --- D recipes: pick the winner. Market is allowed. ---


def d_recipes(train: list[Game]) -> list[dict]:
    """Exactly 15. Each is named and frozen before 2026 is scored."""
    spec = _fit_market_model(train)
    spec_inj0 = _fit_market_model(train, injury_scale=0.0)
    spec_inj2 = _fit_market_model(train, injury_scale=2.0)
    spec_inj05 = _fit_market_model(train, injury_scale=0.5)
    spec_nogp = _fit_market_model(train, drop_gp=True)
    spec_star = _fit_market_model(train, injury_scale=3.0)

    def trained(g, s=spec, **kw):
        return pick_accuracy(_p_from_spec(g, s, **kw))

    recipes = [
        {
            "id": "D01_trained_market",
            "goal": "logistic on A-features + closing line, fit 2024-25",
            "pick": lambda g: trained(g),
        },
        {
            "id": "D02_vegas_favorite",
            "goal": "copy the closing favorite — the almighty-market hypothesis",
            "pick": lambda g: pick_accuracy(g.p_market),
        },
        {
            "id": "D03_a_only",
            "goal": "ignore the market; same pick as Model A",
            "pick": lambda g: pick_accuracy(g.p_a),
        },
        {
            "id": "D04_blend_market85",
            "goal": "log-odds blend, 15% A / 85% market",
            "pick": lambda g: pick_accuracy(log_odds_blend(g.p_a, g.p_market, 0.15)),
        },
        {
            "id": "D05_blend_market65",
            "goal": "log-odds blend, 35% A / 65% market",
            "pick": lambda g: pick_accuracy(log_odds_blend(g.p_a, g.p_market, 0.35)),
        },
        {
            "id": "D06_blend_50",
            "goal": "log-odds blend, 50/50",
            "pick": lambda g: pick_accuracy(log_odds_blend(g.p_a, g.p_market, 0.50)),
        },
        {
            "id": "D07_blend_a65",
            "goal": "log-odds blend, 65% A / 35% market",
            "pick": lambda g: pick_accuracy(log_odds_blend(g.p_a, g.p_market, 0.65)),
        },
        {
            "id": "D08_blend_a85",
            "goal": "log-odds blend, 85% A / 15% market",
            "pick": lambda g: pick_accuracy(log_odds_blend(g.p_a, g.p_market, 0.85)),
        },
        {
            "id": "D09_a_when_extreme",
            "goal": "use A only when |p_A-0.5|>=0.15, else the market",
            "pick": lambda g: pick_accuracy(
                g.p_a if abs(g.p_a - 0.5) >= 0.15 else g.p_market
            ),
        },
        {
            "id": "D10_injury_off",
            "goal": "retrain with injury feature zeroed — Kyrie vs Lively test, none",
            "pick": lambda g: trained(g, spec_inj0, injury_scale=0.0),
        },
        {
            "id": "D11_injury_half",
            "goal": "retrain with injury weight * 0.5 (README: importance over-penalises)",
            "pick": lambda g: trained(g, spec_inj05, injury_scale=0.5),
        },
        {
            "id": "D12_injury_double",
            "goal": "retrain with injury weight * 2 — heavier star absences",
            "pick": lambda g: trained(g, spec_inj2, injury_scale=2.0),
        },
        {
            "id": "D13_injury_triple_stars",
            "goal": "retrain with injury weight * 3 — max Kyrie-vs-bench gap",
            "pick": lambda g: trained(g, spec_star, injury_scale=3.0),
        },
        {
            "id": "D14_drop_games_played",
            "goal": "retrain without home/away games-played (known schedule artifact)",
            "pick": lambda g: trained(g, spec_nogp, drop_gp=True),
        },
        {
            "id": "D15_market_unless_a_disagrees_hard",
            "goal": "market unless |p_A - p_market| >= 0.12, then A",
            "pick": lambda g: pick_accuracy(
                g.p_a if abs(g.p_a - g.p_market) >= 0.12 else g.p_market
            ),
        },
    ]
    assert len(recipes) == 15, len(recipes)
    return recipes


# --- E recipes: pick the moneyline. Always $100. ---


def e_recipes(best_d_pick, d_spec) -> list[dict]:
    def ev_pick(p_belief: float, p_market: float, edge: float = 0.0) -> bool:
        dec_h, dec_a = priced(p_market), priced(1.0 - p_market)
        ev_h = expected_value(p_belief, dec_h)
        ev_a = expected_value(1.0 - p_belief, dec_a)
        if abs(ev_h - ev_a) < edge * 100:
            return p_belief >= 0.5
        return ev_h > ev_a

    def fade_heavy(g: Game, cutoff_dec: float) -> bool:
        fav_is_home = g.p_market >= 0.5
        fav_dec = priced(g.p_market if fav_is_home else 1.0 - g.p_market)
        if fav_dec <= cutoff_dec:
            return ev_pick(g.p_a, g.p_market)
        return pick_accuracy(g.p_a)

    recipes = [
        {
            "id": "E01_ev_from_a",
            "goal": "max EV using Model A as belief vs the book",
            "pick": lambda g: ev_pick(g.p_a, g.p_market),
        },
        {
            "id": "E02_ev_from_d_trained",
            "goal": "max EV using D's market-aware p (expected to collapse to vig)",
            "pick": lambda g: ev_pick(_p_from_spec(g, d_spec), g.p_market),
        },
        {
            "id": "E03_ev_blend50",
            "goal": "max EV using 50/50 A+market belief",
            "pick": lambda g: ev_pick(
                log_odds_blend(g.p_a, g.p_market, 0.5), g.p_market
            ),
        },
        {
            "id": "E04_a_pick_always",
            "goal": "bet A's winner every game (money is tracked, not optimised)",
            "pick": lambda g: pick_accuracy(g.p_a),
        },
        {
            "id": "E05_d_pick_always",
            "goal": "bet D's winner every game — replica of D, scored as money",
            "pick": best_d_pick,
        },
        {
            "id": "E06_ev_edge_2pp",
            "goal": "max EV, but if the two sides are within $2, take A's side",
            "pick": lambda g: ev_pick(g.p_a, g.p_market, edge=0.02),
        },
        {
            "id": "E07_ev_edge_5pp",
            "goal": "max EV, $5 indifference band falls back to A",
            "pick": lambda g: ev_pick(g.p_a, g.p_market, edge=0.05),
        },
        {
            "id": "E08_ev_edge_10pp",
            "goal": "max EV, $10 indifference band falls back to A",
            "pick": lambda g: ev_pick(g.p_a, g.p_market, edge=0.10),
        },
        {
            "id": "E09_fade_only_minus600ish",
            "goal": "only hunt EV when the favorite is juiced to decimal <= 1.20",
            "pick": lambda g: fade_heavy(g, 1.20),
        },
        {
            "id": "E10_fade_only_minus400ish",
            "goal": "only hunt EV when the favorite is juiced to decimal <= 1.30",
            "pick": lambda g: fade_heavy(g, 1.30),
        },
        {
            "id": "E11_never_fade_75",
            "goal": "never fade a 75%+ favorite; EV otherwise",
            "pick": lambda g: (
                pick_accuracy(g.p_market)
                if max(g.p_market, 1 - g.p_market) >= 0.75
                else ev_pick(g.p_a, g.p_market)
            ),
        },
        {
            "id": "E12_never_fade_80",
            "goal": "never fade an 80%+ favorite; EV otherwise",
            "pick": lambda g: (
                pick_accuracy(g.p_market)
                if max(g.p_market, 1 - g.p_market) >= 0.80
                else ev_pick(g.p_a, g.p_market)
            ),
        },
        {
            "id": "E13_dog_only_if_a_likes_dog",
            "goal": "fade only when A also prefers the dog; else bet the favorite",
            "pick": lambda g: (
                ev_pick(g.p_a, g.p_market)
                if pick_accuracy(g.p_a) != pick_accuracy(g.p_market)
                else pick_accuracy(g.p_market)
            ),
        },
        {
            "id": "E14_shrink_a_toward_55",
            "goal": "dampen A's confidence (0.7*A + 0.3*0.55) then max EV",
            "pick": lambda g: ev_pick(0.7 * g.p_a + 0.3 * 0.55, g.p_market),
        },
        {
            "id": "E15_form_tilted_a",
            "goal": "nudge A's p by rolling margin, then max EV",
            "pick": lambda g: ev_pick(
                min(0.95, max(0.05, g.p_a + 0.01 * g.features[FORM_I])),
                g.p_market,
            ),
        },
    ]
    assert len(recipes) == 15, len(recipes)
    return recipes


def run_arm(recipes, tune: list[Game], test: list[Game], select: str) -> dict:
    rows = []
    for rec in recipes:
        tune_picks = [rec["pick"](g) for g in tune]
        test_picks = [rec["pick"](g) for g in test]
        row = {
            "id": rec["id"],
            "goal": rec["goal"],
            "tune_2025": _summarize(tune, tune_picks),
            "test_2026": _summarize(test, test_picks),
        }
        rows.append(row)
        print(
            f"  {rec['id']:<32} tune_acc={row['tune_2025']['accuracy']:.1%} "
            f"test_acc={row['test_2026']['accuracy']:.1%} "
            f"test_pnl={row['test_2026']['net_pnl']:+,.0f}",
            flush=True,
        )
    key = (
        (lambda r: r["tune_2025"]["accuracy"])
        if select == "accuracy"
        else (lambda r: r["tune_2025"]["roi_pct"])
    )
    winner = max(rows, key=key)
    return {"runs": rows, "selected_on_2025": winner["id"], "winner": winner}


def main() -> None:
    print("loading odds and seasons (gated features, market for D/E only)...")
    odds = load_odds(season=None)
    train = []
    for season in TRAIN_SEASONS:
        season_games = _load_season(season, odds)
        print(f"  season {season}: {len(season_games)} games with a closing line")
        train.extend(season_games)
    # Select on the later train season so we are not fitting 2026.
    tune = [g for g in train if g.season == 2025]
    test = _load_season(TEST_SEASON, odds)
    print(f"  tune 2025: {len(tune)}  test 2026: {len(test)}")

    print("\n=== MODEL D — 15 accuracy recipes ===")
    d_recs = d_recipes(train)
    d_spec = _fit_market_model(train)
    d_out = run_arm(d_recs, tune, test, select="accuracy")
    winner_d = next(r for r in d_recs if r["id"] == d_out["selected_on_2025"])

    print("\n=== MODEL E — 15 profit recipes ===")
    e_out = run_arm(e_recipes(winner_d["pick"], d_spec), tune, test, select="roi")

    payload = {
        "contract": "de-search-15x15-select-on-2025-report-2026-v1",
        "note": (
            "D is selected by 2024-25/2025 accuracy, E by 2025 ROI. "
            "2025-26 numbers are the holdout. A/B/C cannot see the market."
        ),
        "D": d_out,
        "E": e_out,
        "baselines_2026": {
            "always_home": _summarize(test, [True] * len(test)),
            "vegas_favorite": _summarize(
                test, [pick_accuracy(g.p_market) for g in test]
            ),
        },
    }
    # lambdas are not serializable; drop pick callables from winner copies
    for arm in ("D", "E"):
        payload[arm]["winner"] = {
            k: v for k, v in payload[arm]["winner"].items() if k != "pick"
        }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {LOG_PATH}")
    print(
        f"FOLLOW D: {d_out['selected_on_2025']} "
        f"2026 acc {d_out['winner']['test_2026']['accuracy']:.1%} "
        f"pnl {d_out['winner']['test_2026']['net_pnl']:+,.0f}"
    )
    print(
        f"FOLLOW E: {e_out['selected_on_2025']} "
        f"2026 acc {e_out['winner']['test_2026']['accuracy']:.1%} "
        f"pnl {e_out['winner']['test_2026']['net_pnl']:+,.0f}"
    )


if __name__ == "__main__":
    main()
