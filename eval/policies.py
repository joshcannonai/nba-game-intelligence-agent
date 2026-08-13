"""Models D and E: same information, different objectives.

A, B, and C are the Gemma / logistic baseline and must not see the market.
D and E are allowed to see the closing line because that is the experiment:
does market information help you pick winners (D) or make money (E)?

The committed full-season D/E numbers are still these Python policies.
Live Gemma D/E search is `python -m eval.gemma_de`: same Gemma 4 as B/C, plus
`retrieve_betting_line`, 30 sequential prompt tweaks on a date-gated sample.
The LLM stays in D/E only if it beats this CPU baseline on that sample.

    D  accuracy   logistic on A-features + close + log-odds of the close;
                  playoff games copy the closing favorite
    E  profit     never fade a 70%+ favorite, else max EV from Model A;
                  overlay: bet a home underdog on a short line (market p
                  in [0.42, 0.50)) when the home team has more rest

E uses A, not D, as its belief on purpose. If E priced bets with D's probability,
D would already contain the market and both sides' EV would collapse to the vig.
The -600 example only works when the belief is independent of the price.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from eval.betting import HOLD, STAKE, decimal_to_american, priced, settle
from models.predict import score_features

ROOT = Path(__file__).resolve().parents[1]
D_MODEL_PATH = ROOT / "models" / "win_probability_d.json"
EPS = 1e-15


def _clip(p: float) -> float:
    return min(1.0 - EPS, max(EPS, float(p)))


def log_odds_blend(p_model: float, p_market: float, model_weight: float = 0.5) -> float:
    """Fixed a-priori blend. Weights are not fit on the 2025-26 test season."""
    a = math.log(_clip(p_model) / (1.0 - _clip(p_model)))
    b = math.log(_clip(p_market) / (1.0 - _clip(p_market)))
    z = model_weight * a + (1.0 - model_weight) * b
    return 1.0 / (1.0 + math.exp(-z))


def load_d_model() -> dict | None:
    if not D_MODEL_PATH.exists():
        return None
    spec = json.loads(D_MODEL_PATH.read_text(encoding="utf-8"))
    expected = list(spec.get("feature_names") or [])
    if "market_home_prob" not in expected:
        raise ValueError("Model D JSON is missing market_home_prob")
    return spec


def score_d_features(features_with_market, spec: dict) -> float:
    z = spec["intercept"]
    for x, mean, scale, coef in zip(
        features_with_market,
        spec["scaler_mean"],
        spec["scaler_scale"],
        spec["coefficients"],
    ):
        z += coef * ((x - mean) / (scale or 1.0))
    return 1.0 / (1.0 + math.exp(-z))


def d_home_win_prob(features, p_market: float | None, spec: dict | None) -> float:
    """Market-aware P(home wins). Falls back to a 50/50 blend if D is untrained."""
    p_a = score_features(features)
    if p_market is None:
        return p_a
    if spec is None:
        return log_odds_blend(p_a, p_market)
    extras = [p_market]
    names = spec.get("feature_names") or []
    if "market_logit" in names:
        extras.append(math.log(_clip(p_market) / (1.0 - _clip(p_market))))
    return score_d_features(tuple(features) + tuple(extras), spec)


def pick_d(p_home: float, p_market: float, playoffs: bool = False) -> bool:
    """True = pick home. Playoff games copy the closing favorite."""
    if playoffs:
        return p_market >= 0.5
    return p_home >= 0.5


def pick_accuracy(p_home: float) -> bool:
    """True = pick home. Accuracy maximizer; 0.5 counts as home."""
    return p_home >= 0.5


def expected_value(p_win: float, decimal_odds: float, stake: float = STAKE) -> float:
    """Flat-stake EV of betting `stake` at `decimal_odds` with belief p_win."""
    return p_win * stake * (decimal_odds - 1.0) + (1.0 - p_win) * (-stake)


def pick_ev(p_home_belief: float, p_market_home: float) -> bool:
    """True = bet home moneyline. Profit maximizer; ties break to home."""
    dec_home = priced(p_market_home)
    dec_away = priced(1.0 - p_market_home)
    ev_home = expected_value(p_home_belief, dec_home)
    ev_away = expected_value(1.0 - p_home_belief, dec_away)
    if ev_home == ev_away:
        return p_home_belief >= 0.5
    return ev_home > ev_away


def pick_e(
    p_home_belief: float,
    p_market_home: float,
    never_fade: float = 0.70,
    rest_diff: float | None = None,
) -> bool:
    """Selected E recipe: home-dog + rest overlay, else never-fade 70%, else max EV.

    The overlay (home underdog, market p in [0.42, 0.50), rest_diff > 0) was
    +0.33% ROI on 2024-25 and −0.66% on 2025-26, better 2026 money than
    never-fade 70% alone (−1.15%). Rest is schedule-gated, not a result leak.
    """
    if rest_diff is not None and 0.42 <= p_market_home < 0.5 and rest_diff > 0:
        return True
    if max(p_market_home, 1.0 - p_market_home) >= never_fade:
        return pick_accuracy(p_market_home)
    return pick_ev(p_home_belief, p_market_home)


def settle_pick(pick_home: bool, home_won: bool, p_market_home: float) -> dict:
    """Settle one $100 moneyline at the reconstructed closing price."""
    dec = priced(p_market_home if pick_home else 1.0 - p_market_home)
    pnl = settle(pick_home, home_won, p_market_home)
    return {
        "stake": STAKE,
        "decimal_odds": dec,
        "american_odds": decimal_to_american(dec),
        "net_pnl": pnl,
        "hold": HOLD,
    }
