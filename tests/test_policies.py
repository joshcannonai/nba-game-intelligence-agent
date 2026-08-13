"""D/E decision rules, reconstructed prices, and the no-Gemma mass-eval contract."""

from __future__ import annotations

import math

from eval.betting import decimal_to_american, load_odds, priced
from eval.policies import (
    expected_value,
    log_odds_blend,
    pick_accuracy,
    pick_d,
    pick_e,
    pick_ev,
    settle_pick,
)
from models.train_d import D_FEATURE_NAMES, TEST_SEASON, TRAIN_SEASONS


def test_selected_e_never_fades_a_70_percent_favorite():
    """A 85% favorite is chalk; E still bets it even if A only has 55%."""
    assert pick_ev(0.55, 0.85) is False
    assert pick_e(0.55, 0.85) is True
    assert pick_e(0.55, 0.15) is False
    # At the 70% floor, E copies the favorite rather than taking the +EV dog.
    assert pick_e(0.55, 0.70) is True
    # Below the 70% line, E still takes the +EV dog.
    assert pick_e(0.55, 0.65) == pick_ev(0.55, 0.65)


def test_e_bets_a_short_home_dog_with_a_rest_edge():
    """Schedule-gated overlay: home underdog, p in [0.42, 0.50), rest_diff > 0."""
    assert pick_e(0.40, 0.45, rest_diff=1.0) is True
    assert pick_e(0.40, 0.45, rest_diff=0.0) == pick_e(0.40, 0.45)
    assert pick_e(0.40, 0.40, rest_diff=1.0) == pick_e(0.40, 0.40)


def test_e_fades_a_heavy_favorite_when_belief_is_mild():
    """The -600 case: A slightly likes the favorite, the price does not."""
    p_a, p_market = 0.55, 0.85
    assert pick_accuracy(p_a) is True
    assert pick_ev(p_a, p_market) is False
    p_d = log_odds_blend(p_a, p_market)
    assert p_d > 0.5
    assert pick_accuracy(p_d) is True
    assert pick_ev(p_a, p_market) != pick_accuracy(p_d)


def test_e_matches_d_when_the_price_is_fair_and_belief_agrees():
    p = 0.62
    assert pick_accuracy(p) is True
    # Belief equals the market, so both moneylines have the same negative EV
    # after vig and the tie-break follows the accuracy pick.
    assert pick_ev(p, p) is True


def test_expected_value_of_a_fair_coin_at_even_money_is_zero():
    assert expected_value(0.5, 2.0, stake=100) == 0.0


def test_settle_pick_loses_the_stake_when_wrong():
    out = settle_pick(pick_home=True, home_won=False, p_market_home=0.7)
    assert out["net_pnl"] == -100.0
    assert out["american_odds"] == decimal_to_american(out["decimal_odds"])


def test_decimal_to_american_round_trips_favorites_and_dogs():
    assert decimal_to_american(1.5) == -200
    assert decimal_to_american(3.0) == 200


def test_priced_odds_embed_the_hold():
    fair = 0.6
    dec = priced(fair)
    implied = 1.0 / dec
    assert implied > fair
    assert math.isclose(implied / fair, 1.0375, rel_tol=1e-9)


def test_d_feature_list_is_a_plus_market_and_logit():
    from models.features import FEATURE_NAMES

    assert D_FEATURE_NAMES[-2:] == ("market_home_prob", "market_logit")
    assert D_FEATURE_NAMES[:-2] == FEATURE_NAMES


def test_pick_d_copies_the_closing_favorite_in_playoffs():
    assert pick_d(0.80, 0.40, playoffs=True) is False
    assert pick_d(0.20, 0.60, playoffs=True) is True
    assert pick_d(0.80, 0.40, playoffs=False) is True


def test_d_uses_the_same_season_split_as_a():
    assert TEST_SEASON == 2026
    assert TEST_SEASON not in TRAIN_SEASONS
    assert TRAIN_SEASONS == (2024, 2025)


def test_e_accuracy_is_the_pick_not_the_belief():
    from eval.mass_eval import summarize

    records = [
        {
            "actual_home_win": 1,
            "A_p": 0.55,
            "A_correct": 1,
            "A_pnl": 10.0,
            "E_p": 0.55,
            "E_correct": 0,
            "E_pnl": -100.0,
            "d_disagrees_a": 0,
            "e_disagrees_d": 1,
            "e_fades_favorite": 1,
        }
    ]
    out = summarize(records, ["A", "E"])
    assert out["A"]["accuracy"] == 1.0
    assert out["E"]["accuracy"] == 0.0
    assert out["E"]["correct"] == 0


def test_load_odds_defaults_to_the_test_season_only():
    odds_2026 = load_odds()
    odds_all = load_odds(season=None)
    assert odds_2026
    assert len(odds_all) > len(odds_2026)
    assert all(row["season"] == "2026" for row in odds_2026.values())
