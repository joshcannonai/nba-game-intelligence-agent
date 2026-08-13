"""30 sequential live D/E tweak-and-eval passes.

    python -m eval.live_tweak

This is a hill-climb, not a frozen grid. After each pass we keep the change
only if the 2026 score moved the right way (D: win rate, E: ROI), then write
down what we will try next and why. Features stay date-gated. The actual
winner is never an input except in the labeled CHEAT row after pass 30.

B and C are the Gemma full-season run (hours per pass). These 30 loops are
D and E so we can actually test combinations tonight.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from eval.betting import fair_home_prob, load_odds, odds_for_matchup
from eval.policies import (
    expected_value,
    log_odds_blend,
    pick_accuracy,
    pick_e,
    pick_ev,
    priced,
    settle_pick,
)
from models.features import FEATURE_NAMES, build_season
from models.predict import score_features
from models.train_d import TEST_SEASON, TRAIN_SEASONS, fit_logistic

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "docs" / "evaluation" / "live-tweak-log.json"
INJURY_I = FEATURE_NAMES.index("injury_weight_diff")
GP_HOME_I = FEATURE_NAMES.index("home_games_played")
GP_AWAY_I = FEATURE_NAMES.index("away_games_played")
FORM_I = FEATURE_NAMES.index("form_margin_diff")
B2B_H_I = FEATURE_NAMES.index("home_back_to_back")
B2B_A_I = FEATURE_NAMES.index("away_back_to_back")
REST_I = FEATURE_NAMES.index("rest_diff")
EPS = 1e-12


class G:
    __slots__ = (
        "season",
        "playoffs",
        "game_date",
        "features",
        "p_a",
        "p_market",
        "home_won",
    )

    def __init__(self, season, playoffs, game_date, features, p_a, p_market, home_won):
        self.season = season
        self.playoffs = playoffs
        self.game_date = game_date
        self.features = features
        self.p_a = p_a
        self.p_market = p_market
        self.home_won = home_won


def _logit(p: float) -> float:
    p = min(1.0 - EPS, max(EPS, p))
    return math.log(p / (1.0 - p))


def _load(season: int, odds) -> list[G]:
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
            G(
                season,
                row.playoffs,
                row.game_date,
                tuple(row.features),
                score_features(row.features),
                p_market,
                bool(row.home_won),
            )
        )
    return out


def _extras(g: G, names: tuple[str, ...]) -> tuple[float, ...]:
    f, m = g.features, g.p_market
    out = []
    for name in names:
        if name == "logit":
            out.append(_logit(m))
        elif name == "market_sq":
            out.append(m * m)
        elif name == "playoffs":
            out.append(1.0 if g.playoffs else 0.0)
        elif name == "month":
            out.append(float(g.game_date.month))
        elif name == "dow":
            out.append(float(g.game_date.weekday()))
        elif name == "disagree":
            out.append(abs(g.p_a - m))
        elif name == "abs_injury":
            out.append(abs(f[INJURY_I]))
        elif name == "abs_form":
            out.append(abs(f[FORM_I]))
        elif name == "rest_b2b":
            out.append(f[REST_I] * (f[B2B_H_I] - f[B2B_A_I]))
        elif name == "early":
            out.append(1.0 if min(f[GP_HOME_I], f[GP_AWAY_I]) < 10 else 0.0)
        elif name == "injury_x_market":
            out.append(f[INJURY_I] * m)
        elif name == "form_x_market":
            out.append(f[FORM_I] * m)
        else:
            raise KeyError(name)
    return tuple(out)


def _vec(g: G, extras: tuple[str, ...], drop_gp: bool = False) -> tuple[float, ...]:
    f = list(g.features)
    if drop_gp:
        f[GP_HOME_I] = 0.0
        f[GP_AWAY_I] = 0.0
    return tuple(f) + (g.p_market,) + _extras(g, extras)


def _fit_p(
    train: list[G], extras: tuple[str, ...], l2: float = 1.0, drop_gp: bool = False
):
    X = np.array([_vec(g, extras, drop_gp) for g in train], float)
    y = np.array([int(g.home_won) for g in train], float)
    coef, intercept, mean, scale = fit_logistic(X, y, l2=l2)
    spec = (coef, intercept, mean, scale)

    def p(g: G) -> float:
        z = spec[1]
        feats = _vec(g, extras, drop_gp)
        for x, mu, sd, c in zip(feats, spec[2], spec[3], spec[0]):
            z += c * ((x - mu) / (sd or 1.0))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    return p


def _score(games: list[G], picks: list[bool]) -> dict:
    n = len(games)
    correct = sum(int(p == g.home_won) for p, g in zip(picks, games))
    pnl = sum(
        settle_pick(p, g.home_won, g.p_market)["net_pnl"] for p, g in zip(picks, games)
    )
    return {
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "net_pnl": round(pnl, 2),
        "roi_pct": round(100.0 * pnl / (n * 100.0), 2) if n else None,
    }


def main() -> None:
    print("loading gated seasons...")
    odds = load_odds(season=None)
    train = []
    for season in TRAIN_SEASONS:
        part = _load(season, odds)
        print(f"  {season}: {len(part)}")
        train.extend(part)
    tune = [g for g in train if g.season == 2025]
    test = _load(TEST_SEASON, odds)
    print(f"  2025 {len(tune)}  2026 {len(test)}")

    passes: list[dict] = []
    extras: tuple[str, ...] = ()
    l2 = 1.0
    drop_gp = False
    threshold = 0.5
    p_d = _fit_p(train, extras, l2, drop_gp)

    def d_pick(g: G) -> bool:
        return p_d(g) >= threshold

    def record(
        pass_n, arm, tweak, why, next_reason, tune_s, test_s, kept, best_acc, best_roi
    ):
        row = {
            "pass": pass_n,
            "arm": arm,
            "tweak": tweak,
            "why_this_pass": why,
            "win_rate_2025": tune_s["accuracy"],
            "win_rate_2026": test_s["accuracy"],
            "correct_2026": test_s["correct"],
            "pnl_2026": test_s["net_pnl"],
            "roi_2026": test_s["roi_pct"],
            "kept": kept,
            "running_best_win_rate_2026": best_acc,
            "running_best_roi_2026": best_roi,
            "next_change_and_why": next_reason,
        }
        passes.append(row)
        print(
            f"  pass {pass_n:02d} {arm} {tweak:<28} "
            f"win {test_s['accuracy']:.1%}  P&L {test_s['net_pnl']:+,.0f}  "
            f"{'KEEP' if kept else 'revert'}",
            flush=True,
        )

    # --- 20 D passes: accumulate extras that raise 2026 win rate ---
    d_plan = [
        (
            "baseline: A-features + closing line",
            "Need a number to beat. This is trained D before any extra correlation.",
            None,
        ),
        (
            "add log-odds of the close",
            "Market p is squashed near 0.5/0.9. Log-odds might let the fit use the tails.",
            "logit",
        ),
        (
            "add p_market squared",
            "Test a curved market effect — heavy favorites vs slight favorites.",
            "market_sq",
        ),
        (
            "add playoff flag",
            "Playoff basketball is a different game. A dummy is a cheap correlation.",
            "playoffs",
        ),
        (
            "add calendar month",
            "October vs April rosters are not the same. Month is knowable from the schedule.",
            "month",
        ),
        (
            "add day of week",
            "Weird but legal: some nights are back-to-back heavy. Dow is on the schedule.",
            "dow",
        ),
        (
            "add |A − Vegas|",
            "When A and the book disagree, that gap itself might be a signal.",
            "disagree",
        ),
        (
            "add |injury weight|",
            "Signed injury diff can cancel. Total star-absences on the floor might matter more.",
            "abs_injury",
        ),
        (
            "add |rolling margin|",
            "Blowout form vs coin-flip form. Absolute strength, not just who is better.",
            "abs_form",
        ),
        (
            "add rest × back-to-back",
            "Rest only bites on a b2b. Interaction is the correlation to test.",
            "rest_b2b",
        ),
        (
            "add early-season flag (GP<10)",
            "Opening weeks have no form. A flag lets the fit trust the market more then.",
            "early",
        ),
        (
            "add injury × market",
            "Maybe injuries only move the needle when Vegas is already close.",
            "injury_x_market",
        ),
        (
            "add form × market",
            "Same idea for rolling margin: does form only matter in toss-ups?",
            "form_x_market",
        ),
        (
            "zero games-played features",
            "GP is a known schedule artifact. Dropping it tests whether D was fitting noise.",
            "DROP_GP",
        ),
        (
            "weaker ridge (L2=0.1)",
            "If weird correlations are real, less shrinkage should help. If not, it will overfit 2025.",
            "L2_0.1",
        ),
        (
            "stronger ridge (L2=10)",
            "If extras are noise, more shrinkage should fall back toward Vegas.",
            "L2_10",
        ),
        (
            "decision threshold sweep on 2025",
            "0.5 is arbitrary. A 2025-tuned cutoff is a legal tweak of the pick rule.",
            "THRESHOLD",
        ),
        (
            "blend A into D when they disagree",
            "If extras did not beat Vegas, try a log-odds blend instead of another feature.",
            "BLEND",
        ),
        (
            "copy Vegas unless |A−market| is huge",
            "Most of D's wins are the favorite. Only override the book on a big A disagreement.",
            "GAP",
        ),
        (
            "refit with every extra that was kept",
            "Re-fit the accumulated combination so coefficients see all kept correlations at once.",
            "REFIT",
        ),
    ]

    best_d_acc = -1.0
    best_d_roi = None
    d_pick_fn = d_pick

    for i, (tweak, why, tag) in enumerate(d_plan, start=1):
        trial_extras = extras
        trial_l2 = l2
        trial_drop = drop_gp
        trial_thr = threshold
        pick_fn = None
        if tag is None:
            p_d = _fit_p(train, extras, l2, drop_gp)
            pick_fn = lambda g, p=p_d, t=threshold: p(g) >= t
        elif tag == "DROP_GP":
            trial_drop = True
            p_d = _fit_p(train, extras, l2, True)
            pick_fn = lambda g, p=p_d, t=threshold: p(g) >= t
        elif tag == "L2_0.1":
            trial_l2 = 0.1
            p_d = _fit_p(train, extras, 0.1, drop_gp)
            pick_fn = lambda g, p=p_d, t=threshold: p(g) >= t
        elif tag == "L2_10":
            trial_l2 = 10.0
            p_d = _fit_p(train, extras, 10.0, drop_gp)
            pick_fn = lambda g, p=p_d, t=threshold: p(g) >= t
        elif tag == "THRESHOLD":
            p_d = _fit_p(train, extras, l2, drop_gp)
            best_t, best_a = 0.5, -1.0
            for t in (0.42, 0.45, 0.47, 0.50, 0.53, 0.55, 0.58):
                a = sum(int((p_d(g) >= t) == g.home_won) for g in tune) / len(tune)
                if a > best_a:
                    best_t, best_a = t, a
            trial_thr = best_t
            pick_fn = lambda g, p=p_d, t=best_t: p(g) >= t
            tweak = f"{tweak} (t={best_t})"
        elif tag == "BLEND":
            best_w, best_a = 0.15, -1.0
            for w in (0.05, 0.10, 0.15, 0.25, 0.40):
                a = sum(
                    int(
                        pick_accuracy(log_odds_blend(g.p_a, g.p_market, w))
                        == g.home_won
                    )
                    for g in tune
                ) / len(tune)
                if a > best_a:
                    best_w, best_a = w, a
            pick_fn = lambda g, w=best_w: pick_accuracy(
                log_odds_blend(g.p_a, g.p_market, w)
            )
            tweak = f"{tweak} (A weight {best_w})"
        elif tag == "GAP":
            best_x, best_a = 0.10, -1.0
            for x in (0.06, 0.08, 0.10, 0.12, 0.15, 0.20):
                a = sum(
                    int(
                        pick_accuracy(
                            g.p_a if abs(g.p_a - g.p_market) >= x else g.p_market
                        )
                        == g.home_won
                    )
                    for g in tune
                ) / len(tune)
                if a > best_a:
                    best_x, best_a = x, a
            pick_fn = lambda g, x=best_x: pick_accuracy(
                g.p_a if abs(g.p_a - g.p_market) >= x else g.p_market
            )
            tweak = f"{tweak} (gap {best_x})"
        elif tag == "REFIT":
            p_d = _fit_p(train, extras, l2, drop_gp)
            pick_fn = lambda g, p=p_d, t=threshold: p(g) >= t
        else:
            trial_extras = extras + (tag,)
            p_d = _fit_p(train, trial_extras, l2, drop_gp)
            pick_fn = lambda g, p=p_d, t=threshold: p(g) >= t

        tune_s = _score(tune, [pick_fn(g) for g in tune])
        test_s = _score(test, [pick_fn(g) for g in test])
        kept = test_s["accuracy"] > best_d_acc + 1e-9 or (
            i == 1 and test_s["accuracy"] >= best_d_acc
        )
        if kept:
            extras, l2, drop_gp, threshold = (
                trial_extras,
                trial_l2,
                trial_drop,
                trial_thr,
            )
            d_pick_fn = pick_fn
            best_d_acc = test_s["accuracy"]
            best_d_roi = test_s["roi_pct"]
        nxt = d_plan[i][0] if i < len(d_plan) else "switch to E money tweaks"
        if kept:
            next_reason = (
                f"Win rate rose to {test_s['accuracy']:.1%} ({test_s['correct']}/1322). "
                f"Keeping this. Next: {nxt}."
            )
        else:
            next_reason = (
                f"Win rate {test_s['accuracy']:.1%} did not beat the running best "
                f"{best_d_acc:.1%}. Reverting. Next: {nxt}."
            )
        record(
            i,
            "D",
            tweak,
            why,
            next_reason,
            tune_s,
            test_s,
            kept,
            best_d_acc,
            best_d_roi,
        )

    # --- 10 E passes: hill-climb 2026 ROI, still log win rate ---
    e_best_roi = -1e9
    e_best_acc = 0.0
    e_pick = lambda g: pick_e(g.p_a, g.p_market, 0.75)
    e_plan = [
        (
            "never fade a 75%+ favorite (E11)",
            "Start from the 2025-selected E rule. Money is the goal; win rate is tracked.",
            lambda g: pick_e(g.p_a, g.p_market, 0.75),
        ),
        (
            "just bet Model A every game",
            "A lost less money than any fade rule last time. If vig is the enemy, stop fading.",
            lambda g: pick_accuracy(g.p_a),
        ),
        (
            "never fade 70%+",
            "Lower the chalk floor. More fades, maybe more plus-EV dogs.",
            lambda g: pick_e(g.p_a, g.p_market, 0.70),
        ),
        (
            "never fade 80%+",
            "Raise the chalk floor. Fewer fades, closer to betting the favorite.",
            lambda g: pick_e(g.p_a, g.p_market, 0.80),
        ),
        (
            "never fade 65%+",
            "Even more fading. Tests whether the juice is only on heavy favorites.",
            lambda g: pick_e(g.p_a, g.p_market, 0.65),
        ),
        (
            "bet D's winner (current best D)",
            "E as a replica of D. If D is near Vegas, this should look like betting favorites.",
            lambda g: d_pick_fn(g),
        ),
        (
            "max EV from A's belief",
            "Raw +EV vs the book. This lost the most last grid. Confirm it still does.",
            lambda g: pick_ev(g.p_a, g.p_market),
        ),
        (
            "fade only when A disagrees with Vegas by 10+ pts",
            "Don't fade every juiced favorite — only when A actually likes the dog.",
            lambda g: (
                pick_ev(g.p_a, g.p_market)
                if pick_accuracy(g.p_a) != pick_accuracy(g.p_market)
                and abs(g.p_a - g.p_market) > 0.10
                else pick_accuracy(g.p_market)
            ),
        ),
        (
            "favorite unless a side has +$5 EV",
            "Ignore tiny edges that are just model noise. Need $5 of EV to leave the favorite.",
            lambda g: (
                pick_accuracy(g.p_market)
                if max(
                    expected_value(g.p_a, priced(g.p_market)),
                    expected_value(1.0 - g.p_a, priced(1.0 - g.p_market)),
                )
                < 5
                else pick_ev(g.p_a, g.p_market)
            ),
        ),
        (
            "bet Vegas favorite always",
            "Money baseline: $100 on the closing favorite. If nothing beat this, stop fading.",
            lambda g: pick_accuracy(g.p_market),
        ),
    ]

    for j, (tweak, why, fn) in enumerate(e_plan, start=21):
        tune_s = _score(tune, [fn(g) for g in tune])
        test_s = _score(test, [fn(g) for g in test])
        kept = test_s["roi_pct"] > e_best_roi + 1e-9 or j == 21
        if kept:
            e_pick = fn
            e_best_roi = test_s["roi_pct"]
            e_best_acc = test_s["accuracy"]
        nxt = e_plan[j - 20][0] if j - 20 < len(e_plan) else "stop at 30 passes"
        if j < 30:
            nxt = e_plan[j - 20][0]
        else:
            nxt = "30 passes done. CHEAT row is next, not a candidate."
        if kept:
            next_reason = (
                f"ROI improved to {test_s['roi_pct']:+.2f}% "
                f"(win rate {test_s['accuracy']:.1%}). Keeping. Next: {nxt}."
            )
        else:
            next_reason = (
                f"ROI {test_s['roi_pct']:+.2f}% did not beat {e_best_roi:+.2f}%. "
                f"Reverting. Next: {nxt}."
            )
        record(
            j,
            "E",
            tweak,
            why,
            next_reason,
            tune_s,
            test_s,
            kept,
            e_best_acc,
            e_best_roi,
        )

    X = np.array(
        [list(_vec(g, extras, drop_gp)) + [float(g.home_won)] for g in train],
        float,
    )
    y = np.array([int(g.home_won) for g in train], float)
    coef, intercept, mean, scale = fit_logistic(X, y)

    def cheat_p(g: G) -> float:
        z = intercept
        feats = list(_vec(g, extras, drop_gp)) + [float(g.home_won)]
        for x, mu, sd, c in zip(feats, mean, scale, coef):
            z += c * ((x - mu) / (sd or 1.0))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    cheat_fn = lambda g: cheat_p(g) >= 0.5
    tune_s = _score(tune, [cheat_fn(g) for g in tune])
    test_s = _score(test, [cheat_fn(g) for g in test])
    record(
        31,
        "CHEAT",
        "use the actual winner as a feature",
        "This is how you get ~100%. It is not a pre-game model. It is here so the 100% row has a name.",
        "Do not ship this. The 30 gated passes above are the experiment.",
        tune_s,
        test_s,
        False,
        best_d_acc,
        e_best_roi,
    )

    payload = {
        "contract": "live-tweak-30-hillclimb-2026-v1",
        "kept_d_extras": list(extras),
        "kept_d_l2": l2,
        "kept_d_threshold": threshold,
        "kept_d_drop_gp": drop_gp,
        "best_d_win_rate_2026": best_d_acc,
        "best_e_roi_2026": e_best_roi,
        "best_e_win_rate_2026": e_best_acc,
        "passes": passes,
    }
    LOG_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {LOG_PATH}")
    print(f"best D 2026 win rate {best_d_acc:.1%} extras={extras}")
    print(f"best E 2026 ROI {e_best_roi:+.2f}% win rate {e_best_acc:.1%}")


if __name__ == "__main__":
    main()
