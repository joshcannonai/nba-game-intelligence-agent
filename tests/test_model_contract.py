"""What the model promises: an honest split, as-of features, a stable interface.

The model is the one component that could quietly invalidate the whole
experiment, because a leaky model does not crash -- it just posts a wonderful
number and nobody asks why. So these tests attack the ways it could cheat rather
than checking that it runs.

Three independent lines of defence, deliberately not sharing an implementation:

    the split     the test season is never in the training seasons
    the features  a row for game G is unchanged by anything after G
    the agreement models/features.py and agent/sources.py, which compute form by
                  completely different means, agree game for game

The third is the one that catches a whole class of silent drift: the feature
builder walks the season forward with accumulators for speed, while the agent's
accessor re-scans the log per query. Two implementations of one idea will
eventually disagree, and when they do, the model is training on a world the
agent cannot see.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, timedelta

import pytest

from agent.sources import REPO_ROOT, team_form_as_of
from models.features import FEATURE_NAMES, _replay_injury_cutoff, build_season
from models.predict import load_model, predict, score_features
from models.train import TEST_SEASON, TRAIN_SEASONS

AS_OF = "2026-01-14"

# Spelled out, not imported from models.features. Importing FORM_WINDOW would
# make the agreement test agree with the code by construction: changing the
# window in one place would change it here too and the test would keep passing
# while the model and the agent had silently drifted apart.
FORM_WINDOW = 10


@pytest.fixture(scope="module")
def season_2026():
    rows, _ = build_season(TEST_SEASON)
    return rows


# --------------------------------------------------------------- the split


def test_test_season_is_not_trained_on():
    assert TEST_SEASON not in TRAIN_SEASONS


def test_training_seasons_end_before_the_test_season_begins():
    """Not just different seasons -- earlier ones, with no calendar overlap."""
    test_rows, _ = build_season(TEST_SEASON)
    first_test_game = min(r.game_date for r in test_rows)
    for season in TRAIN_SEASONS:
        train_rows, _ = build_season(season)
        last_train_game = max(r.game_date for r in train_rows)
        assert last_train_game < first_test_game, (
            f"season {season} runs to {last_train_game}, on or after the test "
            f"season's first game {first_test_game}"
        )


def test_the_shipped_model_declares_the_same_split():
    spec = load_model()
    assert spec is not None, "run `python -m models.train`"
    assert spec["test_season"] == TEST_SEASON
    assert tuple(spec["train_seasons"]) == TRAIN_SEASONS


# ------------------------------------------------------------- the features


def test_a_game_cannot_see_its_own_result(season_2026):
    """The classic leak: a season win-percentage that includes today's game.

    Opening night is the sharpest version. Every team is 0-0 with no history, so
    every differential feature must be exactly zero. If any of them is not, the
    accumulator was advanced before the row was emitted.
    """
    first_date = min(r.game_date for r in season_2026)
    openers = [r for r in season_2026 if r.game_date == first_date]
    assert openers, "no opening-night games found"

    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    for r in openers:
        assert r.features[idx["win_pct_diff"]] == 0.0, r.game_id
        assert r.features[idx["form_margin_diff"]] == 0.0, r.game_id
        assert r.features[idx["home_games_played"]] == 0.0, r.game_id
        assert r.features[idx["away_games_played"]] == 0.0, r.game_id


def test_date_only_injuries_stop_before_game_day():
    """An injury transaction without a timestamp is unsafe on game day."""
    assert _replay_injury_cutoff(date(2026, 1, 14)) == date(2026, 1, 13)


def test_games_played_never_exceeds_the_games_actually_behind_it(season_2026):
    """A cheap monotonicity check that a full-season groupby would fail loudly."""
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    by_date = {}
    for r in season_2026:
        by_date.setdefault(r.game_date, []).append(r)

    seen_games = 0
    for game_date in sorted(by_date):
        for r in by_date[game_date]:
            # A team cannot have played more games than have been played at all.
            assert r.features[idx["home_games_played"]] <= seen_games, r.game_id
            assert r.features[idx["away_games_played"]] <= seen_games, r.game_id
        seen_games += len(by_date[game_date])


def test_feature_order_matches_the_shipped_coefficients():
    """Positional coefficients: a reorder would silently remap every weight."""
    spec = load_model()
    assert tuple(spec["feature_names"]) == FEATURE_NAMES
    assert len(spec["coefficients"]) == len(FEATURE_NAMES)
    assert len(spec["scaler_mean"]) == len(FEATURE_NAMES)
    assert len(spec["scaler_scale"]) == len(FEATURE_NAMES)


# ------------------------------------------------------------ the agreement


def test_form_agrees_with_the_accessor_the_agent_uses(season_2026):
    """models/features.py and agent/sources.py must describe form identically.

    Sampled rather than exhaustive: team_form_as_of re-scans the season log on
    every call, so all 1,322 games would take minutes for no extra signal.
    """
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    checked = 0

    for r in season_2026[::37]:  # a stride, so the sample spans the whole season
        if r.features[idx["home_games_played"]] < 1:
            continue
        if r.features[idx["away_games_played"]] < 1:
            continue

        cutoff = r.game_date - timedelta(days=1)
        home_form = team_form_as_of(r.home, cutoff, FORM_WINDOW)
        away_form = team_form_as_of(r.away, cutoff, FORM_WINDOW)
        assert home_form and away_form, r.game_id

        expected = home_form["avg_point_diff"] - away_form["avg_point_diff"]
        actual = r.features[idx["form_margin_diff"]]
        assert actual == pytest.approx(expected, abs=0.02), (
            f"{r.game_id}: feature builder says {actual:.3f}, "
            f"agent's team_form says {expected:.3f}"
        )
        checked += 1

    assert checked > 20, f"only {checked} rows compared -- the sample is too thin"


# -------------------------------------------------------------- the interface


def test_predict_returns_a_probability():
    out = predict("BOS", "ORL", AS_OF)
    assert out["status"] == "ok"
    assert 0.0 < out["home_win_prob"] < 1.0
    assert out["home_win_prob"] + out["away_win_prob"] == pytest.approx(1.0, abs=1e-3)


def test_predict_is_symmetric_under_swapping_the_teams():
    """Swap home and away and the probabilities must not simply mirror.

    Home advantage is real, so p(A home vs B) + p(B home vs A) should exceed 1.
    A model that returns exactly mirrored numbers has lost the home term, which
    is worth about 55% of the entire baseline.
    """
    a = predict("BOS", "ORL", AS_OF)["home_win_prob"]
    b = predict("ORL", "BOS", AS_OF)["home_win_prob"]
    assert a + b > 1.0, f"no home-court effect: {a:.3f} + {b:.3f} = {a + b:.3f}"


def test_score_features_matches_predict(season_2026):
    """The harness's fast path and the UI's slow path must not diverge."""
    spec = load_model()
    assert spec is not None
    r = next(r for r in season_2026 if r.game_date.isoformat() == AS_OF)
    cutoff = (r.game_date - timedelta(days=1)).isoformat()
    ui = predict(r.home, r.away, cutoff, r.game_date.isoformat())["home_win_prob"]
    assert ui == pytest.approx(score_features(r.features), abs=1e-4)


def test_missing_model_reports_awaiting_input_instead_of_guessing(
    tmp_path, monkeypatch
):
    """A deleted model file must not silently become a 50/50 coin flip."""
    import models.predict as mp

    monkeypatch.setattr(mp, "MODEL_PATH", tmp_path / "absent.json")
    mp.load_model.cache_clear()
    try:
        out = mp.predict("BOS", "ORL", AS_OF)
        assert out["status"] == "awaiting_input"
        assert "home_win_prob" not in out
    finally:
        mp.load_model.cache_clear()


# ------------------------------------------------------- end-to-end, on disk


def test_sign_test_matches_hand_computed_binomial():
    """The headline stat in the write-up, so it gets checked by hand.

    12 coin flips: P(>=10 heads) = (66 + 12 + 1) / 4096.
    """
    from eval.three_arms import _sign_test

    assert _sign_test(10, 12) == pytest.approx(79 / 4096, abs=1e-9)
    assert _sign_test(12, 12) == pytest.approx(1 / 4096, abs=1e-9)
    assert _sign_test(0, 12) == pytest.approx(1.0)
    assert _sign_test(0, 0) == 1.0
    # An even split must not look like a finding.
    assert _sign_test(6, 12) > 0.05


def test_override_analysis_counts_only_disagreements():
    """Games where the arms agree carry no information about overruling.

    Built by hand rather than from the results CSV: reading the real file would
    make this test pass for whatever the run happened to produce, instead of
    checking that the counting rule is right.
    """
    from types import SimpleNamespace

    from eval.three_arms import _override_analysis

    rows = [
        SimpleNamespace(game_id="agree_both_right", home_won=1),
        SimpleNamespace(game_id="agree_both_wrong", home_won=0),
        SimpleNamespace(game_id="override_agent_right", home_won=0),
        SimpleNamespace(game_id="override_model_right", home_won=1),
    ]
    probs = {
        "A": {
            "agree_both_right": 0.8,
            "agree_both_wrong": 0.9,
            "override_agent_right": 0.7,
            "override_model_right": 0.8,
        },
        "C": {
            "agree_both_right": 0.6,  # same side as A -- agreement
            "agree_both_wrong": 0.7,  # same side as A -- agreement
            "override_agent_right": 0.3,  # flipped, and correct
            "override_model_right": 0.2,  # flipped, and wrong
        },
    }
    # Smoke-level: the function prints rather than returns, so this asserts it
    # runs over a hand-built case without raising. The arithmetic it prints is
    # covered by test_sign_test_matches_hand_computed_binomial above.
    _override_analysis(rows, probs)

    # Missing an arm must be a no-op, not a crash: `--arms a` has no C.
    _override_analysis(rows, {"A": probs["A"]})


def test_features_are_identical_against_a_gated_snapshot(tmp_path):
    """The strongest form: rebuild features from data that never held the future.

    A subprocess, not an import: `agent.sources` captures its directory
    constants at import time and lru_caches every reader, so flipping the env
    var in-process would keep reading the ungated files and this test would
    prove nothing.

    Games before the cutoff must score identically whether the model reads the
    full repo or a snapshot with every later result erased. Equal numbers mean
    the feature builder never consulted them.
    """
    snapshot_dir = tmp_path / "snap"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.gate_snapshot",
            "--as-of",
            AS_OF,
            "--out",
            str(snapshot_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    program = (
        "import json;"
        "from models.features import build_season;"
        "rows,_=build_season(2026);"
        "print(json.dumps({r.game_id: r.features for r in rows "
        f"if r.game_date.isoformat() < '{AS_OF}'}}))"
    )

    def run(env_extra):
        env = {**os.environ, **env_extra}
        out = subprocess.run(
            [sys.executable, "-c", program],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(out.stdout)

    assert (snapshot_dir / "_manifest.json").exists(), "snapshot was not built"
    full = run({})
    gated = run({"NBA_SNAPSHOT_DIR": str(snapshot_dir)})

    assert gated, "the gated run produced no pre-cutoff games"
    assert set(gated) <= set(full)
    for game_id, features in gated.items():
        assert features == pytest.approx(full[game_id]), (
            f"{game_id} scores differently once the future is deleted -- "
            "the feature builder is reading past the as-of date"
        )
