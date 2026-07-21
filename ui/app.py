"""Streamlit front end for the NBA Game Intelligence Agent.

Runs the deterministic (no-LLM) path by default so it is fast, free, and safe
to demo. Everything shown here comes from the same tools the agent calls --
there is no display-only data and nothing is mocked up for the screenshot.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.run import dry_run, status_board  # noqa: E402
from agent.sources import get_source  # noqa: E402

st.set_page_config(
    page_title="NBA Game Intelligence Agent", page_icon="🏀", layout="wide"
)

st.markdown(
    """
<style>
  .block-container { padding-top: 2.2rem; max-width: 1180px; }
  h1 { font-size: 1.9rem !important; letter-spacing: -0.02em; font-weight: 700; }
  .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; border-bottom: 1px solid #232C38; }
  .stTabs [data-baseweb="tab"] { font-weight: 600; padding: 0.5rem 0.9rem; }
  [data-testid="stMetric"] {
      background: #151C25; border: 1px solid #232C38; border-radius: 10px;
      padding: 0.9rem 1.1rem;
  }
  [data-testid="stMetricValue"] {
      font-variant-numeric: tabular-nums; font-size: 2.1rem !important;
  }
  [data-testid="stMetricLabel"] { opacity: 0.72; font-size: 0.82rem !important; }
  .card {
      background: #151C25; border: 1px solid #232C38; border-left: 3px solid #F97316;
      border-radius: 8px; padding: 0.85rem 1rem; margin-bottom: 0.6rem;
  }
  .card .k { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em;
             opacity: 0.6; margin-bottom: 0.3rem; }
  .card .v { font-size: 0.95rem; line-height: 1.45; }
  .gap { border-left-color: #64748B; }
  .lede { color: #94A3B8; font-size: 0.95rem; line-height: 1.6; margin-bottom: 1.1rem; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_games(season: int) -> list[dict]:
    path = REPO_ROOT / "data" / "samples" / f"game_logs_{season}.csv"
    if not path.exists():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def parse_iso(s: str) -> date:
    y, m, d = (int(p) for p in s.split("-"))
    return date(y, m, d)


st.title("🏀 NBA Game Intelligence Agent")
st.markdown(
    '<div class="lede">Pick a game, then pick a date you are asking <b>from</b>. '
    "The system answers using only what was knowable that morning — no future "
    "information reaches the prediction. That constraint is the point: it is what "
    "lets us test on a season that has already happened without the model simply "
    "remembering the result.<br><span style='opacity:.65'>CECS 499 · Josh Cannon · "
    "agent lane</span></div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Matchup")
    season = st.selectbox("Season sample", [2025, 2024], index=0)
    games = load_games(season)

    if not games:
        st.error(f"data/samples/game_logs_{season}.csv not found.")
        st.stop()

    teams = sorted({g["home"] for g in games})
    home = st.selectbox(
        "Home team", teams, index=teams.index("BOS") if "BOS" in teams else 0
    )
    options = [g for g in games if g["home"] == home]
    labels = [f"{g['away']} @ {g['home']}  ·  {g['game_date']}" for g in options]
    # Default to a mid-season game: the probe dates on the gating tab then land
    # in-season, where the injury list actually moves. The opener does not.
    default_i = next(
        (i for i, g in enumerate(options) if g["game_date"][5:7] in ("12", "01")), 0
    )
    pick = st.selectbox(
        "Game", range(len(labels)), index=default_i, format_func=lambda i: labels[i]
    )
    game = options[pick]

    matchup_id = f"{game['away']}-{game['home']}-{game['game_date']}"
    tip = parse_iso(game["game_date"])
    as_of = st.date_input(
        "As-of date (what we knew)",
        value=tip - timedelta(days=1),
        max_value=tip - timedelta(days=1),
    )
    st.caption(f"`{matchup_id}`")

    source_kind = st.radio("Data source", ["real", "mock"], horizontal=True)

source = get_source(source_kind)
as_of_str = as_of.isoformat()

report_tab, gating_tab, status_tab = st.tabs(
    ["Pregame report", "Date-gating proof", "Build status"]
)

# ---------------------------------------------------------------- report
with report_tab:
    try:
        report = json.loads(dry_run(matchup_id, as_of_str, source))
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()

    hp, ap = report.get("home_win_prob"), report.get("away_win_prob")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        f"{game['home']} (home) win prob", f"{hp:.1%}" if hp is not None else "n/a"
    )
    c2.metric(
        f"{game['away']} (away) win prob", f"{ap:.1%}" if ap is not None else "n/a"
    )
    c3.metric("As of", as_of_str)

    st.warning(
        "Win probability is `stub_net_rating_v0` -- a net-rating heuristic, **not** the "
        "XGBoost model. It ignores the injury list entirely. Sarvvesh's model drops into "
        "this same tool signature."
    )

    st.subheader("What drove it")
    st.markdown(
        '<div class="lede">Every line is a real value pulled through a tool and filtered '
        "to the as-of date. None of this is written by an LLM.</div>",
        unsafe_allow_html=True,
    )
    for factor in report.get("key_factors", []):
        st.markdown(
            f'<div class="card"><div class="v">{factor}</div></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Narrative")
    st.write(report.get("narrative", ""))

    missing = report.get("missing", [])
    if missing:
        st.subheader(f"Not available yet — {len(missing)} of 10 tools")
        st.markdown(
            '<div class="lede">The agent reports gaps instead of guessing. These are the '
            "tools the full report still needs, and who owes each one.</div>",
            unsafe_allow_html=True,
        )
        for m in missing:
            parts = [p.strip() for p in m.split("--")]
            name = parts[0] if parts else m
            owner = parts[1] if len(parts) > 1 else "?"
            needs = " -- ".join(parts[2:]) if len(parts) > 2 else ""
            st.markdown(
                f'<div class="card gap"><div class="k">{name} &nbsp;·&nbsp; {owner}</div>'
                f'<div class="v">{needs}</div></div>',
                unsafe_allow_html=True,
            )

    with st.expander("Raw JSON"):
        st.json(report)

# ---------------------------------------------------------------- gating
with gating_tab:
    st.subheader("Same game, three different days of knowledge")
    st.caption(
        "The 2025-26 season already happened, so any online LLM may remember the "
        "results. Every query carries an as-of date and returns only records from "
        "before it -- this is what makes leakage-free replay possible."
    )

    probe_dates = [
        tip - timedelta(days=n)
        for n in (60, 30, 1)
        if tip - timedelta(days=n) > date(1990, 1, 1)
    ]
    cols = st.columns(len(probe_dates))
    for col, d in zip(cols, probe_dates):
        with col:
            st.markdown(f"**as of {d.isoformat()}**")
            try:
                r = json.loads(dry_run(matchup_id, d.isoformat(), source))
            except Exception as exc:
                st.error(f"{type(exc).__name__}")
                continue
            inj = [
                k
                for k in r.get("key_factors", [])
                if "out as of" in k or "No players" in k
            ]
            st.info(inj[0] if inj else "no injury line")
            hp = r.get("home_win_prob")
            st.caption(
                f"home win prob: {hp:.1%}" if hp is not None else "home win prob: n/a"
            )

    st.success(
        "If the injury lists differ across these columns, the gate is real: the system "
        "cannot see a report filed after the date you asked from."
    )

# ---------------------------------------------------------------- status
with status_tab:
    st.subheader("What is built, and what is blocked")
    st.caption(
        "Generated by probing all ten tools -- `python -m agent.run --status`. "
        "Owners come from the tool contracts in `agent/tools.py`."
    )
    st.code(status_board(matchup_id, as_of_str, source), language="text")
