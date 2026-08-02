"""Streamlit front end for the NBA Game Intelligence Agent.

Runs the deterministic (no-LLM) path by default so it is fast, free, and safe
to demo. Everything shown here comes from the same tools the agent calls --
there is no display-only data and nothing is mocked up for the screenshot.
"""

from __future__ import annotations

import csv
import json
import os
import socket
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.run import _probe_args, dry_run, status_board  # noqa: E402
from agent.sources import SAMPLE_DIR, get_source  # noqa: E402
from agent.tools import build_tools  # noqa: E402
from scripts.gate_snapshot import build_snapshot  # noqa: E402


@st.cache_data(show_spinner="Gating data to the as-of date…")
def snapshot_for(as_of: str) -> str:
    """Materialise the gated copy the agent will read. Cached per date."""
    return str(build_snapshot(date(*(int(p) for p in as_of.split("-")))))


def report_from_snapshot(
    matchup: str, as_of: str, source_kind: str
) -> tuple[dict, dict]:
    """Run the report against a snapshot, in its own interpreter.

    A subprocess, not an env var flipped in place: `agent.sources` captures its
    directory constants at import and caches every reader, so an in-process
    switch would keep reading the ungated data and the gate would be theatre.
    """
    snap = snapshot_for(as_of)
    env = dict(os.environ, NBA_SNAPSHOT_DIR=snap)
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent.run",
            "--dry-run",
            "--source",
            source_kind,
            "--matchup",
            matchup,
            "--as-of",
            as_of,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    manifest = json.loads((Path(snap) / "_manifest.json").read_text())
    return json.loads(out.stdout), manifest


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


# The chat front end runs as its own Streamlit app. Override the port with
# NBA_CHAT_PORT if 8701 is taken.
CHAT_PORT = int(os.environ.get("NBA_CHAT_PORT", "8701"))
CHAT_URL = f"http://localhost:{CHAT_PORT}"


@st.cache_data(ttl=10)
def chat_is_up(url: str) -> bool:
    """Is the chat app actually listening? A dead button mid-demo is worse
    than a disabled one that says how to start it."""
    host, port = urlparse(url).hostname, urlparse(url).port
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False


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
    # Discovered from disk, newest first, so a season landing in data/samples
    # shows up without editing this list. 2026 is the replay/test season.
    seasons = sorted(
        (int(p.stem.rsplit("_", 1)[1]) for p in SAMPLE_DIR.glob("game_logs_*.csv")),
        reverse=True,
    )
    season = st.selectbox("Season sample", seasons, index=0)
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

    # The advisor's architecture (2026-07-28): gate the data on disk first,
    # then point the agent at only that copy. Off by default because it costs a
    # subprocess per run; the query-time filter is always on either way.
    pregate = st.checkbox(
        "Pre-gate data on disk",
        value=False,
        help="Copy only what was knowable by the as-of date into "
        "data/snapshots/<date>, then run the agent against that directory.",
    )

    st.divider()
    if chat_is_up(CHAT_URL):
        st.link_button("🏀  Ask the agent", CHAT_URL, use_container_width=True)
        st.caption("Same tools, same date gate — conversational.")
    else:
        st.button(
            "🏀  Ask the agent",
            disabled=True,
            use_container_width=True,
            help="The chat app is not running.",
        )
        st.caption(f"Start it: `streamlit run ui/chat.py --server.port {CHAT_PORT}`")

source = get_source(source_kind)
as_of_str = as_of.isoformat()

report_tab, tools_tab, gating_tab, status_tab = st.tabs(
    ["Pregame report", "Agent tools", "Date-gating proof", "Build status"]
)

# ---------------------------------------------------------------- report
with report_tab:
    manifest = None
    try:
        if pregate:
            report, manifest = report_from_snapshot(matchup_id, as_of_str, source_kind)
        else:
            report = json.loads(dry_run(matchup_id, as_of_str, source))
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()

    if manifest:
        cleared = sum(f["outcomes_cleared"] for f in manifest["files"])
        dropped = sum(f["rows_in"] - f["rows_out"] for f in manifest["files"])
        st.success(
            f"Agent read a gated copy of the data, not the repo. Building it "
            f"dropped **{dropped:,} rows** and cleared **{cleared:,} game results** "
            f"dated after {as_of_str}. The report below could not have used them "
            "because they were not on disk."
        )

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
        "Win probability is `stub_net_rating_v2` -- net rating, rest and an "
        "injury penalty weighted by minutes. It is **not** the XGBoost model, and "
        "measured over all 1,322 games of 2025-26 the injury term currently makes "
        "it *worse* (63.4% vs 66.3% without it), so treat it as a placeholder with "
        "a known flaw. Sarvesh's model drops into this same tool signature."
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
        st.subheader(f"Awaiting input — {len(missing)} of 7 tools")
        st.markdown(
            '<div class="lede">The agent reports gaps instead of guessing. These tools '
            "are written and callable — they are waiting on data or a model.</div>",
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

# ---------------------------------------------------------------- tools
with tools_tab:
    st.subheader("What the agent is allowed to do")
    st.markdown(
        '<div class="lede">These seven functions are the agent\'s entire world. It cannot '
        "query a database, browse the web, or invent a number — it can only call these, "
        "and every one takes an <b>as-of date</b>. All seven are written by the agent lane; "
        "what varies is whether the data or model behind each one exists yet.</div>",
        unsafe_allow_html=True,
    )

    tools = {t.name: t for t in build_tools(source)}
    probes = _probe_args(matchup_id, as_of_str)

    STATE = {
        "built": ("✅ built", "#22C55E"),
        "stub": ("⚠️ placeholder logic", "#F59E0B"),
        "gap": ("⏳ awaiting input", "#64748B"),
    }

    for name, tool in tools.items():
        try:
            payload = json.loads(tool.invoke(probes[name]))
        except Exception as exc:
            payload = {"status": "awaiting_input", "needs_from": "?", "needs": str(exc)}

        if payload.get("status") == "awaiting_input":
            key, owner = "gap", payload.get("needs_from", "?")
            detail = payload.get("needs", "")
        elif payload.get("warning") or str(payload.get("model", "")).startswith(
            "stub_"
        ):
            key, owner = "stub", "Sarvesh"
            detail = payload.get("warning", "placeholder logic")
        else:
            key, owner, detail = "built", "—", "Returns real, date-gated data."

        label, colour = STATE[key]
        summary = tool.description.strip().split("\n")[0]
        args = ", ".join(tool.args.keys())

        st.markdown(
            f'<div class="card" style="border-left-color:{colour}">'
            f'<div class="k">{label} &nbsp;·&nbsp; input from: {owner}</div>'
            f'<div class="v"><b>{name}</b>({args})<br>{summary}<br>'
            f'<span style="opacity:.65">{detail}</span></div></div>',
            unsafe_allow_html=True,
        )


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

    st.divider()
    st.subheader("The second gate: what never reached the disk")
    st.caption(
        "The columns above prove the query-time filter refuses future records. "
        "This proves something stronger -- with **Pre-gate data on disk** ticked "
        "in the sidebar, the agent reads a copy that never contained them. "
        "Built by `python -m scripts.gate_snapshot --as-of DATE`."
    )
    if manifest:
        st.table(
            [
                {
                    "file": f["file"],
                    "kept": f"{f['rows_out']:,}",
                    "dropped": f"{f['rows_in'] - f['rows_out']:,}",
                    "results cleared": f"{f['outcomes_cleared']:,}",
                    "rule": f["rule"],
                }
                for f in manifest["files"]
            ]
        )
    else:
        st.info(
            "Tick **Pre-gate data on disk** in the sidebar to build and inspect it."
        )

# ---------------------------------------------------------------- status
with status_tab:
    st.subheader("What is built, and what is blocked")
    st.caption(
        "Generated by probing all seven tools -- `python -m agent.run --status`. "
        "Owners come from the tool contracts in `agent/tools.py`."
    )
    st.code(status_board(matchup_id, as_of_str, source), language="text")
