"""Chat front end for the NBA Game Intelligence Agent.

    streamlit run ui/chat.py

Same ten tools and the same as-of gate as ui/app.py -- this is the
conversational view of them. It runs free and key-less by default:
"deterministic" mode routes your question to one tool and shows you exactly
what that tool returned, with no LLM in the loop. Switch the backend in the
sidebar to let Claude or local Gemma 4 do the routing instead.

Every answer is a tool result. Nothing here is written by a language model
unless you pick a model, and even then the numbers come from the tools.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.run import _probe_args, dry_run  # noqa: E402
from agent.sources import get_source  # noqa: E402
from agent.tools import build_tools  # noqa: E402

ACCENT = "#F97316"

st.set_page_config(page_title="NBA Agent — Chat", page_icon="🏀", layout="centered")

st.markdown(
    f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 820px; }}
  h1 {{ font-size: 1.6rem !important; letter-spacing: -0.02em; font-weight: 700; }}

  /* Both bubbles carry the accent: the user's as a fill, the agent's as a rule. */
  .me, .bot {{
      border-radius: 10px; padding: 0.7rem 0.95rem; line-height: 1.5;
      font-size: 0.94rem;
  }}
  .me  {{ background: rgba(249,115,22,0.12); border: 1px solid rgba(249,115,22,0.38); }}
  .bot {{ background: #151C25; border: 1px solid #232C38;
          border-left: 3px solid {ACCENT}; }}

  .bot b, .me b {{ color: {ACCENT}; }}
  .tool {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em;
           opacity: 0.6; margin-bottom: 0.35rem; }}
  .caveat {{ font-size: 0.8rem; opacity: 0.72; margin-top: 0.5rem;
             border-top: 1px solid #232C38; padding-top: 0.45rem; }}
  .gated {{ border-left-color: #64748B; }}

  [data-testid="stChatMessage"] {{ background: transparent; padding: 0.15rem 0; }}
  .stChatInput textarea:focus {{ border-color: {ACCENT} !important; }}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_games(season: int) -> list[dict]:
    import csv

    path = REPO_ROOT / "data" / "samples" / f"game_logs_{season}.csv"
    if not path.exists():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def parse_iso(s: str) -> date:
    y, m, d = (int(p) for p in s.split("-"))
    return date(y, m, d)


# Keyword -> tool. The point of the deterministic mode is that you can see the
# routing: there is no hidden model deciding what to call.
ROUTES: list[tuple[tuple[str, ...], str]] = [
    (
        ("line", "spread", "odds", "vegas", "moneyline", "total", "favor"),
        "retrieve_betting_line",
    ),
    (("injur", "out", "hurt", "available"), "retrieve_injuries"),
    (("form", "streak", "last 10", "recent"), "retrieve_team_form"),
    (("rest", "back-to-back", "b2b", "schedule"), "retrieve_schedule"),
    (("news",), "retrieve_news"),
    (
        ("win", "probability", "odds of winning", "who wins", "predict"),
        "predict_win_probability",
    ),
    (("best player", "mvp", "star"), "predict_best_player"),
    (("stat line", "points", "rebounds", "assists"), "predict_stat_line"),
    (("split",), "retrieve_player_splits"),
    (("context", "matchup", "overview"), "retrieve_matchup_context"),
]


def route(question: str) -> str | None:
    q = question.lower()
    for keywords, name in ROUTES:
        if any(k in q for k in keywords):
            return name
    return None


def render_betting_line(p: dict) -> str:
    """The one tool whose honest answer needs more than a JSON dump."""
    if p.get("status") == "gated":
        return (
            "<b>No line.</b> " + p["reason"] + "<div class='caveat'>This refusal is "
            "the date gate working. The closing line sits one column from the "
            "final score in the source data.</div>"
        )
    if p.get("status") != "ok":
        return "<b>No line found.</b> " + p.get("reason", "")

    fav, total = p["favorite"], p["total"]
    body = (
        f"<b>{fav}</b> is favoured. "
        f"Home {p['spread_home']:+g}, away {p['spread_away']:+g}"
        + (f", total {total:g}." if total is not None else ", total unavailable.")
    )
    if p.get("moneyline_home") is not None:
        body += f" Moneyline {p['moneyline_away']:+g} / {p['moneyline_home']:+g}."
    for note in p.get("unavailable", []):
        body += f"<div class='caveat'>{note}</div>"
    body += f"<div class='caveat'>{p['caveat']}</div>"
    return body


def answer_deterministically(question: str, matchup_id: str, as_of: str, source) -> str:
    name = route(question)
    if name is None:
        return (
            "<b>I route questions to one of ten tools</b>, and nothing in that "
            "matched. Try asking about the line, injuries, team form, rest, news, "
            "win probability, or a player's stat line — or switch the backend to "
            "Claude in the sidebar and ask freely."
        )

    tools = {t.name: t for t in build_tools(source)}
    args = _probe_args(matchup_id, as_of)[name]
    try:
        payload = json.loads(tools[name].invoke(args))
    except Exception as exc:
        return f"<div class='tool'>{name}</div><b>{type(exc).__name__}</b>: {exc}"

    head = f"<div class='tool'>{name}</div>"

    if name == "retrieve_betting_line":
        return head + render_betting_line(payload)

    if payload.get("status") == "awaiting_input":
        return (
            head
            + f"<b>Not built yet.</b> Input from {payload.get('needs_from', '?')}. "
            + payload.get("needs", "")
        )

    return (
        head
        + f"<pre style='white-space:pre-wrap;margin:0'>{json.dumps(payload, indent=2)}</pre>"
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
    default_i = next(
        (i for i, g in enumerate(options) if g["game_date"][5:7] in ("12", "01")), 0
    )
    pick = st.selectbox(
        "Game", range(len(labels)), index=default_i, format_func=lambda i: labels[i]
    )
    game = options[pick]

    tip = parse_iso(game["game_date"])
    matchup_id = f"{game['away']}-{game['home']}-{game['game_date']}"
    as_of = st.date_input(
        "As-of date (what we knew)",
        value=tip - timedelta(days=1),
        max_value=tip - timedelta(days=1),
    )
    st.caption(f"`{matchup_id}`")

    backend = st.radio(
        "Backend",
        ["deterministic", "anthropic", "ollama"],
        help="deterministic routes your question to one tool, free and key-less. "
        "anthropic needs ANTHROPIC_API_KEY; ollama needs a local Gemma 4.",
    )
    source_kind = st.radio("Data source", ["real", "mock"], horizontal=True)

    if st.button("Clear conversation"):
        st.session_state.messages = []

source = get_source(source_kind)
as_of_str = as_of.isoformat()

st.title("🏀 Ask the agent")
st.caption(
    f"{matchup_id} · answering only from what was knowable on {as_of_str}. "
    "Every reply is a tool result."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for role, body in st.session_state.messages:
    with st.chat_message(role, avatar="🏀" if role == "assistant" else None):
        css = "bot" if role == "assistant" else "me"
        st.markdown(f'<div class="{css}">{body}</div>', unsafe_allow_html=True)

if question := st.chat_input("Who's favoured? Who's out? What's the line?"):
    st.session_state.messages.append(("user", question))
    with st.chat_message("user"):
        st.markdown(f'<div class="me">{question}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="🏀"):
        with st.spinner("Calling tools…"):
            if backend == "deterministic":
                reply = answer_deterministically(
                    question, matchup_id, as_of_str, source
                )
            else:
                try:
                    from agent.run import run_matchup

                    reply = (
                        "<div class='tool'>agent · "
                        + backend
                        + "</div><pre style='white-space:pre-wrap;margin:0'>"
                        + run_matchup(matchup_id, as_of_str, source, backend)
                        + "</pre>"
                    )
                except Exception as exc:
                    reply = (
                        f"<b>{type(exc).__name__}</b>: {exc}"
                        "<div class='caveat'>Switch the backend to deterministic "
                        "to run without a key.</div>"
                    )
        st.markdown(f'<div class="bot">{reply}</div>', unsafe_allow_html=True)
    st.session_state.messages.append(("assistant", reply))

with st.expander("What this can answer"):
    st.markdown(
        "The agent's whole world is ten functions, each taking an as-of date. "
        "In deterministic mode your question is keyword-routed to one of them:"
    )
    st.code("\n".join(f"{name:<28} {', '.join(k)}" for k, name in ROUTES), "text")
    st.caption(
        "Nothing is generated. If a tool has no data it says so and names who "
        "owes the input — see the Build status tab in ui/app.py."
    )
    st.code(dry_run(matchup_id, as_of_str, source), language="json")
