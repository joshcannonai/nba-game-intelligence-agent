"""Assemble skills/*.md into one review document.

Patrick agreed on 2026-08-03 to review and edit the skills. He is not comfortable
in GitHub, so the review copy needs to be a single readable document rather than
eight files behind a pull request. This builds that document from the skill files,
so the review copy cannot drift from what the agent actually loads.

    python scripts/skills_doc.py                 # -> docs/SKILLS.md
    python scripts/skills_doc.py --stdout        # print instead

Edits made in the document have to come back into `skills/*.md` by hand. That is
the trade for Patrick not needing GitHub, and it is why the header says so.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.skills import load_skills  # noqa: E402
from agent.sources import get_source  # noqa: E402
from agent.tools import build_tools  # noqa: E402

HEADER = """# Agent skills — review copy

**NBA Game Intelligence Agent · CECS 499 · generated {today}**

Each of the agent's tools has a *skill*: a short set of rules telling it when to call
that tool and what to do with the answer. The agent loads these at startup, so these
rules are what it actually follows — this is not documentation written after the fact.

**Patrick — this is the doc from Tuesday's call.** Read it as "is this what we want the
agent to do?" You do not need to touch any code to have an opinion here. Comment
directly on anything that looks wrong, missing, or too strict, and I will move the
changes back into the repo.

Two places I would especially like a second opinion:

1. **`retrieve_injuries`** — we agreed on the call to encode something like "if a
   player averages over 20 ppg and is out, drop the odds by X%". I could not find an
   X the data supports, so the rule currently says the opposite: report the injury and
   let the fitted model price it. The measurement behind that is in the skill itself.
   If you think that is the wrong call, say so.
2. **`retrieve_team_form`** — the "under 5 games is noise" threshold is my judgement,
   not a measured number.

Source of truth is `skills/` in the repo. Regenerate this document with
`python scripts/skills_doc.py`.

---

## The tools at a glance

| Tool | Use when |
|---|---|
{summary}

---
"""


def build(order: list[str]) -> str:
    skills = load_skills()
    rows = []
    for name in order:
        skill = skills.get(name)
        rows.append(
            f"| `{name}` | {skill.use_when if skill else '— no skill file —'} |"
        )

    parts = [HEADER.format(today=date.today().isoformat(), summary="\n".join(rows))]

    for name in order:
        skill = skills.get(name)
        if not skill:
            parts.append(f"\n## `{name}`\n\n> **No skill file.** Needs writing.\n")
            continue
        parts.append(f"\n## `{name}`\n")
        parts.append(f"**Use when:** {skill.use_when}\n")
        # Skill bodies use `##` for their own sections; push them down one level so
        # they nest under the tool heading instead of competing with it.
        parts.append(skill.body.replace("\n## ", "\n### ").replace("\n# ", "\n## "))
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stdout", action="store_true", help="Print instead of writing")
    args = ap.parse_args()

    # Tool order comes from the agent, not from the filesystem, so the document
    # reads in the order the agent is told to use them.
    order = [t.name for t in build_tools(get_source("mock"), include_model=True)]
    text = build(order)

    if args.stdout:
        print(text)
        return

    out = ROOT / "docs" / "SKILLS.md"
    out.write_text(text)
    print(f"wrote {out.relative_to(ROOT)}  ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
