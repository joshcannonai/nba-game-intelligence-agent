# Agent skills

One file per tool. Each says **when the agent should call that tool** and **what to
do with what comes back**. The agent loads these into its system prompt at startup,
so editing a file here changes agent behaviour with no code change.

This is deliberate: the domain rules live in Markdown that anyone on the team can
edit, not buried in Python that only the agent lane touches.

## Format

```markdown
---
tool: retrieve_injuries        # must match the tool's function name exactly
use_when: one line, shown to the agent as the trigger
---

## What it gives you
## When to call it
## How to read it
## Rules
```

## Editing these

- **Keep `tool:` matching the function name.** A skill whose tool does not exist
  fails a test, and a tool with no skill fails a different one.
- **Write rules the agent can actually follow.** "Consider recent form" is not a
  rule. "If a team has played fewer than 5 games this season, form is unreliable —
  say so rather than leaning on it" is.
- **Do not invent thresholds.** If a number is not measured, say it is not
  measured. `skills/retrieve_injuries.md` explains why the obvious
  "star out → drop N%" rule is not in here.
- After editing, run `pytest tests/test_skills.py`.

## Why not a database

We considered putting everything in Postgres and letting the agent query freely.
Rejected: a free-form query surface is exactly how an agent reaches data nobody
intended it to see, and the date gate depends on every read passing through
`agent/sources.py`. Named tools with written rules is a smaller, checkable
surface. See `docs/REPORT.md` §5.
