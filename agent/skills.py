"""Load the per-tool skill files in `skills/` into the agent's system prompt.

A skill is a Markdown file saying when to call one tool and what to do with what
comes back. Keeping that in Markdown rather than in the Python prompt is the point:
the domain rules are the part teammates need to edit, and they should not have to
open `agent/run.py` to do it.

    from agent.skills import skills_block
    prompt = SYSTEM + skills_block([t.name for t in tools])

WHY THE TOOL LIST IS AN ARGUMENT. The block is built from the tools the agent was
actually given, never from whatever happens to be on disk. Arm B and arm C differ by
exactly one tool, and the whole three-arm result rests on that being the *only*
difference. If arm B were handed the win-probability skill it does not have a tool
for, the arms would differ by a tool AND a paragraph of instructions, and the
measurement would be meaningless. `tests/test_skills.py` pins this.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.S)


@dataclass(frozen=True)
class Skill:
    tool: str
    use_when: str
    body: str
    path: Path


def _parse(path: Path) -> Skill | None:
    """Parse one skill file. Returns None for anything without frontmatter."""
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER.match(text)
    if not match:
        return None

    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()

    tool = meta.get("tool")
    if not tool:
        return None
    return Skill(
        tool=tool,
        use_when=meta.get("use_when", ""),
        body=match.group(2).strip(),
        path=path,
    )


@lru_cache(maxsize=1)
def load_skills() -> dict[str, Skill]:
    """Every skill on disk, keyed by the tool it documents.

    Missing directory is not an error. The agent ran without skills before these
    existed and must keep running if someone checks out an older tree.
    """
    if not SKILLS_DIR.is_dir():
        return {}
    skills = {}
    for path in sorted(SKILLS_DIR.glob("*.md")):
        skill = _parse(path)
        if skill:
            skills[skill.tool] = skill
    return skills


def skills_block(tool_names: list[str]) -> str:
    """The prompt section for exactly these tools, in the order given.

    A tool with no skill file is skipped silently here -- the agent should still
    work -- but `tests/test_skills.py` fails on it, so the gap surfaces in CI
    rather than as quietly degraded behaviour at runtime.
    """
    skills = load_skills()
    chosen = [skills[name] for name in tool_names if name in skills]
    if not chosen:
        return ""

    parts = [
        "\n\n=== TOOL SKILLS ===",
        "One entry per tool you have. These are operating rules, not background "
        "reading: where a skill states a rule, follow it.",
    ]
    for skill in chosen:
        parts.append(f"\n--- {skill.tool} ---")
        if skill.use_when:
            parts.append(f"Use when: {skill.use_when}")
        parts.append(skill.body)
    return "\n".join(parts)
