"""The skills layer: one Markdown file per tool, loaded into the system prompt.

The rules in `skills/` are meant to be edited by teammates who are not touching
Python. That only stays safe if a bad edit fails loudly here rather than quietly
changing what the agent does.

The load-bearing test is `test_arm_b_and_arm_c_skill_blocks_differ_by_one_skill`.
The three-arm result depends on the arms differing by exactly one tool; if the
skills block leaked the win-probability rules into arm B, the arms would differ by
a tool AND a page of instructions, and §9.6 would be measuring nothing.
"""

from __future__ import annotations

import re

import pytest

from agent.skills import SKILLS_DIR, load_skills, skills_block
from agent.sources import get_source
from agent.tools import build_tools


@pytest.fixture(scope="module")
def tool_names() -> list[str]:
    return [t.name for t in build_tools(get_source("mock"), include_model=True)]


def test_every_tool_has_a_skill(tool_names):
    """A tool with no skill silently gets no rules. Catch it here."""
    missing = [n for n in tool_names if n not in load_skills()]
    assert not missing, f"tools with no skills/*.md file: {missing}"


def test_every_skill_names_a_real_tool(tool_names):
    """A skill for a tool that does not exist is a rule nobody will ever follow."""
    orphans = [tool for tool in load_skills() if tool not in tool_names]
    assert not orphans, f"skills naming tools that do not exist: {orphans}"


def test_every_skill_file_parsed():
    """A malformed file is skipped by the loader, so count files against skills."""
    on_disk = {p for p in SKILLS_DIR.glob("*.md") if p.name != "README.md"}
    parsed = {s.path for s in load_skills().values()}
    assert on_disk == parsed, (
        f"failed to parse (missing or malformed frontmatter): {on_disk - parsed}"
    )


def test_skills_carry_a_use_when():
    """`use_when` is the trigger the agent sees. An empty one is a silent no-op."""
    empty = [t for t, s in load_skills().items() if not s.use_when.strip()]
    assert not empty, f"skills with no use_when: {empty}"


def test_block_contains_only_requested_tools():
    block = skills_block(["retrieve_injuries"])
    assert "retrieve_injuries" in block
    assert "predict_win_probability" not in block


def test_empty_tool_list_yields_no_block():
    """No tools means no skills section at all, not an empty header."""
    assert skills_block([]) == ""


def test_arm_b_and_arm_c_skill_blocks_differ_by_one_skill():
    """The arms must differ by exactly the win-probability tool and its rules."""
    source = get_source("mock")
    arm_c = [t.name for t in build_tools(source, include_model=True)]
    arm_b = [t.name for t in build_tools(source, include_model=False)]

    assert set(arm_c) - set(arm_b) == {"predict_win_probability"}

    block_c, block_b = skills_block(arm_c), skills_block(arm_b)
    assert "predict_win_probability" not in block_b
    assert "predict_win_probability" in block_c

    # Everything except that one section must be identical between the arms.
    # Strip each section: whichever skill lands last has no trailing newline, so
    # an unstripped compare reports a spurious second difference.
    def sections(block: str) -> set[str]:
        return {s.strip() for s in re.split(r"\n--- \w+ ---\n", block)[1:]}

    only_in_c = sections(block_c) - sections(block_b)
    assert len(only_in_c) == 1, (
        "arms differ by more than the win-probability skill; the three-arm "
        "comparison would no longer be measuring one variable"
    )


def test_injury_skill_forbids_a_hand_rolled_penalty():
    """The measured finding in eval/injury_impact.py has to survive edits.

    If someone helpfully adds "drop the odds 10% when a star is out", the agent
    goes back to the behaviour that lost 15 of 19 overrides.
    """
    body = load_skills()["retrieve_injuries"].body.lower()
    assert "do not apply your own injury penalty" in body
    assert "eval.injury_impact" in body


def test_win_probability_skill_states_the_override_evidence():
    body = load_skills()["predict_win_probability"].body
    assert "15 of them" in body or "15 times" in body, (
        "the deference rule must carry the measurement that justifies it, or it "
        "reads as arbitrary and gets edited away"
    )
