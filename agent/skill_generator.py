"""内模拟驱动的技能生成器。

LLM 提案 → 内模拟验证 → 择优录用 → 入技能库（无需真实执行）。

这形成了新的生长飞轮：
  更多技能 → 更多模式匹配 → 快回路接管更多任务 → 慢回路专注新任务
"""

from __future__ import annotations
import json, re, copy
from typing import Optional
from dataclasses import dataclass

from agent.wsg import WorldStateGraph, SubGoal
from agent.simulator import Simulator
from agent.skill_lib import SkillLibrary, Skill
from agent.intent import IntentVector


@dataclass
class Candidate:
    values: list[str]
    description: str
    score: float = 0.0       # 综合评分
    avg_confidence: float = 0.0
    all_high_confidence: bool = False
    rejected: bool = False
    reason: str = ""


CANDIDATE_PROMPT = """You are a desktop operation planner. Generate 2-3 different VALID ways to accomplish the task.

CRITICAL RULES:
- Calculator: split multi-digit numbers like "12" into ["1", "2"].
- Notepad: use "text_area" for click on text area, and "type:TEXT" for typing.
  CORRECT notepad: ["text_area", "type:Hello World"]
  WRONG notepad:   ["H", "e", "l", "l", "o", " ", "W", "o", "r", "l", "d"]

Output a JSON object with a "candidates" list. Each candidate has:
- values: list of button values or actions to perform
- description: why this approach should work

Available UI elements: {available}

Examples:
Calculator "calculate 3+4":
{{"candidates": [
  {{"values": ["3", "+", "4", "="], "description": "direct addition"}}
]}}

Calculator "calculate 12+34":
{{"candidates": [
  {{"values": ["1", "2", "+", "3", "4", "="], "description": "split digits"}}
]}}

Notepad "type Hello World":
{{"candidates": [
  {{"values": ["text_area", "type:Hello World"], "description": "click text area then type"}}
]}}

Output ONLY valid JSON."""


class SkillGenerator:
    """Generates skills via LLM proposal + inner simulation verification."""

    def __init__(self, planner, simulator: Simulator,
                 skill_lib: SkillLibrary):
        self.planner = planner
        self.simulator = simulator
        self.skill_lib = skill_lib
        self.stats = {'generated': 0, 'accepted': 0, 'rejected': 0}

    def generate(self, instruction: str, wsg: WorldStateGraph,
                 task_type: str = 'calculator') -> Optional[Skill]:
        """Generate a skill for a task using LLM + simulation.

        Returns the compiled Skill if successful, None if no candidate passed.
        """
        # 1. Get candidate sequences from LLM
        candidates = self._propose_candidates(instruction, wsg)
        if not candidates:
            print(f"  [GEN] No candidates from LLM")
            return None

        print(f"  [GEN] {len(candidates)} candidates from LLM")

        # 2. Verify each candidate via inner simulation
        for c in candidates:
            self._verify_candidate(c, wsg)

        # 3. Select best candidate
        best = self._select_best(candidates)
        if best is None:
            print(f"  [GEN] All candidates rejected")
            return None

        print(f"  [GEN] Selected: {best.description} "
              f"(conf={best.avg_confidence:.2f}, values={best.values})")

        # 4. Compile into skill without real execution
        skill = self._compile_skill(best, instruction, task_type)
        self.stats['generated'] += 1
        if skill:
            self.stats['accepted'] += 1
        return skill

    def _propose_candidates(self, instruction: str, wsg: WorldStateGraph
                            ) -> list[Candidate]:
        """Ask LLM for multiple candidate action sequences."""
        # Filter WSG to only calculator-relevant buttons
        relevant = [e for e in wsg.entities
                     if e.type == 'button' and e.text and e.width >= 30]
        available = sorted(set(e.text for e in relevant))
        prompt = f"## Instruction\n{instruction}\n\n## Available buttons\n{available}"
        response = self.planner._backend.generate(
            prompt, CANDIDATE_PROMPT)

        if not response:
            return []

        # Clean LLM response: remove common formatting issues
        cleaned = response.strip()
        if cleaned.startswith('```'):
            # Remove code fences
            import re
            cleaned = re.sub(r'```(?:json)?\s*', '', cleaned)
            cleaned = cleaned.rstrip('`').strip()
        # Replace double braces with singles (LLM copies prompt format)
        cleaned = cleaned.replace('{{', '{').replace('}}', '}')

        try:
            data = json.loads(cleaned)
            raw = data.get("candidates", data if isinstance(data, list) else [])
            if isinstance(raw, dict):
                raw = [raw]
            return [
                Candidate(values=c.get("values", []),
                          description=c.get("description", ""))
                for c in raw if c.get("values")
            ]
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            print(f"  [GEN] JSON parse failed: {e}")
            return []

    def _verify_candidate(self, candidate: Candidate, wsg: WorldStateGraph):
        """Run inner simulation on a candidate sequence."""
        # Convert values to SubGoal list
        plan = self._values_to_subgoals(candidate.values)
        if not plan:
            candidate.rejected = True
            candidate.reason = "failed to parse"
            return

        # Run multi-step rollout
        result = self.simulator.rollout(plan, wsg)
        candidate.avg_confidence = result.avg_confidence
        candidate.score = result.avg_confidence

        # Penalize long sequences
        if len(candidate.values) > 10:
            candidate.score *= 0.9

        # Check if all steps had high confidence
        confs = [s.confidence for s in result.steps]
        candidate.all_high_confidence = all(c >= 0.5 for c in confs)

        if not candidate.all_high_confidence:
            candidate.rejected = True
            candidate.reason = f"low confidence steps: confs={confs}"
            self.stats['rejected'] += 1
        else:
            candidate.rejected = False

    def _select_best(self, candidates: list[Candidate]) -> Optional[Candidate]:
        """Pick the best non-rejected candidate."""
        valid = [c for c in candidates if not c.rejected]
        if not valid:
            return None
        valid.sort(key=lambda c: (-c.score, len(c.values)))
        return valid[0]

    def _compile_skill(self, candidate: Candidate,
                       instruction: str,
                       task_type: str = 'calculator') -> Optional[Skill]:
        """Compile candidate into a skill without real execution."""
        values = candidate.values
        if len(values) < 2:
            return None

        # Detect template: digits become A, B, C... (26 variable limit)
        template = []
        vc = 0
        for v in values:
            if isinstance(v, str) and v.isdigit():
                template.append(chr(65 + vc))  # A, B, C...
                vc += 1
            else:
                template.append(v)

        # Determine operator
        operators = [v for v in values if v in '+-*/=']
        operator = operators[0] if operators else '+'

        if any('type:' in v for v in values):
            name = "notepad_type"
        else:
            name = f"gen_{operator}" if operator else "gen_op"
        for s in self.skill_lib.skills:
            if s.name == name and not s.deprecated:
                name = f"{name}_v{s.version + 1}"

        from agent.skill_lib import COMPILE_THRESHOLD
        skill = Skill(
            name=name,
            task_type=task_type,
            pattern={"operator": operator, "num_operands": vc},
            value_template=template,
            count=COMPILE_THRESHOLD,
            successes=COMPILE_THRESHOLD,
        )
        self.skill_lib.skills.append(skill)
        self.skill_lib._save()
        return skill

    def _values_to_subgoals(self, values: list[str]) -> list[SubGoal]:
        """Convert values to SubGoal list for simulation.

        Handles "type:..." prefix for type actions.
        """
        if not values:
            return []
        subgoals = []
        for i, v in enumerate(values):
            if v.startswith('type:'):
                text = v[5:]
                subgoals.append(SubGoal(
                    step=i + 1, action='type', target_id=0,
                    value=text, description=f'type "{text}"'))
            else:
                subgoals.append(SubGoal(
                    step=i + 1, action='click', target_id=0,
                    value='', description=f'click {v}'))
        return subgoals