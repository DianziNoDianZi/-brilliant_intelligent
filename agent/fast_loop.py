"""Fast loop — skill library based, no MLP.

Fast path: compiled skill match → fill_template → execute (no LLM).
Slow path: LLM plans → execute → compile skill for future.
"""

from __future__ import annotations
import re
from typing import Optional

from agent.intent import IntentVector
from agent.skill_lib import SkillLibrary


class FastLoop:
    """Fast-slow loop coordinator.

    Fast path: compiled skill matches → fill_template → execute directly.
    Slow path: LLM plans → execute → batch compile skill.
    """

    def __init__(self, skill_lib: Optional[SkillLibrary] = None):
        self.skill_lib = skill_lib or SkillLibrary()
        self.stats = {'fast_hits': 0, 'slow_calls': 0}

    def decide(self, intent: IntentVector) -> tuple[bool, Optional[list[str]], Optional[str]]:
        """Decide fast or slow path.

        Returns: (use_fast, button_values, skill_name)
        - use_fast=True: fast loop can handle it (skill_name populated)
        - use_fast=False: need to call slow loop (LLM)
        """
        values = self._parse_instruction(intent.raw_instruction)
        if not values:
            self.stats['slow_calls'] += 1
            return False, None, None

        skill = self.skill_lib.match(values, intent.task_type)
        if skill and skill.is_compiled:
            # Check if this skill is degraded
            if skill.name in self.skill_lib.degraded_skills:
                self.stats['slow_calls'] += 1
                return False, None, skill.name

            vals = self.skill_lib.fill_template(skill, intent.raw_instruction)
            if vals:
                self.stats['fast_hits'] += 1
                return True, vals, skill.name

        self.stats['slow_calls'] += 1
        return False, None, None

    def record_success(self, intent: IntentVector, values: list[str]):
        """Record a successful execution for batch compilation."""
        self.skill_lib.record_success(values, intent.task_type)

    def _parse_instruction(self, instruction: str) -> Optional[list[str]]:
        """Extract button value sequence from instruction.

        Calculator: "calculate 3+4" → ["3", "+", "4", "="]
        Notepad: "type Hello World" → ["text_area", "type:Hello World"]
        """
        il = instruction.lower()
        # Notepad: type text
        if il.startswith('type') or 'type ' in il:
            # Extract the text after "type"
            match = re.search( r'["\'](.*?)["\']', instruction)
            if match:
                text = match.group(1)
            else:
                text = re.sub(r'(?i)type\s*', '', instruction, count=1).strip()
                text = re.sub(r'\s+and\s+save.*', '', text).strip()
            if text:
                return ["text_area", f"type:{text}"]

        # Calculator
        numbers = re.findall(r'\d+', instruction)
        ops = re.findall(r'[+\-*/]', instruction)
        if numbers and ops:
            result = []
            for d in numbers[0]:
                result.append(d)
            result.append(ops[0])
            for n in numbers[1:]:
                for d in n:
                    result.append(d)
            result.append('=')
            return result
        return None

    def __str__(self):
        return (f"FastLoop(fast={self.stats['fast_hits']}, "
                f"slow={self.stats['slow_calls']})")
