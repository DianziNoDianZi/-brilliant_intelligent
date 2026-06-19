"""技能重组引擎 — 组合独立技能为跨应用复合技能。

核心流程：
1. 分析技能库，找输出/输入状态兼容的技能对
2. LLM 生成衔接动作（Alt+Tab、点击窗口等）
3. 完整序列送入内模拟验证
4. 高置信序列入库为复合技能

这是 Phase 4 层级记忆的"技能重组"机制的独立实现。
"""

from __future__ import annotations
import json, re
from typing import Optional
from dataclasses import dataclass

from agent.skill_lib import SkillLibrary, Skill, COMPILE_THRESHOLD
from agent.simulator import Simulator
from agent.wsg import WorldStateGraph, SubGoal


@dataclass
class CompoundCandidate:
    skill_a: Skill
    skill_b: Skill
    transition_steps: list[SubGoal]
    full_sequence: list[SubGoal]
    confidence: float = 0.0
    feasible: bool = False


# 技能类型标识：用于判断兼容性
SKILL_TAGS = {
    'calculator': {
        'add': 'calc_result',
        'subtract': 'calc_result',
        'multiply': 'calc_result',
        'divide': 'calc_result',
        'add_multi': 'calc_result',
        'clear_then_add': 'calc_result',
    },
    'notepad': {
        'type_text': 'notepad_edit',
        'type_then_save': 'notepad_edit',
        'multi_line': 'notepad_edit',
    },
}

# 兼容规则：(output_tag, input_tag) → 需要衔接
COMPATIBILITY = [
    ('calc_result', 'notepad_edit'),   # 计算器结果 → 记事本输入
]


class SkillRecombinator:
    """组合独立技能为复合技能，经验证后入库。"""

    def __init__(self, planner, simulator: Simulator,
                 skill_lib: SkillLibrary):
        self.planner = planner
        self.simulator = simulator
        self.skill_lib = skill_lib

    def recombin_all(self, wsg: WorldStateGraph) -> list[Skill]:
        """查找所有兼容技能对，尝试重组。"""
        compounds = []
        pairs = self._find_compatible_pairs()
        print(f"[RECOMB] Found {len(pairs)} compatible pairs")

        for skill_a, skill_b in pairs:
            compound = self._try_combine(skill_a, skill_b, wsg)
            if compound:
                compounds.append(compound)
        return compounds

    def _find_compatible_pairs(self) -> list[tuple[Skill, Skill]]:
        """根据兼容规则查找可组合的技能对。"""
        pairs = []
        # Build tag index
        tag_index = {}
        for s in self.skill_lib.skills:
            app_tags = SKILL_TAGS.get(s.task_type, {})
            # Match by name prefix or exact name
            tag = None
            for k, v in app_tags.items():
                if s.name.startswith(k):
                    tag = v
                    break
            if tag:
                if tag not in tag_index:
                    tag_index[tag] = []
                tag_index[tag].append(s)

        for out_tag, in_tag in COMPATIBILITY:
            out_skills = [s for s in tag_index.get(out_tag, []) if s.is_compiled]
            in_skills = [s for s in tag_index.get(in_tag, []) if s.is_compiled]
            for a in out_skills:
                for b in in_skills:
                    if a != b:
                        pairs.append((a, b))
        print(f"[RECOMB] {len(out_skills)} compiled output skills, "
              f"{len(in_skills)} compiled input skills")
        return pairs

    def _try_combine(self, skill_a: Skill, skill_b: Skill,
                     wsg: WorldStateGraph) -> Optional[Skill]:
        """尝试组合两个技能，生成复合技能。"""
        print(f"[RECOMB] Trying: {skill_a.name} → {skill_b.name}")

        # 1. Generate transition steps via LLM
        transitions = self._generate_transitions(skill_a, skill_b, wsg)
        if not transitions:
            print(f"[RECOMB] No transitions generated")
            return None

        # 2. Build full sequence: A + transitions + B
        seq_a = self._skill_to_subgoals(skill_a)
        seq_b = self._skill_to_subgoals(skill_b)
        full_seq = seq_a + transitions + seq_b

        # 3. Inner simulation
        result = self.simulator.rollout(full_seq, wsg)
        confidence = result.avg_confidence

        print(f"[RECOMB] Simulation confidence: {confidence:.2f} "
              f"(sim={result.simulated_count}, exec={result.executed_count})")

        if confidence < self.simulator.confidence_threshold:
            print(f"[RECOMB] Rejected: low confidence")
            return None

        # 4. Compile as compound skill
        name = f"compound_{skill_a.name}_to_{skill_b.name}"
        template = (skill_a.value_template +
                    ['alt_tab'] * len(transitions) +
                    skill_b.value_template)

        compound = Skill(
            name=name,
            task_type='compound',
            pattern={'depends_on': [skill_a.name, skill_b.name],
                     'transitions': len(transitions)},
            value_template=template,
            successes=COMPILE_THRESHOLD,
        )
        self.skill_lib.skills.append(compound)
        self.skill_lib._save()
        print(f"[RECOMB] Compound skill created: {name}")
        return compound

    def _generate_transitions(self, skill_a: Skill, skill_b: Skill,
                               wsg: WorldStateGraph) -> Optional[list[SubGoal]]:
        """LLM 生成衔接动作。

        E.g., calculator → notepad: [Alt+Tab, click text area]
        """
        prompt = (
            f"I have completed task '{skill_a.name}' in the calculator.\n"
            f"Now I need to switch to notepad to do '{skill_b.name}'.\n"
            f"Current screen has these UI elements:\n"
            f"{wsg.serialize_for_llm()}\n\n"
            f"What key presses or clicks are needed to switch from "
            f"calculator to notepad and prepare for the next action?\n"
            f"Output ONLY a JSON list of steps:\n"
            f'[{{"step":1,"action":"key","value":"alt+tab"}},...]'
        )
        response = self.planner._backend.generate(
            prompt, "You are a Windows desktop automation assistant.")
        if not response:
            return []

        # Parse response into SubGoal list
        steps = []
        try:
            data = json.loads(response)
            if isinstance(data, list):
                for item in data:
                    action = item.get('action', 'click')
                    if action == 'key':
                        action = 'type'
                    steps.append(SubGoal(
                        step=item.get('step', len(steps) + 1),
                        action=action,
                        target_id=0,
                        value=item.get('value', ''),
                        description=item.get('description', ''),
                    ))
        except (json.JSONDecodeError, TypeError):
            pass
        return steps if steps else None

    def _skill_to_subgoals(self, skill: Skill) -> list[SubGoal]:
        """Convert a skill template to SubGoal list for simulation."""
        return [
            SubGoal(
                step=i + 1, action='click', target_id=0,
                value='', description=f'skill:{skill.name} step {i+1}',
            )
            for i in range(len(skill.value_template))
        ]
