"""Skill library — compiled skills with version management and batch compilation.

Skills are compiled from multiple successful slow-loop executions (batch trigger).
New skills are compared against existing ones before replacing.
"""

from __future__ import annotations
import json, os, re
from typing import Optional
from dataclasses import dataclass, field


COMPILE_THRESHOLD = 3  # minimum successes before a skill is compiled


@dataclass
class Skill:
    name: str
    task_type: str
    pattern: dict
    value_template: list[str]
    action_template: list[str] = None
    version: int = 1
    count: int = 0
    successes: int = 0
    failures: int = 0
    deprecated: bool = False
    app: str = None                 # L2: "calculator" / "notepad" / "compound"
    tags: list[str] = None          # L2: ["arithmetic","addition"] auto-detected
    last_used: float = 0.0          # L2: timestamp of last execution
    abstract_ref: str = ""          # L2 → L3 指针: 关联的抽象模板 ID

    def __post_init__(self):
        if self.action_template is None:
            self.action_template = ["click"] * len(self.value_template)
        if self.app is None:
            self.app = self._infer_app()
        if self.tags is None:
            self.tags = self._infer_tags()

    def _infer_app(self) -> str:
        """Auto-detect app from template content."""
        t = ' '.join(self.value_template).lower()
        if any(w in t for w in ['text_area', 'save', 'filename', 'type:']):
            return 'notepad'
        if any(v in t for v in ['0','1','2','3','4','5','6','7','8','9','+','-','*','/','=']):
            return 'calculator'
        return 'generic'

    def _infer_tags(self) -> list[str]:
        """Auto-detect operation type tags from template."""
        t = ' '.join(self.value_template)
        tags = []
        # Arithmetic operators
        if '+' in t: tags.extend(['arithmetic', 'addition'])
        if '-' in t: tags.extend(['arithmetic', 'subtraction'])
        if '*' in t: tags.extend(['arithmetic', 'multiplication'])
        if '/' in t: tags.extend(['arithmetic', 'division'])
        # Notepad operations
        if 'text_area' in t or 'type:' in t: tags.append('text_input')
        if 'save' in t: tags.append('save_file')
        if 'LINE' in t or '\\n' in t: tags.append('multi_line')
        # Multi-digit
        var_count = sum(1 for v in self.value_template if isinstance(v, str) and len(v) == 1 and v.isalpha())
        if var_count >= 4: tags.append('multi_digit')
        # Compound
        if 'alt_tab' in t: tags.append('cross_app')
        if not tags:
            tags.append('generic')
        return tags

    @property
    def accuracy(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.0

    @property
    def is_compiled(self) -> bool:
        return self.successes >= COMPILE_THRESHOLD


class SkillLibrary:
    """Skill library with version management and batch compilation."""

    def __init__(self, storage_path: str = ""):
        self.skills: list[Skill] = []
        self._pending_compile: dict[str, list] = {}  # pattern_key → [value_seqs]
        self.storage_path = storage_path or "/d/tmp/skills.json"
        self._load()
        self._init_defaults()

    def _init_defaults(self):
        existing = {(s.name, tuple(s.value_template)) for s in self.skills}
        defaults = [
            # Calculator: binary operations (4 templates)
            Skill("add", "calculator", {"operator": "+", "num_operands": 2},
                  ["X", "+", "Y", "="]),
            Skill("subtract", "calculator", {"operator": "-", "num_operands": 2},
                  ["X", "-", "Y", "="]),
            Skill("multiply", "calculator", {"operator": "*", "num_operands": 2},
                  ["X", "*", "Y", "="]),
            Skill("divide", "calculator", {"operator": "/", "num_operands": 2},
                  ["X", "/", "Y", "="]),

            # Calculator: multi-digit operands
            Skill("add_multi", "calculator", {"operator": "+", "num_operands": 2},
                  ["X", "X", "+", "Y", "Y", "="],
                  ["click","click","click","click","click","click"]),

            # Calculator: clear then new operation (chained)
            Skill("clear_then_add", "calculator", {"operator": "+", "num_operands": 2},
                  ["C", "X", "+", "Y", "="],
                  ["click","click","click","click","click"]),

            # Notepad: type text and save
            Skill("type_text", "notepad", {},
                  ["text_area", "VALUE"],
                  ["click", "type"]),
            Skill("save_file", "notepad", {},
                  ["file", "save", "filename", "save_button"],
                  ["click", "click", "type", "click"]),
            Skill("type_then_save", "notepad", {},
                  ["text_area", "VALUE", "file", "save", "filename", "save_button"],
                  ["click", "type", "click", "click", "type", "click"]),

            # Notepad: multi-line
            Skill("multi_line", "notepad", {},
                  ["text_area", "LINE1", "enter", "LINE2"],
                  ["click", "type", "click", "type"]),
        ]
        for s in defaults:
            key = (s.name, tuple(s.value_template))
            if key not in existing:
                self.skills.append(s)

    def match(self, values: list[str], task_type: str) -> Optional[Skill]:
        """Find best non-deprecated compiled skill for a value sequence."""
        operators = [v for v in values if v in '+-*/=']
        operands = [v for v in values if v.isdigit()]
        candidates = []

        for s in self.skills:
            if s.deprecated or s.task_type != task_type:
                continue
            p = s.pattern
            expected_op = p.get("operator", "")
            if expected_op and expected_op not in operators:
                continue
            expected_n = p.get("num_operands", 0)
            if expected_n and len(operands) != expected_n:
                continue
            candidates.append(s)

        if not candidates:
            return None
        # Pick best: highest accuracy, then highest count
        candidates.sort(key=lambda s: (s.accuracy, s.count), reverse=True)
        return candidates[0]

    def match_template(self, values: list[str]) -> Optional[list[str]]:
        """Extract template pattern from a value sequence.

        E.g. ["3", "+", "4", "="] → ["X", "+", "Y", "="]
        """
        template = []
        var_counter = 0
        for v in values:
            if v.isdigit():
                template.append(chr(ord('X') + var_counter))
                var_counter += 1
            else:
                template.append(v)
        return template if var_counter > 0 and '=' in template else None

    def pattern_key(self, values: list[str]) -> str:
        """Generate a pattern key for grouping similar sequences.

        E.g. ["3","+","4","="] and ["5","+","8","="] → "binary_+_="
        """
        operators = [v for v in values if v in '+-*/=']
        operands = [v for v in values if v.isdigit()]
        op = operators[0] if operators else '?'
        count = len(operands)
        return f"{'single' if count==1 else 'binary' if count==2 else 'multi'}_{op}_{operators[-1] if operators else '?'}"

    def record_success(self, values: list[str], task_type: str
                       ) -> Optional[Skill]:
        """Record a success. Returns skill only when newly compiled."""
        skill = self.match(values, task_type)
        if skill:
            skill.successes += 1
            skill.count += 1
            self._save()
            # Only return when just reached compile threshold
            if skill.successes == COMPILE_THRESHOLD:
                return skill
            return None

        # No existing skill → accumulate for batch compile
        key = self.pattern_key(values)
        if key not in self._pending_compile:
            self._pending_compile[key] = []
        self._pending_compile[key].append(values)
        self._save()

        if len(self._pending_compile[key]) >= COMPILE_THRESHOLD:
            return self._batch_compile(key, task_type)
        return None

    def record_failure(self, values: list[str], task_type: str):
        """Record a failed execution. Triggers degradation at threshold."""
        skill = self.match(values, task_type)
        if skill:
            skill.failures += 1
            self._save()

    @property
    def degraded_skills(self) -> list[str]:
        """Names of skills that have failed too often."""
        return [s.name for s in self.skills
                if s.failures >= COMPILE_THRESHOLD and not s.deprecated]

    def _batch_compile(self, key: str, task_type: str) -> Optional[Skill]:
        """Compile a skill from accumulated examples."""
        examples = self._pending_compile.get(key, [])
        if len(examples) < COMPILE_THRESHOLD:
            return None

        # Use the most common template
        templates = {}
        for ex in examples:
            t = tuple(self.match_template(ex) or [])
            templates[t] = templates.get(t, 0) + 1

        best_template = max(templates, key=templates.get)
        if not best_template:
            return None

        # Extract operator
        operators = [v for v in best_template if v in '+-*/=']
        operator = operators[0] if operators else ''
        name = operator if operator else task_type

        new_skill = Skill(
            name=name, task_type=task_type,
            pattern={"operator": operator, "num_operands": len(best_template) - 2},
            value_template=list(best_template),
            successes=len(examples), count=len(examples),
        )

        # Version conflict resolution
        existing = [s for s in self.skills
                    if s.name == name and not s.deprecated]
        if existing:
            old = existing[0]
            if new_skill.accuracy >= old.accuracy and new_skill.count >= old.count * 0.8:
                old.deprecated = True
                new_skill.version = old.version + 1
                self.skills.append(new_skill)
                self._pending_compile.pop(key, None)
                self._save()
                return new_skill
            else:
                # New skill not good enough; keep accumulating
                return None
        else:
            self.skills.append(new_skill)
            self._pending_compile.pop(key, None)
            self._save()
            return new_skill

    def expected_result(self, values: list[str]) -> Optional[str]:
        """Compute expected calculator display from value sequence.

        E.g. ["3", "+", "4", "="] → "7"
        """
        try:
            # Extract expression before "="
            if '=' in values:
                eq_idx = values.index('=')
                expr = ''.join(values[:eq_idx])
                # Evaluate safely (digits and operators only)
                if all(c in '0123456789+-*/' for c in expr):
                    result = eval(expr)
                    return str(result)
        except Exception:
            pass
        return None

    def verify_and_record(self, values: list[str], task_type: str,
                           actual_display: str) -> bool:
        """Verify execution result. Records success or failure.

        Returns True if result matches expected.
        """
        expected = self.expected_result(values)
        if expected is None:
            self.record_success(values, task_type)
            return True

        # Extract number from display text (e.g. "显示为 7" → "7")
        import re
        display_nums = re.findall(r'-?\d+\.?\d*', actual_display)
        if not display_nums:
            self.record_failure(values, task_type)
            return False

        if expected == display_nums[0]:
            self.record_success(values, task_type)
            return True
        else:
            self.record_failure(values, task_type)
            return False

    def fill_template(self, skill: Skill, instruction: str) -> Optional[list[str]]:
        """Fill skill template with values from instruction.

        Calculator: template ['A','B','+','C','D','='], "12+34"
        → ['1','2','+','3','4','=']

        Notepad: template ['text_area','type:Hello'], "type World"
        → ['text_area','type:World']
        """
        result = list(skill.value_template)

        # Notepad: replace VALUE or type:* with text from instruction
        type_positions = [i for i, v in enumerate(result)
                          if isinstance(v, str) and (v == 'VALUE' or v.startswith('type:'))]
        if type_positions:
            # Extract quoted text first, then fallback to after "type"
            text = ''
            m = re.search(r'["\'](.*?)["\']', instruction)
            if m:
                text = m.group(1)
            else:
                text = re.sub(r'(?i)^type\s*', '', instruction).strip()
                text = re.sub(r'\s+and\s+save.*', '', text).strip()
            for pos in type_positions:
                result[pos] = f'type:{text}'
            return result

        # Calculator: replace digit variables A, B, C...
        numbers = re.findall(r'\d+', instruction)
        all_digits = list(''.join(numbers))
        var_positions = [i for i, v in enumerate(skill.value_template)
                         if isinstance(v, str) and len(v) == 1 and v.isalpha()]
        if len(all_digits) < len(var_positions):
            return None
        # Process RIGHT-TO-LEFT to avoid position shifting
        for idx in reversed(range(len(var_positions))):
            pos = var_positions[idx]
            if idx < len(all_digits):
                result[pos] = all_digits[idx]
        return result

    def _save(self):
        try:
            data = [{'name': s.name, 'task_type': s.task_type,
                     'pattern': s.pattern, 'value_template': s.value_template,
                     'version': s.version, 'count': s.count,
                     'successes': s.successes, 'failures': s.failures,
                     'deprecated': s.deprecated, 'app': s.app,
                     'tags': s.tags, 'last_used': s.last_used,
                     'abstract_ref': s.abstract_ref}
                    for s in self.skills]
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load(self):
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path) as f:
                    data = json.load(f)
                for d in data:
                    self.skills.append(Skill(**d))
        except Exception:
            pass

    @property
    def compiled_skills(self) -> list[Skill]:
        return [s for s in self.skills if s.is_compiled and not s.deprecated]

    def __len__(self):
        return len(self.skills)

    def __str__(self):
        compiled = len(self.compiled_skills)
        pending = sum(len(v) for v in self._pending_compile.values())
        return f"SkillLibrary({len(self.skills)} total, {compiled} compiled, {pending} pending)"
