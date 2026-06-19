"""L3 抽象操作模板 — 跨应用通用的操作模式。

从 L2 技能块中按标签自动聚合，提取不绑定具体应用的抽象模板。
"""

from __future__ import annotations
import json, os, time
from dataclasses import dataclass, field
from typing import Optional

from agent.skill_lib import SkillLibrary, Skill


@dataclass
class AbstractTemplate:
    template_id: str
    name: str
    description: str
    tag_pattern: list[str]          # 触发此模板的标签组合
    concrete_skills: list[str]      # 关联的 L2 技能名列表
    transfer_count: int = 0
    version: int = 1

    def match_tags(self, tags: list[str]) -> bool:
        """Check if a skill's tags match this template."""
        return all(t in tags for t in self.tag_pattern)


class AbstractLibrary:
    """L3 抽象模板库 — 从 L2 技能库自动提取并聚合模板。"""

    def __init__(self, storage_path: str = ""):
        self.templates: list[AbstractTemplate] = []
        self.storage_path = storage_path or "D:/d/tmp/abstracts.json"
        self._load()

    def extract_all(self, skill_lib: SkillLibrary):
        """Scan L2 skills and extract/update L3 templates.

        Groups skills by tag patterns and creates abstract templates
        for each group that has 3+ concrete implementations.
        """
        # Group compiled skills by tag intersection
        tag_groups = {}
        for s in skill_lib.compiled_skills:
            if not s.tags:
                continue
            key = tuple(sorted(s.tags))
            if key not in tag_groups:
                tag_groups[key] = []
            tag_groups[key].append(s.name)

        # Define known abstract patterns
        patterns = [
            ("binary_arithmetic", "二元算术运算",
             ['arithmetic'], "加/减/乘/除等二元计算"),
            ("text_input", "文本输入",
             ['text_input'], "在文本编辑区输入内容"),
            ("save_document", "文档保存",
             ['save_file'], "保存当前文档"),
            ("input_then_save", "输入并保存",
             ['text_input', 'save_file'], "输入内容后保存"),
            ("cross_app_chain", "跨应用串联",
             ['cross_app'], "在多个应用间串联操作"),
        ]

        for tid, name, pattern, desc in patterns:
            matched = []
            for key, skill_names in tag_groups.items():
                if all(t in key for t in pattern):
                    matched.extend(skill_names)

            if matched:
                existing = [t for t in self.templates if t.template_id == tid]
                if existing:
                    existing[0].concrete_skills = matched
                else:
                    self.templates.append(AbstractTemplate(
                        template_id=tid, name=name,
                        description=desc, tag_pattern=pattern,
                        concrete_skills=matched,
                    ))

        # Update L2 abstract_ref pointers
        for t in self.templates:
            for s in skill_lib.skills:
                if s.name in t.concrete_skills:
                    s.abstract_ref = t.template_id

        self._save()
        skill_lib._save()

    def find_matches(self, tags: list[str]) -> list[AbstractTemplate]:
        """Find templates matching a given tag set."""
        return [t for t in self.templates if t.match_tags(tags)]

    def report(self) -> str:
        lines = [f"Abstract Templates ({len(self.templates)}):"]
        for t in self.templates:
            lines.append(f"  {t.template_id:25s} {t.name:15s}"
                         f" {len(t.concrete_skills):2d} skills"
                         f" x{t.transfer_count} transfers")
        return '\n'.join(lines)

    def _save(self):
        try:
            data = [{'template_id': t.template_id, 'name': t.name,
                     'description': t.description, 'tag_pattern': t.tag_pattern,
                     'concrete_skills': t.concrete_skills,
                     'transfer_count': t.transfer_count, 'version': t.version}
                    for t in self.templates]
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
                    self.templates.append(AbstractTemplate(**d))
        except Exception:
            pass
