"""LLM-based planner — converts natural language into action sequences.

Calculator: outputs button values ["3", "+", "4", "="]
Notepad: outputs steps with "type:" prefix ["text_area", "type:Hello"]
"""

from __future__ import annotations
import json
import re
from typing import Optional

from agent.wsg import WorldStateGraph, SubGoal

SYSTEM_PROMPT = """You are a desktop operation planner. Given UI elements and an instruction, output the action sequence.

## Calculator
For "calculate 3+4": ["3", "+", "4", "="]
For "compute 15*2": ["1", "5", "*", "2", "="]
Rules: use 0-9, +, -, *, /, =; last step is always "="

## Notepad / Text input
For "type Hello World": find the text input area (type=input), click it, then type.
Output: ["text_area", "type:Hello World"]

## Output
ONLY a JSON list. Nothing else."""


def _normalize_val(v: str) -> str:
    """Map Chinese/Long-form values to standard English symbols."""
    mapping = {
        '一':'1','二':'2','三':'3','四':'4','五':'5',
        '六':'6','七':'7','八':'8','九':'9','零':'0',
        '加':'+','减':'-','乘以':'*','除以':'/','等于':'=',
    }
    return mapping.get(v.strip(), v)

def _parse_value_sequence(text: str, wsg=None) -> Optional[list[str]]:
    """Parse LLM response into button values like ["3", "+", "4", "="].

    Handles three formats:
    - Value array:   ["3", "+", "4", "="]
    - Object array:  [{"step":1,"action":"click","target_id":25}, ...]
    - target_id lookup via WSG: id=25 → text="6"
    """
    if not text:
        return None
    json_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', text)
    if json_match:
        text = json_match.group(1)
    try:
        data = json.loads(text.strip())
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            # Format 1: string array
            if isinstance(first, str):
                return [_normalize_val(str(v)) for v in data]
            # Format 2: object array with target_id
            if isinstance(first, dict):
                values = []
                for item in data:
                    tid = item.get('target_id', 0)
                    if wsg and tid > 0:
                        e = wsg.get_entity_by_id(tid)
                        if e and e.text:
                            v = _normalize_val(e.text)
                            values.append(v)
                if values:
                    return values
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _values_to_plan(values: list[str], wsg: WorldStateGraph) -> list[SubGoal]:
    """Convert button values to SubGoal list by matching WSG entities.

    The LLM outputs English values ("3", "+"), but WSG entities have
    Chinese text ("三", "加"). Maps using cn_map.
    """
    # English→Chinese mapping (reverse of cn_map in wsg.py)
    en_to_cn = {
        '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
        '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
        '+': '加', '-': '减', '*': '乘以', '/': '除以', '=': '等于',
    }
    plan = []
    text_to_id = {}
    for e in wsg.entities:
        if e.text and e.text not in text_to_id:
            text_to_id[e.text] = e.id

    last_input_id = 0
    for i, val in enumerate(values):
        # Handle "alt_tab" → action='key', no target needed
        if val == 'alt_tab':
            plan.append(SubGoal(
                step=i + 1, action='key', target_id=0,
                value='alt+tab', description='switch window',
            ))
            continue

        # Handle "type:..." → action='type', reuse last target_id
        if val.startswith('type:'):
            text = val[5:]
            plan.append(SubGoal(
                step=i + 1, action='type', target_id=last_input_id,
                value=text, description=f'type "{text}"',
            ))
            continue

        # Handle "text_area" → find an 'input' type entity
        if val == 'text_area':
            eid = 0
            for e in wsg.entities:
                if e.type == 'input':
                    eid = e.id
                    break
            if eid == 0:
                # fallback: any text-type entity
                for e in wsg.entities:
                    if e.type == 'text' and e.width > 100:
                        eid = e.id
                        break
            last_input_id = eid
            plan.append(SubGoal(
                step=i + 1, action='click', target_id=eid,
                value='', description='click text area',
            ))
            continue

        # Calculator: try direct match, then Chinese translation
        eid = text_to_id.get(val, 0)
        if eid == 0:
            cn = en_to_cn.get(val, val)
            eid = text_to_id.get(cn, 0)
        if eid == 0:
            for e in wsg.entities:
                if e.text and (val == e.text or val in e.text):
                    eid = e.id
                    break
        desc_map = {'+': 'plus', '-': 'minus', '*': 'multiply',
                    '/': 'divide', '=': 'equals'}
        desc = desc_map.get(val, f'digit {val}')
        plan.append(SubGoal(
            step=i + 1, action='click', target_id=eid,
            value='', description=f'click {desc}',
        ))
    return plan


# ─── Ollama backend ──────────────────────────────────────────────

class _OllamaBackend:
    def __init__(self, model="qwen2.5:7b"):
        self.model = model
        self._api = "http://localhost:11434/api/generate"
        self._check()

    def _check(self):
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            if not any(self.model in name for name in models):
                print(f"[WARN] Model '{self.model}' not found in Ollama")
        except Exception:
            print("[WARN] Ollama not running. Start: ollama serve")

    def generate(self, prompt: str, system: str) -> Optional[str]:
        import requests
        try:
            resp = requests.post(self._api, json={
                "model": self.model, "prompt": prompt, "system": system,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 2000},
            }, timeout=180)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            print(f"[ERROR] Ollama: {e}")
            return None


# ─── llama-cpp-python backend ────────────────────────────────────

class _LlamaBackend:
    def __init__(self, model_path: str, n_gpu_layers=-1, n_ctx=4096):
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self._llm = None

    def _load(self):
        if self._llm is not None:
            return
        from llama_cpp import Llama
        print(f"[LLAMA] Loading model from {self.model_path}...")
        print(f"[LLAMA] GPU layers: {self.n_gpu_layers}, Context: {self.n_ctx}")
        self._llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            verbose=False,
        )
        print(f"[LLAMA] Model loaded")

    def generate(self, prompt: str, system: str) -> Optional[str]:
        self._load()
        full_prompt = f"{system}\n\n{prompt}"
        try:
            output = self._llm(
                full_prompt,
                max_tokens=2000,
                temperature=0.1,
                stop=["\n\n\n"],
                echo=False,
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            print(f"[ERROR] llama.cpp: {e}")
            return None


# ─── Unified Planner ─────────────────────────────────────────────

class Planner:
    """LLM planner supporting both Ollama and llama-cpp-python backends.

    Args:
        backend: 'ollama' (default) or 'llama'
        model: model name for Ollama, or GGUF file path for llama
    """

    def __init__(self, backend="ollama", model=None, use_classifier=False):
        self.backend_name = backend
        if backend == "llama":
            model = model or "/d/models/qwen2.5-7b-instruct-q4_k_m.gguf"
            self._backend = _LlamaBackend(model)
        else:
            model = model or "qwen2.5:7b"
            self._backend = _OllamaBackend(model)

        # Phase 5: 影子模式分类器
        self.use_classifier = use_classifier
        self.classifier = None
        self.shadow_consistency = []  # 跟踪最近 100 次一致性
        if use_classifier:
            from agent.classifier import IntentClassifier
            self.classifier = IntentClassifier()
            if not self.classifier.load():
                print(f"[CLASSIFIER] No trained model found at {self.classifier.model_path}")
                print(f"[CLASSIFIER] Run tools/train_classifier.py first")
                self.use_classifier = False

    def plan_with_classifier(self, instruction: str, wsg: WorldStateGraph
                              ) -> Optional[list[SubGoal]]:
        """分类器优先（每类独立阈值），LLM 兜底。"""
        if not self.classifier or not self.use_classifier:
            return None

        from agent.classifier import CLASS_THRESHOLDS

        pred = self.classifier.predict(instruction)
        intent_name = pred['intent_name']
        threshold = CLASS_THRESHOLDS.get(intent_name, 0.8)

        if pred['confidence'] < threshold:
            print(f"  [CLASSIFIER] {intent_name} conf={pred['confidence']:.2f} < threshold={threshold} -> LLM fallback")
            return None

        slots = pred['slots']
        print(f"  [CLASSIFIER] intent={intent_name} conf={pred['confidence']:.2f} slots={slots} (threshold={threshold})")

        # 映射到 L3 模板 → 具体值
        if intent_name == 'binary_arithmetic':
            a = slots.get('A', '0')
            b = slots.get('B', '0')
            op = '?'
            for v in ['+','-','*','/']:
                if v in instruction:
                    op = v
                    break
            values = list(a) + [op] + list(b) + ['=']
        elif intent_name == 'text_input':
            text = slots.get('VALUE', '')
            values = ['text_area', f'type:{text}']
        else:
            return None

        return _values_to_plan(values, wsg)

    def generate(self, instruction: str, available_values: str) -> Optional[str]:
        prompt = (
            f"## User instruction\n{instruction}\n\n"
            f"## Available button values\n{available_values}\n\n"
            f"Output the button sequence as a JSON list."
        )
        return self._backend.generate(prompt, SYSTEM_PROMPT)

    def plan(self, instruction: str, wsg: WorldStateGraph
             ) -> Optional[list[SubGoal]]:
        # Phase 5: 影子模式 — 分类器优先尝试
        classifier_plan = self.plan_with_classifier(instruction, wsg)
        if classifier_plan is not None:
            return classifier_plan

        # LLM 兜底
        cn_rev = {'零':'0','一':'1','二':'2','三':'3','四':'4','五':'5',
                  '六':'6','七':'7','八':'8','九':'9','加':'+','减':'-',
                  '乘以':'*','除以':'/','等于':'='}
        available = []
        for e in wsg.entities:
            if e.type != 'button':
                continue
            if e.text in cn_rev:
                available.append(cn_rev[e.text])
            elif e.text in '0123456789+-*/=C':
                available.append(e.text)
        response = self.generate(instruction, json.dumps(sorted(set(available))))
        values = _parse_value_sequence(response, wsg)
        if not values:
            return None
        return _values_to_plan(values, wsg)

    def fix_plan(self, instruction: str, wsg: WorldStateGraph,
                 error_message: str) -> Optional[list[SubGoal]]:
        available = sorted(set(e.text for e in wsg.entities if e.text))
        prompt = (
            f"## Instruction\n{instruction}\n\n"
            f"## Available button values\n{json.dumps(list(available))}\n\n"
            f"## Previous plan had errors\n{error_message}\n\n"
            f"Output the correct button sequence as a JSON list."
        )
        response = self._backend.generate(prompt, SYSTEM_PROMPT)
        values = _parse_value_sequence(response, wsg)
        if values:
            return _values_to_plan(values, wsg)
        return None
