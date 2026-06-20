"""意图日志记录。

每次成功执行后自动记录 (指令, L3模板ID, 槽位值) 到 data/intent_log.jsonl。
"""

import json, os, re, time
from typing import Optional

LOG_PATH = "D:/briliant_intelligent/data/intent_log.jsonl"


def ensure_dir():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def log_execution(instruction: str, plan_values: list[str],
                  l3_template_id: str = "", confidence: float = 0.0,
                  success: bool = True):
    """记录一次成功执行。

    Args:
        instruction: 原始指令 "calculate 3+4"
        plan_values: 执行的值序列 ["3","+","4","="]
        l3_template_id: 匹配的 L3 抽象模板 ID
        confidence: 快回路置信度
        success: 是否成功
    """
    ensure_dir()
    slots = extract_slots(instruction, plan_values)
    record = {
        "timestamp": time.time(),
        "instruction": instruction,
        "plan_values": plan_values,
        "l3_template_id": l3_template_id,
        "slots": slots,
        "confidence": round(confidence, 3),
        "success": success,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_slots(instruction: str, plan_values: list[str]) -> dict:
    """从指令和值序列中提取槽位。

    "calculate 3+4" + ["3","+","4","="] → {"A": "3", "B": "4"}
    "type Hello World" + ["text_area","type:Hello World"] → {"VALUE": "Hello World"}
    """
    slots = {}
    numbers = re.findall(r'\d+', instruction)
    num_idx = 0
    for v in plan_values:
        if v.isdigit() and num_idx < len(numbers):
            slot_key = chr(65 + num_idx)  # A, B, C...
            slots[slot_key] = v
            num_idx += 1
    # Handle type:VALUE for notepad
    type_vals = [v for v in plan_values if v.startswith('type:')]
    if type_vals:
        slots["VALUE"] = type_vals[-1][5:]
    return slots


def load_log(limit: int = 0) -> list[dict]:
    """加载所有历史记录。"""
    if not os.path.exists(LOG_PATH):
        return []
    records = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if limit > 0:
        records = records[-limit:]
    return records


def get_intent_label(record: dict, l3_templates: list[str]) -> int:
    """将记录映射到 L3 模板 ID。

    如果记录没有 l3_template_id，尝试从 plan_values 推断。
    """
    if record.get("l3_template_id"):
        tid = record["l3_template_id"]
        for i, t in enumerate(l3_templates):
            if tid == t:
                return i
    # Fallback: infer from plan_values
    vals = record.get("plan_values", [])
    if any(v in '+-*/' for v in vals):
        return 0  # binary_arithmetic
    if 'text_area' in vals or any(v.startswith('type:') for v in vals):
        return 1  # text_input
    return 0
