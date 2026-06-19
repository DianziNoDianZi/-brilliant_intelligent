"""Intent vector — decodes operation structure from operand values.

Key design: operation pattern (operator, operand count) is encoded
SEPARATELY from specific operand values. This makes "3+4" and "5+8"
map to the same operation-cluster in vector space — the only
difference is the operand value slots.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


INTENT_FEAT_DIM = 24


@dataclass
class IntentVector:
    task_type: str = "generic"
    values: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    skill_name: str = ""
    raw_instruction: str = ""

    def encode(self) -> np.ndarray:
        """Encode as [task_type(3) + operator(4) + structure(3) + operand_values(14)].

        The first 10 dims encode the OPERATION PATTERN (same for all additions).
        The remaining dims encode SPECIFIC OPERAND VALUES.
        """
        feat = np.zeros(INTENT_FEAT_DIM, dtype=np.float32)

        # ── Pattern part: operation structure, NOT operand values ──
        # dim 0: has_operator (shared by all calculator ops)
        ops = ['+', '-', '*', '/']
        feat[0] = 1.0 if any(v in ops for v in self.values) else 0.0

        # dim 1-4: operator one-hot WEIGHTED x3 for separation
        op_found = False
        for v in self.values:
            if v in ops:
                feat[1 + ops.index(v)] = 3.0
                op_found = True
                break
        if not op_found:
            feat[5] = 1.0  # no operator

        # dim 6: has equals
        feat[6] = 1.0 if '=' in self.values else 0.0

        # dim 7-8: operand count (1, 2, 3+)
        operands = [v for v in self.values if v.isdigit()]
        count = min(len(operands), 3)
        if count > 0:
            feat[6 + count] = 1.0

        # dim 9: is_notepad_action (for distinguishing calculator vs notepad)
        feat[9] = 1.0 if self.task_type == 'notepad' else 0.0

        # ── Operand part (specific values, changes per invocation) ──
        # dim 10-23: operand values + metadata
        for i, v in enumerate(operands[:9]):
            try:
                feat[10 + i] = int(v) / 9.0
            except ValueError:
                pass
        # Multi-digit flag
        multi = [v for v in operands if len(v) > 1]
        feat[23] = min(len(multi) / 3.0, 1.0)

        return feat

    @staticmethod
    def from_values(values: list[str], instruction: str = "",
                    task_type: str = "calculator") -> IntentVector:
        return IntentVector(
            task_type=task_type, values=values,
            raw_instruction=instruction,
        )

    @staticmethod
    def get_feature_dim() -> int:
        return INTENT_FEAT_DIM


def detect_task_type(instruction: str) -> str:
    il = instruction.lower()
    if any(w in il for w in ['calculator', 'calc', 'calculate', '计算',
                              '加', '减', '乘', '除']):
        return 'calculator'
    if any(w in il for w in ['notepad', '记事本', '输入', 'type', 'write']):
        return 'notepad'
    return 'generic'
