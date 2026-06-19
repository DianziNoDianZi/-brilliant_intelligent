"""World State Graph — structured representation of a desktop UI state."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import numpy as np


ENTITY_TYPES = {'text', 'button', 'input', 'icon', 'label', 'window'}
VALID_ACTIONS = {
    'text': ['click', 'type'],
    'button': ['click'],
    'input': ['click', 'type'],
    'icon': ['click'],
    'label': [],
    'window': ['click'],
}


@dataclass
class WSGEntity:
    id: int
    type: str
    text: str = ''
    bbox: list = field(default_factory=lambda: [0, 0, 0, 0])  # x1,y1,x2,y2
    properties: dict = field(default_factory=dict)

    @property
    def center(self) -> tuple:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> dict:
        d = {'id': self.id, 'type': self.type, 'text': self.text,
             'bbox': self.bbox}
        if self.properties and 'label' in self.properties:
            d['label'] = self.properties['label']
        return d


@dataclass
class SubGoal:
    step: int
    action: str          # click | type | tab | enter | wait
    target_id: int
    value: str = ''
    description: str = ''

    def to_dict(self) -> dict:
        return {'step': self.step, 'action': self.action,
                'target_id': self.target_id, 'value': self.value,
                'description': self.description}


class WorldStateGraph:
    """Structured representation of a desktop screenshot's UI state."""

    def __init__(self, entities: list = None, relations: list = None,
                 screenshot: np.ndarray = None, window_offset=(0, 0)):
        self.entities: list[WSGEntity] = entities or []
        self.relations: list[dict] = relations or []
        self.screenshot: Optional[np.ndarray] = screenshot
        self.window_offset = window_offset  # (offset_x, offset_y) on screen

    def get_entity_by_id(self, eid: int) -> Optional[WSGEntity]:
        for e in self.entities:
            if e.id == eid:
                return e
        return None

    def get_entities_by_type(self, etype: str) -> list[WSGEntity]:
        return [e for e in self.entities if e.type == etype]

    def get_entity_by_text(self, text: str, fuzzy=True) -> Optional[WSGEntity]:
        text_lower = text.lower()
        for e in self.entities:
            if fuzzy and text_lower in e.text.lower():
                return e
            if e.text == text:
                return e
        return None

    def get_entity_by_position(self, x: int, y: int) -> Optional[WSGEntity]:
        """Find smallest entity containing the given point."""
        containing = []
        for e in self.entities:
            x1, y1, x2, y2 = e.bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                containing.append((area, e))
        if containing:
            return min(containing, key=lambda t: t[0])[1]
        return None

    def screenshot_to_screen(self, sx: int, sy: int) -> tuple:
        """Convert screenshot coordinates to screen coordinates."""
        ox, oy = self.window_offset
        return (sx + ox, sy + oy)

    def serialize_for_llm(self, max_entities=50) -> str:
        """Filtered JSON for LLM — keeps only interactive digit/operator buttons.

        Chinese calculator buttons are annotated with English: '一 (1)', '加 (+)'.
        Excludes scientific functions, window chrome, and decorative text."""
        cn_map = {
            '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
            '六': '6', '七': '7', '八': '8', '九': '9', '零': '0',
            '加': '+', '减': '-', '乘以': '*', '除以': '/', '等于': '=',
            '清除': 'C', 'Backspace': '⌫', '正负': '±',
        }
        filtered = []
        for e in self.entities:
            if e.bbox[1] < 60:
                continue
            if e.width < 30 or e.height < 20:
                continue
            # Calculator buttons: map Chinese to English for LLM output
            anno = cn_map.get(e.text.strip(), '')
            if anno:
                english_text = anno.replace('⌫', 'BKSP').replace('±', '+-')
                d = {'id': e.id, 'type': e.type, 'text': english_text, 'bbox': e.bbox}
                filtered.append(d)
                continue
            # Notepad: keep input/text areas
            elif e.type in ('input', 'text') and e.text.strip():
                if e.type == 'text' and (e.width > 200 or e.height < 30):
                    continue
                if e.width < 40 or e.height < 20:
                    continue
                filtered.append(e.to_dict())

        filtered.sort(key=lambda d: (d['bbox'][1], d['bbox'][0]))
        return json.dumps({'entities': filtered}, ensure_ascii=False, indent=2)

    def add_entity(self, entity: WSGEntity):
        self.entities.append(entity)

    def remove_entity(self, eid: int):
        self.entities = [e for e in self.entities if e.id != eid]

    def compute_spatial_relations(self):
        """Auto-compute spatial relations (above/below/left_of/right_of)."""
        self.relations = []
        for a in self.entities:
            for b in self.entities:
                if a.id >= b.id:
                    continue
                ax1, ay1, ax2, ay2 = a.bbox
                bx1, by1, bx2, by2 = b.bbox
                a_cx, a_cy = a.center
                b_cx, b_cy = b.center

                # Horizontal overlap check
                h_overlap = min(ax2, bx2) - max(ax1, bx1) > 0
                v_overlap = min(ay2, by2) - max(ay1, by1) > 0

                # Vertical relationship
                if h_overlap and a_cy < b_cy:
                    self.relations.append(
                        {'source_id': a.id, 'target_id': b.id, 'type': 'above'})
                elif h_overlap and a_cy > b_cy:
                    self.relations.append(
                        {'source_id': a.id, 'target_id': b.id, 'type': 'below'})

                # Horizontal relationship
                if v_overlap and a_cx < b_cx:
                    self.relations.append(
                        {'source_id': a.id, 'target_id': b.id, 'type': 'left_of'})
                elif v_overlap and a_cx > b_cx:
                    self.relations.append(
                        {'source_id': a.id, 'target_id': b.id, 'type': 'right_of'})

    def __str__(self) -> str:
        return f"WSG({len(self.entities)} entities, {len(self.relations)} relations)"


def action_is_valid_for_entity(action: str, entity: WSGEntity) -> bool:
    """Check if an action type is valid for a given entity type."""
    allowed = VALID_ACTIONS.get(entity.type, [])
    return action in allowed
