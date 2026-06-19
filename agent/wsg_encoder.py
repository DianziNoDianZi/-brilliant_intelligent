"""Encode WSG (variable-length entity list) into fixed-size feature vectors.

Used by the world model for inner simulation.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from agent.wsg import WorldStateGraph, SubGoal

MAX_ENTITIES = 50
TYPE_VOCAB = ['text', 'button', 'input', 'icon', 'label']
ACTION_VOCAB = ['click', 'type', 'tab', 'enter', 'wait']

# Per-entity feature dimensions
# type(5) + center(2) + size(2) + has_text(1) + text_len(1) = 11
ENTITY_FEAT_DIM = len(TYPE_VOCAB) + 4 + 2
STATE_DIM = MAX_ENTITIES * ENTITY_FEAT_DIM

# Action feature dimensions
# type_onehot(5) + target_cx(1) + target_cy(1) + has_value(1) = 8
ACTION_FEAT_DIM = len(ACTION_VOCAB) + 3


def encode_wsg(wsg: WorldStateGraph) -> np.ndarray:
    """Encode WSG entities into a fixed-size feature vector.

    Returns: (STATE_DIM,) float32 array, values in [0, 1].
    """
    features = np.zeros(STATE_DIM, dtype=np.float32)
    h_img = 1.0
    w_img = 1.0
    if wsg.screenshot is not None:
        h_img, w_img = wsg.screenshot.shape[:2]

    for i, entity in enumerate(wsg.entities[:MAX_ENTITIES]):
        offset = i * ENTITY_FEAT_DIM
        # One-hot type
        type_idx = TYPE_VOCAB.index(entity.type) if entity.type in TYPE_VOCAB else 0
        features[offset + type_idx] = 1.0
        # Normalized center and size
        x1, y1, x2, y2 = entity.bbox
        features[offset + 5] = ((x1 + x2) / 2) / max(w_img, 1)
        features[offset + 6] = ((y1 + y2) / 2) / max(h_img, 1)
        features[offset + 7] = (x2 - x1) / max(w_img, 1)
        features[offset + 8] = (y2 - y1) / max(h_img, 1)
        # Text features
        features[offset + 9] = 1.0 if entity.text else 0.0
        features[offset + 10] = min(len(entity.text) / 100.0, 1.0)

    return features


def encode_action(step: SubGoal, wsg: WorldStateGraph) -> np.ndarray:
    """Encode a sub-goal (action + target) into a feature vector.

    Returns: (ACTION_FEAT_DIM,) float32 array.
    """
    features = np.zeros(ACTION_FEAT_DIM, dtype=np.float32)
    act_idx = ACTION_VOCAB.index(step.action) if step.action in ACTION_VOCAB else 0
    features[act_idx] = 1.0

    # Target position (reuse from WSG if available)
    entity = wsg.get_entity_by_id(step.target_id)
    if entity:
        cx, cy = entity.center
        h_img = max(wsg.screenshot.shape[0], 1) if wsg.screenshot is not None else 1
        w_img = max(wsg.screenshot.shape[1], 1) if wsg.screenshot is not None else 1
        features[5] = cx / w_img
        features[6] = cy / h_img

    features[7] = 1.0 if step.value else 0.0
    return features


def decode_predicted_changes(pred_delta: np.ndarray,
                             original_wsg: WorldStateGraph) -> dict:
    """Decode predicted state delta back to human-readable changes.

    Returns dict with lists of entities likely to change.
    """
    changes = []
    h_img = 1.0
    w_img = 1.0
    if original_wsg.screenshot is not None:
        h_img, w_img = original_wsg.screenshot.shape[:2]

    for i, entity in enumerate(original_wsg.entities[:MAX_ENTITIES]):
        offset = i * ENTITY_FEAT_DIM
        delta_magnitude = np.mean(np.abs(pred_delta[offset:offset + ENTITY_FEAT_DIM]))
        if delta_magnitude > 0.05:
            changes.append({
                'entity_id': entity.id,
                'text': entity.text,
                'change_score': float(delta_magnitude),
            })

    return {
        'changed_entities': changes,
        'total_delta_magnitude': float(np.mean(np.abs(pred_delta))),
    }


def combine_state_and_action(state_vec: np.ndarray,
                              action_vec: np.ndarray) -> np.ndarray:
    """Combine encoded state and action into a single model input."""
    return np.concatenate([state_vec, action_vec]).astype(np.float32)
