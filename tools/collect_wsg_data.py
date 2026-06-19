"""收集真实桌面WSG变化数据。

运行: python tools/collect_wsg_data.py
输出: /d/tmp/wsg_transitions.json (WSG_before, action, WSG_after 序列)
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environment.desktop_env import DesktopEnv
from agent.wsg import WorldStateGraph, SubGoal
from agent.wsg_encoder import encode_wsg


def collect_sequence(env, actions: list[tuple], save_path: str):
    """Execute action sequence and record WSG transitions."""
    transitions = []
    wsg_before = env.reset()

    for action_name, action_args in actions:
        # Record before
        feat_before = encode_wsg(wsg_before).tolist()

        # Execute
        wsg_after = env.step(action_name, **action_args)

        # Record after
        feat_after = encode_wsg(wsg_after).tolist()

        transitions.append({
            'action': action_name,
            'target': action_args,
            'entity_count_before': len(wsg_before.entities),
            'entity_count_after': len(wsg_after.entities),
            'display_text_before': _get_display_text(wsg_before),
            'display_text_after': _get_display_text(wsg_after),
        })
        wsg_before = wsg_after

    with open(save_path, 'w') as f:
        json.dump(transitions, f, indent=2, ensure_ascii=False)
    print(f"[DATA] Saved {len(transitions)} transitions to {save_path}")
    print(f"[DATA] Actions: {[t['action'] for t in transitions]}")


def _get_display_text(wsg: WorldStateGraph) -> str:
    """Find calculator display text."""
    for e in wsg.entities:
        if e.type == 'text' and e.text.strip().isdigit() and e.width > 100:
            return e.text.strip()
    return ''


def collect_calculator_ops(env: DesktopEnv):
    """Run a standard calculator operation sequence."""
    import pyautogui

    ops = [
        ('click', {'x': 0, 'y': 0}),  # placeholder
    ]
    # Replace placeholder with actual entity positions
    wsg = env.reset()
    one = next((e for e in wsg.entities if e.text in ('一', '1') and e.type == 'button'), None)
    plus = next((e for e in wsg.entities if '加' in e.text or e.text == '+'), None)
    two = next((e for e in wsg.entities if e.text in ('二', '2') and e.type == 'button'), None)
    equals = next((e for e in wsg.entities if '等于' in e.text or e.text == '='), None)

    if not all([one, plus, two, equals]):
        print("[ERROR] Calculator buttons not found")
        return []

    actions = [
        ('click', {'x': one.center[0] + wsg.window_offset[0],
                    'y': one.center[1] + wsg.window_offset[1]}),
        ('click', {'x': plus.center[0] + wsg.window_offset[0],
                    'y': plus.center[1] + wsg.window_offset[1]}),
        ('click', {'x': two.center[0] + wsg.window_offset[0],
                    'y': two.center[1] + wsg.window_offset[1]}),
        ('click', {'x': equals.center[0] + wsg.window_offset[0],
                    'y': equals.center[1] + wsg.window_offset[1]}),
    ]
    return actions


if __name__ == '__main__':
    env = DesktopEnv(app_name='calculator')
    actions = collect_calculator_ops(env)
    if actions:
        collect_sequence(env, actions, '/d/tmp/wsg_transitions.json')
    env.close()
