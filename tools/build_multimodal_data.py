"""从 intent_log.jsonl 生成多模态训练数据。

每条日志 → (wsg_before_encoded, 指令文本, 动作编码, wsg_after_encoded)
"""

import sys, os, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from agent.wsg_encoder import encode_wsg, STATE_DIM
from agent.wsg import WorldStateGraph, WSGEntity
from agent.intent_log import load_log
from agent.classifier import tokenize, VOCAB_SIZE


def make_calc_wsg(display: str = "0") -> WorldStateGraph:
    """Calculator WSG with display text."""
    wsg = WorldStateGraph()
    for txt, x in [('1',0),('2',30),('3',60),('4',90),('5',120),
                    ('6',150),('7',180),('8',210),('9',240),('0',270),
                    ('+',300),('-',330),('*',360),('/',390),('=',420),
                    ('C',450)]:
        wsg.add_entity(WSGEntity(id=len(wsg.entities)+1, type='button',
                                  text=txt, bbox=[x,0,x+25,25]))
    # Display area
    wsg.add_entity(WSGEntity(id=99, type='text', text=display,
                              bbox=[0,50,500,80],
                              properties={'label': 'display'}))
    wsg.compute_spatial_relations()
    return wsg


def make_notepad_wsg(text: str = "") -> WorldStateGraph:
    """Notepad WSG with text content."""
    wsg = WorldStateGraph()
    wsg.add_entity(WSGEntity(id=1, type='input', text='text_area',
                              bbox=[0,30,500,300]))
    wsg.add_entity(WSGEntity(id=2, type='text', text=text or '',
                              bbox=[10,40,490,290]))
    wsg.add_entity(WSGEntity(id=3, type='button', text='file_menu',
                              bbox=[0,0,50,25]))
    wsg.add_entity(WSGEntity(id=4, type='button', text='save',
                              bbox=[0,25,100,50]))
    wsg.compute_spatial_relations()
    return wsg


def simulate_calc(vals: list[str], prev_display="0"):
    """Simulate calculator execution: return new display text."""
    ops = {'+': lambda a,b: a+b, '-': lambda a,b: a-b,
           '*': lambda a,b: a*b, '/': lambda a,b: a//b if b!=0 else 0}
    if '=' in vals:
        eq = vals.index('=')
        expr = ''.join(vals[:eq])
        for op_char in '+-*/':
            if op_char in expr:
                parts = expr.split(op_char)
                if len(parts) >= 2:
                    try:
                        a, b = int(parts[0]), int(parts[1])
                        result = ops.get(op_char, lambda x,y:0)(a, b)
                        return str(result)
                    except: pass
    return prev_display


def simulate_notepad(vals: list[str]):
    """Extract typed text from notepad action sequence."""
    for v in vals:
        if v.startswith('type:'):
            return v[5:]
    return ""


def build_dataset():
    """主函数：从日志生成训练四元组。"""
    records = load_log()
    print(f"[DATA] Loaded {len(records)} records")

    data = []
    for r in records:
        instruction = r.get('instruction', '')
        vals = r.get('plan_values', [])
        if not instruction or not vals:
            continue

        # Detect app type from values
        is_calc = any(v in '0123456789+-*/=' for v in vals)

        if is_calc:
            wsg_before = make_calc_wsg("0")
            new_display = simulate_calc(vals, "0")
            wsg_after = make_calc_wsg(new_display)
        else:
            wsg_before = make_notepad_wsg("")
            text = simulate_notepad(vals)
            wsg_after = make_notepad_wsg(text)

        before_vec = encode_wsg(wsg_before)
        after_vec = encode_wsg(wsg_after)

        # Action encoding (same as wsgencoder)
        act_types = ['click', 'type', 'tab', 'enter', 'wait']
        action_vec = np.zeros(len(act_types), dtype=np.float32)
        for v in vals:
            if v.startswith('type:'):
                action_vec[1] = 1.0  # type
                break
        else:
            action_vec[0] = 1.0  # click

        data.append({
            'instruction': instruction,
            'feat_before': before_vec.tolist(),
            'feat_after': after_vec.tolist(),
            'action_vec': action_vec.tolist(),
            'tokens': tokenize(instruction),
            'app': 'calculator' if is_calc else 'notepad',
        })

    print(f"[DATA] Generated {len(data)} training tuples")
    print(f"  calculator: {sum(1 for d in data if d['app']=='calculator')}")
    print(f"  notepad: {sum(1 for d in data if d['app']=='notepad')}")

    os.makedirs("D:/briliant_intelligent/data", exist_ok=True)
    with open("D:/briliant_intelligent/data/multimodal_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"[DATA] Saved to data/multimodal_data.json")

    # Build text vocab
    all_tokens = []
    for d in data:
        all_tokens.extend(d['tokens'])
    from collections import Counter
    vocab = [w for w, _ in Counter(all_tokens).most_common(VOCAB_SIZE)]
    with open("D:/briliant_intelligent/data/text_vocab.json", "w") as f:
        json.dump(vocab, f)
    print(f"[DATA] Vocab: {len(vocab)} words")

    return data


if __name__ == '__main__':
    build_dataset()
